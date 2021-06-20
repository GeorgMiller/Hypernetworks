import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

import keras
from keras.layers import Dense, Input, Flatten, Dropout, LeakyReLU
from keras.optimizers import Adam

from utils_global import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


class DQNAgent(object):

    def __init__(self,
                 replay_memory_size=2000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=2000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 action_num=309,
                 train_every=1,
                 decay_rate = 10):

        self.use_raw = False


        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.train_every = train_every
        self.decay_rate = decay_rate

        self.gamma = 0.9
        self.hidden = 512

        self.total_t = 0
        self.train_t = 0
        self.training_steps = 1000

        self.input_noise_size = 4
        self.seed = 42
        self.input_shape = [9,5,15]

        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.online_network = Network(action_num= self.action_num)
        self.target_network = Network(action_num= self.action_num)
        self.memories = [Memory(replay_memory_size, batch_size) for _ in range(self.batch_size)]
        
    def feed(self, ts, thread):

        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], done, state['legal_actions'], thread)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size


    def step(self, state):

        A = self.predict(state['obs'])
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
 
        return action

    def eval_step(self, state):

        q_values = self.online_network(np.expand_dims(state['obs'], 0))[0]
        probs = remove_illegal(np.exp(q_values), state['legal_actions'])
        best_action = np.argmax(probs)
        return best_action, probs

    def predict(self, state):

        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        q_values = self.online_network(np.expand_dims(state, 0))[0].numpy()
        best_action = np.argmax(q_values)
        q_values[best_action] += (1.0 - epsilon)
        A = tf.nn.softmax(q_values).numpy()
        return A

    
    def update_weights(self, thread):

        # Get the states from the memory depending on the created worker
        self.states, self.actions, self.rewards, self.next_states, self.dones, self.legal_actions = self.memories[thread].sample()
        
        q_pred_next = self.online_network(self.next_states).numpy()
        q_next = self.target_network(self.next_states).numpy()

        legal_actions = [[] for i in range(self.batch_size)]
        for idx in range(self.batch_size):
            if len(self.legal_actions) == 0:
                legal_actions[idx] = q_pred_next[idx]
            else:
                masked_q_values = np.ones(self.action_num) * (-float('inf'))
                masked_q_values[self.legal_actions[idx]] = q_pred_next[idx][self.legal_actions[idx]]
                legal_actions[idx] = masked_q_values
                legal_actions[idx][308] = -1

        best_actions = tf.math.argmax(legal_actions, axis =1).numpy()
        
        self.dones = np.reshape(self.dones, -1)
        index = np.arange(0,len(self.dones))   
        self.rewards = tf.reshape(self.rewards,-1)
        q_target = self.rewards + self.gamma * np.invert(self.dones).astype(np.float32) * q_next[index, best_actions] #* q_next[np.arange(self.batch_size), best_actions] 

        values = self.online_network(self.states)
        pred = tf.argmax(values, axis=1)
        actions_one_hot = tf.one_hot(self.actions, self.action_num)
        values_v = tf.reduce_sum(tf.multiply(values, actions_one_hot), axis=1)
    
        self.value_loss = tf.reduce_mean(tf.math.square(values_v - q_target.numpy()))

        # Total loss
        return self.value_loss


    def set_weights(self, weights_online_network, num):
        
        # This part is used to set the weights for the Actor
        last_used = 0
        weights = weights_online_network[num]
        for i in range(len(self.online_network.layers)):
            if 'conv' in self.online_network.layers[i].name or  'dense' in self.online_network.layers[i].name: 
                weights_shape = self.online_network.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                self.online_network.layers[i].kernel = new_weights
                last_used += no_of_weights
                
                if self.online_network.layers[i].use_bias:
                    weights_shape = self.online_network.layers[i].bias.shape
                    no_of_weights = tf.reduce_prod(weights_shape)
                    new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                    self.online_network.layers[i].bias = new_weights
                    last_used += no_of_weights

    def set_target_weights(self, weights_target_network, num):
        
        # This part is used to update the weights of the target network
        # TODO: merge this with the set_weights() function above
        last_used = 0
        weights = weights_target_network[num]
        for i in range(len(self.target_network.layers)):
            if 'conv' in self.target_network.layers[i].name or  'dense' in self.target_network.layers[i].name: 
                weights_shape = self.target_network.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                self.target_network.layers[i].kernel = new_weights
                last_used += no_of_weights
                
                if self.target_network.layers[i].use_bias:
                    weights_shape = self.target_network.layers[i].bias.shape
                    no_of_weights = tf.reduce_prod(weights_shape)
                    new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                    self.target_network.layers[i].bias = new_weights
                    last_used += no_of_weights

    def feed_memory(self, state, action, reward, next_state, done, legal_actions, thread):
  
        self.memories[thread].save(state, action, reward, next_state, done, legal_actions)

    
class Network(tf.keras.Model):

    def __init__(self, action_num):
        super().__init__()

        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(512, activation='relu')
        self.value = Dense(action_num, activation='linear')
    
    def call(self, inputs):

        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.value(x)

        return x
        
class Memory(object):

    def __init__(self, memory_size, batch_size):
 
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done, legal_actions):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def sample(self):

        samples = random.sample(self.memory, self.batch_size)

        return map(np.array, zip(*samples))



