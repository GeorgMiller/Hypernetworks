import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

import keras
from keras.layers import Dense, Input, Flatten, Dropout, LeakyReLU
from keras.optimizers import Adam
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.ops.gen_array_ops import quantize_v2_eager_fallback

from utils_global import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


class DQNAgent(object):

    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=2000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=2000,
                 batch_size=32,
                 action_num=309,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=5e-5,
                 decay_rate = 10):

        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.train_every = train_every
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.state_shape = state_shape

        self.gamma = 0.9
        self.use_raw = False
        self.loss = 0
        self.q_values = []

        self.total_t = 0
        self.train_t = 0

        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.online_network = Network(action_num= self.action_num)
        self.target_network = Network(action_num= self.action_num)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate, decay_steps=self.batch_size, decay_rate=self.decay_rate)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipvalue = 1)
        self.memory = Memory(replay_memory_size, batch_size)


    def feed(self, ts):

        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], done, state['legal_actions'])
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

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
        #q_max = np.amax(q_values)
        #q_min = np.amin(q_values)
        #A = (q_values - q_min)/ (q_max -q_min)
        q_values[best_action] += (1.0 - epsilon)
        A = tf.nn.softmax(q_values).numpy()
        return A

    def train(self):
      
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_action_batch = self.memory.sample()
        
        with tf.GradientTape() as tape:

            self.q_values = self.online_network(state_batch)
            q_pred_next = self.online_network(next_state_batch).numpy()
            q_next = self.target_network(next_state_batch).numpy()

            legal_actions = [[] for _ in range(self.batch_size)]

            for idx in range(self.batch_size):
                if len(legal_action_batch) == 0:
                    legal_actions[idx] = q_pred_next[idx]
                else:
                    masked_q_values = np.ones(self.action_num) * (-float('inf'))
                    masked_q_values[legal_action_batch[idx]] = q_pred_next[idx][legal_action_batch[idx]]
                    legal_actions[idx] = masked_q_values
                    legal_actions[idx][308] = -1

            '''
            for idx in range(self.batch_size):
                if len(legal_action_batch[idx]) == 0:
                    legal_actions[idx] = q_pred_next[idx]
                elif legal_action_batch[idx] == list([308]):
                    legal_actions[idx] = q_pred_next[idx]
                else:
                    legal_actions[idx] = remove_illegal(q_pred_next[idx], legal_action_batch[idx])
            '''
            
            best_actions = tf.math.argmax(legal_actions, axis =1)

            q_target = reward_batch + self.gamma * np.invert(done_batch).astype(np.float32) * q_next[np.arange(self.batch_size), best_actions] #* q_next[np.arange(self.batch_size), best_actions] 
            #q_target = self.rewards + self.gamma * np.invert(self.dones).astype(np.float32) * q_next[index, best_actions] #* q_next[np.arange(self.batch_size), best_actions] 

            action_one_hot = tf.one_hot(action_batch, self.action_num, on_value=1,off_value=0).numpy().astype(np.float32)
            values = tf.reduce_sum(tf.multiply(self.q_values, action_one_hot), axis=1)
        
            value_loss = tf.reduce_mean(tf.math.square(values - q_target))
            grads = tape.gradient(value_loss, self.online_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.online_network.trainable_variables))

            self.loss = value_loss
            #print(self.loss)
        
        if self.train_t % self.update_target_estimator_every == 0:
            
            self.target_network.set_weights(self.online_network.get_weights())
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

    def feed_memory(self, state, action, reward, next_state, done, legal_actions):
  
        self.memory.save(state, action, reward, next_state, done, legal_actions)


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

