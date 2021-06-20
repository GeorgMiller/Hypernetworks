import tensorflow as tf     
import keras 
import numpy as np
import random
from keras.layers import Dense, Input, Flatten, BatchNormalization,Dropout
from keras.optimizers import Adam
from collections import namedtuple
from utils_global import remove_illegal


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

class Actor_Critic():
    
    def __init__(self,
                 replay_memory_size=200,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=64,
                 action_num=309,
                 state_shape=[9,5,15],
                 train_every=256,
                 mlp_layers=None,
                 initial_learning_rate=5e-5, 
                 decay_steps=20, 
                 decay_rate=0.99998):
        
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.train_every = train_every
        self.state_shape = state_shape
        self.replay_memory_size = replay_memory_size
        

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0
        self.zero_fixer = 1e-9
        self.epochs = 5
        self.mini_batch_size = 4

        # The epsilon decay schedulers
        # TODO: Doesnt really matter, but remove it for A2C
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        self.memory = Memory(self.replay_memory_size, self.batch_size)

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.initial_learning_rate, self.decay_steps, self.decay_rate)
        lr_schedule_v = tf.keras.optimizers.schedules.ExponentialDecay(self.initial_learning_rate, self.decay_steps, self.decay_rate)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.optimizer_v = tf.keras.optimizers.Adam(learning_rate=lr_schedule_v)
        self.critic = self.create_critic(1, self.state_shape)
        self.actor = self.create_actor(self.action_num, self.state_shape)
        self.history_actor = 0
        self.history_critic = 0
        self.actions = 0
        self.predictions  =0
        self.loss = 0
        
    def feed(self, ts):

        (state, action, reward, next_state, done) = tuple(ts)
        self.total_t += 1
        
        if not done:
            self.feed_memory(state['obs'], action, reward, next_state['obs'], done, state['legal_actions'])
        
        else:
            self.feed_memory(state['obs'], action, reward, next_state['obs'], done, state['legal_actions'] )
            if self.total_t>=200: 
                self.train()

    def step(self, state):

        A = self.predict(state['obs'].astype(np.float32))
        #print(A,'first')
        A = remove_illegal(A, state['legal_actions'])
        #print(A,'second')
        action = np.random.choice(np.arange(len(A)), p=A)
        
        return action

    def eval_step(self, state):

        prediction = self.actor(np.expand_dims(state['obs'].astype(np.float32), 0))[0]
        probs = remove_illegal(np.exp(prediction), state['legal_actions'])
        best_action = np.argmax(probs)
        
        return best_action, probs

    def predict(self, state):

        prediction = self.actor(np.expand_dims(state.astype(np.float32),0))[0].numpy()

        return prediction 
        
    def train(self):
        
        self.states, self.actions, self.rewards, self.next_states, self.dones, legal_actions = self.memory.sample()
        
        self.rewards = self.discounted_rewards(self.rewards)

        best_action = tf.argmax(self.actor(self.states), axis=1)

        for i in range(self.epochs):

            index = np.arange(len(self.rewards))
            np.random.shuffle(index)
            step_size = len(self.rewards)// self.mini_batch_size

            for start in range(0,len(self.rewards), step_size):
                
                end = start + step_size
                idx = index[start:end]

                states = self.states[idx]
                actions = self.actions[idx]
                rewards = self.rewards[idx]
                next_state = self.next_states[idx]
                done = self.dones[idx]
        
                with tf.GradientTape() as tape_v:

                    values = tf.reshape(self.critic(states),-1)

                    value_loss = tf.math.reduce_mean(tf.math.square(values-rewards))
                    grads = tape_v.gradient(value_loss, self.critic.trainable_variables)
                    self.optimizer_v.apply_gradients(zip(grads,self.critic.trainable_variables))
                
                values_next = self.critic(next_state)
                
                gamma = 0.95
                
                # This is only one step look ahead. Probably many step look ahead is better
                advantages = rewards - values + gamma*tf.reshape(values_next, -1)*np.invert(done).astype(np.float32)

                with tf.GradientTape() as tape:

                    probs = self.actor(states)
                    
                    entropy_coeff = 0.1
                    z0 = tf.reduce_sum(probs, axis = 1)
                    p0 = probs / tf.reshape(z0, (-1,1)) 
                    entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
                    mean_entropy = tf.reduce_mean(entropy) 
                    self.entropy_loss =  mean_entropy * entropy_coeff 

                    action_one_hot = tf.one_hot(actions, 309, on_value=1,off_value=0).numpy().astype(np.float32)
                    actions_prob = tf.reduce_sum(tf.multiply(probs,action_one_hot), axis=1)
                    action_log_probs =  tf.math.log(actions_prob+self.zero_fixer)
                    
                    actor_loss1 = action_log_probs * advantages 
                    actor_loss1 = - tf.reduce_mean(actor_loss1) 
                    
                    actor_loss = actor_loss1 + self.entropy_loss 
                    grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads,self.actor.trainable_variables))
                    
                    #predictions = tf.experimental.numpy.amax(probs, axis = 1)
                    best_action = tf.argmax(self.actor(states.astype(np.float32)), axis=1)

        
        self.history_actor = actor_loss 
        self.history_critic =  value_loss 
        self.actions = actions
        self.predictions  = best_action.numpy()
        self.loss = actor_loss + value_loss


    def discounted_rewards(self, reward):

        gamma = 0.95  # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward, dtype='float64')
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: 
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
        
        #if tf.reduce_sum(reward) != 0:
        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) + self.zero_fixer
        return discounted_r

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma*r*(1.-done) # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]

    def feed_memory(self, state, action, reward, next_state, done, legal_actions):

        self.memory.save(state, action, reward, next_state, done, legal_actions)

    
    def create_actor(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512,activation='relu')(x)
        
        output = Dense(action_num, activation='softmax')(x)
        network = keras.Model(inputs = input_x, outputs=output)
        return network
        
    def create_critic(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(512,activation='relu')(x)
        
        output = Dense(action_num)(x)
        network = keras.Model(inputs = input_x, outputs=output)
        return network


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

        samples = self.memory[0:self.batch_size]

        return map(np.array, zip(*samples))

