import tensorflow as tf 
import keras
import numpy as np 
from keras.layers import Dense
import random

import graph 
import logger
import hypernetwork
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import tensorflow_probability as tfp
from scipy.spatial.distance import pdist, squareform
import scipy

            
class Agent():

    def __init__(self, path):
                 

        # These are the hyperparameters which wont be trained (for now)
        self.seed = 42
        self.input_noise_size = 4
        self.gamma = 0.99
        self.zero_fixer = 1e-9
        self.training_steps = 1000
        self.graph = graph.Graph_Game(7,[[5,5]])
        self.graph.build()
        
        # Additional information for the ddqn
        self.memory_size = 1000
        self.update_every = 300

        self.scores, self.average = [], []
        self.left, self.right, self.left_average, self.right_average = [],[],[],[]

        self.step = 0
        self.path = path
        self.row = 0
        self.logger = logger.Logger(self.path, self.row)

        self.total_t = 0
        self.learning_rate = 0.01
        self.sensings = 4
        self.smoothing = 1
        self.lamBda = 0.5
        self.mutation_size = 4
        self.population_size = 20
        self.elite_num = 4
        self.batch_size = 50
        self.evaluate_num = 2
        self.memory = []
        self.sigma = 0.05
        self.n_iter = 1
        self.eps = 10

        self.reward_min = 1.15

        self.weight_space = 5380

        # Change this to a larger mutation size 
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 100 * self.population_size
        self.total_t = 0
        self.epsilons_ddqn = np.linspace(self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)

        self.m = {i:0 for i in range(self.population_size)}
        self.v = {i:0 for i in range(self.population_size)}


    def Adam(self, dx, idx, learning_rate, t, eps = 1e-8, beta1 = 0.9, beta2 = 0.999):
        
        self.m[idx] = beta1 * self.m[idx] + (1 - beta1) * dx
        mt = self.m[idx] / (1 - beta1 ** t)
        self.v[idx] = beta2 * self.v[idx] + (1-beta2) * (dx **2)
        vt = self.v[idx] / (1 - beta2 ** t)
        update = learning_rate * mt / (np.sqrt(vt) + eps)
        return(update)


    def train_generation(self):

        self.step +=1
        self.epsilons = (self.sigma**2) * np.random.normal(0,1,(self.population_size,self.weight_space))
        self.R = []
        embedding_matrix = []
        self.dvd, self.right_down, self.right_up = [],[],[]

        for idx in range(self.population_size):
                     
            agents_weights = self.agents[idx] + self.epsilons[idx]
            self.set_weights(agents_weights)
            reward = self.evaluate()
            self.R.append(reward/self.evaluate_num)

        elite_idx = np.argsort(self.R)[self.population_size-self.elite_num:]
        
        elite = []
        for idx in elite_idx:
            elite.append(self.agents[idx])

        self.epsilons = (self.sigma**2) * np.random.normal(0,1,(self.population_size,self.weight_space))

        for idx in range(self.population_size):
            
            y = np.random.choice(elite_idx)
            agents_weights = self.agents[y] + self.epsilons[idx]
            self.set_weights(agents_weights)
            states, _, _, _, _ = self.sample_from_memory()
            predictions = self.online_network(states).numpy()
            embeddings = np.argmax(predictions, axis = 1).astype('float32')
            embeddings = tf.one_hot(embeddings,4,axis = 1)
            embedding_matrix.append(np.reshape(embeddings,-1))

        embedding_matrix = np.reshape(embedding_matrix,(self.population_size,-1))
        mean = np.mean(self.R)
        #self.R = np.reshape(self.R,(self.population_size,self.population_size))
        
        if self.step >= 3:

            self.agents = []
            determinants = self.calculate_dvd(embedding_matrix)
            diverse = np.argsort(determinants)[self.population_size-self.mutation_size:]

            for i in range(self.population_size):
                
                x = np.random.choice(diverse)
                y = np.random.choice([0,1,2,3])

                self.agents.append(elite[y] + self.epsilons[x])
            
            self.logger.log_performance(self.step, mean, np.max(self.R) - self.reward_min, 0, 0, 0, 0, \
                        0, np.mean(self.right_up), np.mean(self.right_down), self.dvd, 0, self.row)
            print('values', np.mean(self.R),'predictions', self.predictions[0])

            print(self.R, self.R[i])


    def init_population(self):

        self.agents = []

        for i in range(self.population_size):

            self.online_network = self.create_network()
            state = np.zeros((1,14),'float32')
            z = self.online_network(state)
            self.seed += 1

            z = self.online_network.get_weights()
            z = np.concatenate((tf.reshape(z[0],-1), tf.reshape(z[1],-1), tf.reshape(z[2],-1), tf.reshape(z[3],-1),tf.reshape(z[4],-1),tf.reshape(z[5],-1) ))
            
            self.agents.append(z)

    
    def calculate_dvd(self, embedding_matrix):

        determinants = []
        for idx in range(len(self.epsilons)):           
            matrix = []
            for i in range(self.population_size):
                for j in range(self.population_size): 
                    z = tf.exp(-tf.math.squared_difference(embedding_matrix[idx,i],embedding_matrix[idx,j]))
                    matrix.append(tf.reduce_mean(z))
                    
            matrix = tf.reshape(matrix, (self.population_size,self.population_size))  
            determinant = tf.linalg.det(matrix)
            determinants.append(determinant)
        
        return determinants


    def evaluate(self):
        
        total_score = []

        for _ in range(self.evaluate_num):

            # Get the starting point and clear all previous data      
            self.states, self.actions, self.next_states, self.rewards, self.dones, self.predictions = [],[],[],[],[],[]

            self.score = 0
            right_down = 0
            right_up = 0
            done = False
            state = self.graph.start()

            # Play one game
            while not done:
                
                state = state.astype('float32')
                # Get the actions and new states
                
                epsilon = self.epsilons_ddqn[min(self.total_t, self.epsilon_decay_steps-1)]

                prediction = self.online_network(state)
                best_action = np.argmax(prediction)
                actions = [0,1,2,3]
                x = random.random()
                action = np.random.choice(actions)
                if x > epsilon:
                    action = best_action
                next_state, reward, done = self.graph.next(action)
                action_one_hot = np.zeros([4])
                action_one_hot[action] = 1
                self.score += reward
                self.total_t += 1

                # Safe them for later use
                self.states.append(state)
                self.actions.append(action_one_hot)
                self.next_states.append(next_state)
                self.rewards.append(reward)
                self.dones.append(done)
                self.predictions.append(prediction)

                state = next_state

                # Check if the agent is going left or right
                if state[0][3] == 1:
                    if state[0][11]==1 or state[0][12]==1 or state[0][13]==1:
                        right_down = 1
            
                if state[0][10] == 1:
                    if  state[0][0] ==1 or state[0][1]==1 or state[0][2]==1:
                        right_up = 1
                
                if done:
                    # Transform the lists to numpy stacks
                    self.states = np.vstack(self.states)
                    self.actions = np.vstack(self.actions)
                    self.rewards = np.vstack(self.discounted_r(self.rewards))
                    self.next_states = np.vstack(self.next_states)
                    self.dones = np.vstack(self.dones)
                    self.predictions = np.vstack(self.predictions)

                    # Calculate the proportion of left and rights
                    self.right_down.append(right_down)
                    self.right_up.append(right_up)

                    total_score.append(self.score + self.reward_min)

                    for i in range(len(self.dones)):
                        self.save_to_memory(self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i], self.predictions[i])

        return np.mean(total_score)


    def cosine_similarity_actions(self):

        cosine_actions = []
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                score = 0
                v1, v2 = self.embedding[i], self.embedding[j]
        
                for x in range(self.mini_batch_size):
                    score += self.cos_between(v1[x],v2[x])/self.mini_batch_size
                
                cosine_actions.append(score)

        return tf.reduce_mean(cosine_actions)


    def cos_between(self, v1, v2):

        v1_u = v1 / tf.norm(v1)
        v2_u = v2 / tf.norm(v2)
        
        return tf.tensordot(v1_u, v2_u, axes=1)


    def discounted_r(self, reward):

        # This function is used to calculate the discounted rewards of the agents.
        gamma = 0.98 
        running_add = 0
        discounted_r = np.zeros(len(reward))
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
       
        return discounted_r

    def set_weights(self, weights):

        # This part is used to update the weights for the online network
        last_used = 0
        weights = weights.astype('float32')
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

    def save_to_memory(self, states, actions, rewards, next_states, dones, predictions):
        
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        self.memory.append([states, actions, rewards, next_states, dones])

    def sample_from_memory(self):
        
        samples = random.sample(self.memory, self.batch_size)

        return map(np.array, zip(*samples))

    def create_network(self):

        kernel_init = tf.keras.initializers.glorot_uniform(self.seed)
        bias_init = tf.keras.initializers.constant(0)
        model = tf.keras.Sequential([
            Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init),
            Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init),
            Dense(4, activation='linear', kernel_initializer=kernel_init, bias_initializer=bias_init)
        ])  
        return model


