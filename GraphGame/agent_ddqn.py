import tensorflow as tf 
import keras
import numpy as np 
from keras.layers import Dense
import random

import graph 
import logger
import hypernetwork
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from scipy.spatial.distance import pdist, squareform
import scipy

            
class Agent():

    def __init__(self, 
                 epochs,
                 mini_batch_size,
                 learning_rate,
                 lamBda, 
                 decay_rate,
                 weights_init, 
                 pretraining_steps,
                 batch_size, 
                 kernel_diversity, 
                 cosine_diversity, 
                 path, 
                 row):
        
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.lamBda = lamBda
        self.decay_rate = decay_rate
        self.weights_init = weights_init
        self.pretraining_steps = pretraining_steps
        self.batch_size = batch_size 
        self.kernel_diversity = kernel_diversity
        self.cosine_diversity = cosine_diversity
        self.path = path 
        self.row = row 

        # These are the hyperparameters which wont be trained (for now)
        self.seed = 42 
        self.input_noise_size = 4
        self.gamma = 0.99
        self.zero_fixer = 1e-9
        self.training_steps = 3000
        self.graph = graph.Graph_Game(7,[[5,5]])
        
        # Additional information for the ddqn
        self.mini_batch_size = 15
        self.memory_size = 1000
        self.update_every = 300
        self.epsilon_start = 1
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 30000  * self.batch_size
        self.total_t = 0
        self.epsilons = np.linspace(self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps)


        self.scores, self.average = [], []
        self.left, self.right, self.left_average, self.right_average = [],[],[],[]

        self.online_network = self.create_network()
        self.target_network = self.create_network()
        self.hypernetwork = hypernetwork.Hypernetwork_DDQN(self.weights_init)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate,self.mini_batch_size*self.epochs,self.decay_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        self.logger = logger.Logger(self.path, self.row)

    def play_game(self, thread):
        
        # Get the starting point and clear all previous data      
        self.states, self.actions, self.next_states, self.rewards, self.dones, self.predictions, self.values = [],[],[],[],[],[],[] 

        score = 0
        right_down = 0
        right_up = 0
        done = False
        state = self.graph.start()

        # Play one game
        while not done:
            
            # Get the actions and new states
            epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
            prediction = self.online_network(state)[0].numpy()
            best_action = np.argmax(prediction)
            actions = [0,1,2,3]
            x = random.random()
            action = np.random.choice(actions)
            if x > epsilon:
                action = best_action

            next_state, reward, done = self.graph.next(action)
            action_onehot = np.zeros([4])
            action_onehot[action] = 1
            score += reward
            self.total_t += 1

            # Safe them for later use
            self.states.append(state)
            self.actions.append(action_onehot)
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
                self.right_down_average.append(sum(self.right_down[-50:]) / len(self.right_down[-50:]))
                self.right_up_average.append(sum(self.right_up[-50:]) / len(self.right_up[-50:]))

                # Calculate score and average
                self.scores.append(score)
                self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))

                # Save the data to memory depending on the thread aka worker
                for i in range(len(self.dones)):
                    self.save_to_memory(self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i], 0, 0, thread)


    def save_to_memory(self, states, actions, rewards, next_states, dones, predictions, values, thread):
        
        if len(self.memory[thread]) == self.memory_size:
            self.memory[thread].pop(0)
        self.memory[thread].append([states, actions, rewards, next_states, dones, predictions, values])

    def sample_from_memory(self, thread):

        samples = random.sample(self.memory[thread], self.mini_batch_size)

        return map(np.array, zip(*samples))

    def create_memory(self):

        self.memory = [[] for _ in range(self.batch_size)]

    def train_normal(self):

        self.graph.build()
        thread = 0
        self.create_memory()
        self.right_down, self.right_up, self.right_down_average, self.right_up_average = [],[],[],[]

        for step in range(self.training_steps):
            
            self.play_game(thread)

            if step > 10:
                self.states, self.actions, self.rewards, self.next_states, self.dones, self.predictions, self.values = self.sample_from_memory(thread)
        
            with tf.GradientTape() as tape:

                q_pred = self.online_network(self.states)
                q_pred_next = self.online_network(self.next_states)
                q_next = self.target_network(self.next_states)
                q_next = tf.clip_by_value(q_next,-1,1).numpy()

                best_actions = tf.math.argmax(q_pred_next, axis =1).numpy()
                #best_actions = tf.reshape(best_actions,(-1,1))

                self.dones = np.reshape(self.dones, -1)
                index = np.arange(0,len(self.dones))   
                self.rewards = tf.reshape(self.rewards,(-1))

                target =  q_next[index, best_actions] #* q_next[np.arange(self.batch_size), best_actions] 
                q_target = self.rewards + self.gamma * np.invert(self.dones).astype(np.float32) * target
                values = self.online_network(self.states)
                pred = tf.argmax(values, axis=1)
                
                values_v = tf.reduce_sum(tf.multiply(values,self.actions), axis=1)
                xyz = tf.math.square(values_v - q_target.numpy())
                value_loss = tf.reduce_mean(xyz)
                
                grads = tape.gradient(value_loss, self.online_network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.online_network.trainable_variables))
            
            if step % 5 == 0:
                
                winrate = self.average[-1]
                right_up, right_down = self.right_up_average[-1], self.right_down_average[-1] 
                self.logger.log_performance(step, winrate, value_loss.numpy(), value_loss.numpy(), 0, 0, 0, \
                    self.optimizer._decayed_lr(tf.float32).numpy(), right_up, right_down, 0, 0, self.row)
                #print('values', values[0:5],'predictions', values_v)

            if step % self.update_every == 0:
                self.target_network.set_weights(self.online_network.get_weights())

        self.logger.close_files()



    def train_hypernetwork(self):
        
        # Create the enviroment
        self.graph.build()
        self.create_memory()
        self.score, self.average = [], []
        self.right_down, self.right_up, self.right_down_average, self.right_up_average = [],[],[],[]

        # Before the training loop is entered, a first set of workers needs to be generated. Otherwise the loop for training 
        # with tf.GradientTape() can't track the gradients
        z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,self.input_noise_size])
        w1a, w2, w4 = self.hypernetwork(z,self.batch_size)

        # Reshape it for the online network and set the weights for the target network
        weights_online_network = tf.concat(axis=1, values=[tf.reshape(w1a,(self.batch_size,-1)), tf.reshape(w2,(self.batch_size,-1)),tf.reshape(w4,(self.batch_size,-1))])
        weights_target_network = weights_online_network        

        # Start the actual hypernetwork training loop
        for step in range(self.training_steps):
            
            # In the beginning always clear the memory and the score
            self.score = 0

            # Play a game for ever generated worker
            for thread in range(self.batch_size):
                
                self.set_weights(weights_online_network, thread)
                self.play_game(thread)
            
            z = np.random.uniform(low = -0.1, high = 0.1, size = [self.batch_size,self.input_noise_size])
            
            with tf.GradientTape() as tape:
                
                # In the beginning of the gradient, the accuracy loss is set to zero and new weights generated
                # Also make new lists for the cosine calculation of the probabilites
                self.embedding = [[] for _ in range(self.batch_size)] 
                self.loss_acc = 0
                w1, w2, w4 = self.hypernetwork(z,self.batch_size)
                
                # Reshape it for the online network and eventually update the target networks
                weights_online_network = tf.concat(axis=1, values=[tf.reshape(w1,(self.batch_size,-1)), tf.reshape(w2,(self.batch_size,-1)),tf.reshape(w4,(self.batch_size,-1))])
                if step % self.update_every == 0:
                    weights_target_network = weights_online_network

                # For each worker first the weights are set and then one batch calculated and added to the overall loss
            
                for num in range(self.batch_size):

                    self.set_weights(weights_online_network, num)
                    self.set_target_weights(weights_target_network, num)
                    self.update_weights(num)
                    
                # If the number of workers generated is greater than one, diversity is calculated
                if self.batch_size > 1:

                    # Calculate the cosine similarity of the actions and the diversity of the weights
                    # To do that, first the predictions have to be chosen for the same states
                    self.evaluate_actions(weights_online_network)
                    cos = self.cosine_similarity_actions()
                    dvd = self.kernel_similarity() 

                    # It is checked, which diversity shall be calculated
                    if self.kernel_diversity:
                        self.loss_div = dvd * self.lamBda
                    elif self.cosine_diversity:
                        cos  = tf.dtypes.cast(cos, tf.float32)
                        self.loss_div = cos * self.lamBda
                    else:
                        self.loss_div = tf.constant(0.)

                    # And the overall loss is:
                    loss = self.loss_div + self.loss_acc 

                # Otherwise the diversity loss term is set to zero
                else:
                    self.loss_div = tf.constant(0.)
                    loss = self.loss_acc 
                    dvd, cos = tf.constant(0.), tf.constant(0.)

                # Finally, the hypernetwork is updated
                grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))
        
            if step % self.update_every == 0:
            
                self.set_target_weights(weights_online_network, 0)  

            # Logging
            if step % 5 == 0:

                winrate = self.average[-1]
                right_up, right_down = self.right_up_average[-1], self.right_down_average[-1] 
                self.logger.log_performance(step, winrate, self.loss_acc.numpy(), 0, 0, self.loss_div.numpy(), loss.numpy(), \
                    self.optimizer._decayed_lr(tf.float32).numpy(), right_up, right_down, dvd.numpy(), cos.numpy(), self.row)
                #print('values', self.values[0:5],'predictions', self.predictions)

        # Saving + End    
        self.hypernetwork.save_weights('{}/hypernetwork_{}_{}.h5'.format(self.path,self.row,self.score))
        self.logger.close_files()


    def update_weights(self, thread):

        # Get the states from the memory depending on the created worker
        self.states, self.actions, self.rewards, self.next_states, self.dones, self.predictions, self.values = self.sample_from_memory(thread)
        
        # Check if the actor won the game
        if np.sum(self.rewards)>0:
            self.score += 1

        q_pred = self.online_network(self.states)
        q_pred_next = self.online_network(self.next_states)
        q_next = self.target_network(self.next_states)
        q_next = tf.clip_by_value(q_next,-1,1).numpy()

        best_actions = tf.math.argmax(q_pred_next, axis =1).numpy()
        
        self.dones = np.reshape(self.dones, -1)
        index = np.arange(0,len(self.dones))   
        self.rewards = tf.reshape(self.rewards,-1)
        q_target = self.rewards + self.gamma * np.invert(self.dones).astype(np.float32) * q_next[index, best_actions] #* q_next[np.arange(self.batch_size), best_actions] 

        values = self.online_network(self.states)
        pred = tf.argmax(values, axis=1)
        
        values_v = tf.reduce_sum(tf.multiply(values,self.actions), axis=1)
    
        self.value_loss = tf.reduce_mean(tf.math.square(values_v - q_target.numpy()))

        # Total loss
        self.loss_acc += self.value_loss

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


    def kernel_similarity(self):

        X = []
        for i in range(self.batch_size):
            X.append(tf.reshape(self.embedding[i],-1))

        matrix = []
        for i in range(self.batch_size):
            for j in range(self.batch_size): 
                z = -tf.math.squared_difference(X[i],X[j])
                xz = tf.exp(z)
                matrix.append(tf.reduce_mean(xz))
                    
        matrix = tf.reshape(matrix, (self.batch_size,self.batch_size))  

        kernel = rbf_kernel(X)


        determinant = tf.linalg.det(matrix)
        d = tf.linalg.det(kernel)
        #print(determinant,d)
        return - tf.math.log(determinant + self.zero_fixer)
    

    def set_weights(self, weights_online_network, num):
        
        # This part is used to update the weights for the online network
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

    def discounted_r(self, reward):

        # This function is used to calculate the discounted rewards of the agents.
        gamma = 0.98 
        running_add = 0
        discounted_r = np.zeros(len(reward))
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
       
        return discounted_r

    def create_network(self):

        kernel_init = tf.keras.initializers.glorot_uniform(self.seed)
        bias_init = tf.keras.initializers.constant(0)
        model = tf.keras.Sequential([
            Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, input_shape=(14,)),
            Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init),
            Dense(4, activation='linear', kernel_initializer=kernel_init, bias_initializer=bias_init)
        ])  
        return model


    def evaluate_actions(self, weights_online_network):

        e_states = []
        # This function is calculating the action embeddings for the cosine similarity and the kernel similarity
        # Select randomly a replay buffer batch_size x times and append mini_batch_size x times many states to the evaluation list
        for _ in range(self.batch_size):
            thread = np.random.randint(0,self.batch_size)
            states, _ , _ , _ , _ , _ , _  = self.sample_from_memory(thread)
            e_states.append(states)

        # Then collect all the actions of all the agents for the states and save them to the embeddings-list
        for thread in range(self.batch_size):

            self.set_weights(weights_online_network, thread)
            predictions = []
            for batch in e_states:
                predictions.append(self.online_network(batch))
            
            predictions = tf.concat(predictions,axis=0)
            self.embedding[thread] = predictions



