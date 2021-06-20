import tensorflow as tf 
import keras
import numpy as np 
from keras.layers import Dense

import graph 
import logger
import hypernetwork
#from utils import mutual_info, cosine_actions, cosine_weights, KL_estimator
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import tensorflow_probability as tfp
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

        self.scores, self.average = [], []
        self.right_up, self.right_down, self.right_up_average, self.right_down_average = [],[],[],[]

        self.actor = self.create_actor()
        self.critic = self.create_critic()
        self.hypernetwork = hypernetwork.Hypernetwork_A2C(self.weights_init)

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
            prediction = self.actor(state)[0]
            value = self.critic(state)[0]
            actions = [0,1,2,3]
            action = np.random.choice(actions, p=prediction.numpy())
            next_state, reward, done = self.graph.next(action)
            action_onehot = np.zeros([4])
            action_onehot[action] = 1
            score += reward

            # Safe them for later use
            self.states.append(state)
            self.actions.append(action_onehot)
            self.next_states.append(next_state)
            self.rewards.append(reward)
            self.dones.append(done)
            self.predictions.append(prediction)
            self.values.append(value)

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
                self.values = np.vstack(self.values)

                # Calculate the proportion of left and rights
                self.right_down.append(right_down)
                self.right_up.append(right_up)
                self.right_down_average.append(sum(self.right_down[-50:]) / len(self.right_down[-50:]))
                self.right_up_average.append(sum(self.right_up[-50:]) / len(self.right_up[-50:]))

                # Calculate score and average
                self.scores.append(score)
                self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))

                # Save the data to memory depending on the thread aka worker
                self.save_to_memory(self.states, self.actions, self.rewards, self.next_states, self.dones, self.predictions, self.values, thread)


    def save_to_memory(self, states, actions, rewards, next_states, dones, predictions, values, thread):
        
        self.memory[thread] = states, actions, rewards, next_states, dones, predictions, values


    def clear_memory(self):

        self.memory = [[] for _ in range(self.batch_size)]


    def train_normal(self):

        self.graph.build()
        thread = 0

        for step in range(self.training_steps):
            
            self.clear_memory()
            self.play_game(thread)
            self.states, self.actions, self.rewards, self.next_states, self.dones, self.predictions, self.values = self.memory[thread]

            # Create the index for one episode, shuffle it and scale for the data length
        
            for _ in range(self.epochs):
                
                index = np.arange(len(self.rewards))
                np.random.shuffle(index)
                step_size = len(self.rewards)// self.mini_batch_size

                for start in range(0,len(self.rewards), step_size):
                    
                    end = start + step_size
                    idx = index[start:end]

                    states = self.states[idx]
                    actions = self.actions[idx]
                    rewards = self.rewards[idx]
                    next_states = self.next_states[idx]
                    dones = self.dones[idx]
                    old_prob = self.predictions[idx]
                    old_values = self.values[idx]
                    
                    with tf.GradientTape() as tape:

                        # Critic part
                        cliprange = 0.2

                        values = self.critic(states)
                        values_next = self.critic(next_states)
                        values_clipped = old_values + tf.clip_by_value(values - old_values, - cliprange, cliprange)
                        values_loss_1 = tf.math.square(values - rewards)                  
                        values_loss_2 = tf.math.square(values_clipped - rewards)
                                            
                        loss_critic = tf.reduce_mean(tf.math.maximum(values_loss_1, values_loss_2))*0.5
                        grads = tape.gradient(loss_critic, self.critic.trainable_variables)
                        self.optimizer.apply_gradients(zip(grads,self.critic.trainable_variables))

                    with tf.GradientTape() as tape:

                        # Actor part 
                        prob = self.actor(states)

                        advantages = rewards - values + self.gamma *values_next*np.invert(dones).astype(np.float32)
                        advantages = np.reshape(advantages, (-1))
                        
                        log_prob = tf.reduce_sum(tf.math.multiply((prob+self.zero_fixer),actions),axis=1)
                        log_old_prob = tf.reduce_sum(tf.math.multiply((old_prob+self.zero_fixer),actions),axis=1)

                        clipping_value = 0.2
                        r = log_prob/(log_old_prob+self.zero_fixer)
                        r1 = - advantages * r
                        r2 = - advantages * tf.clip_by_value(r, 1 - clipping_value, 1 + clipping_value)
                        
                        entropy_coeff = 0.01
                        z0 = tf.reduce_sum(prob, axis = 1)
                        z0 = tf.stack([z0,z0,z0,z0], axis=-1)
                        p0 = prob / (z0 + self.zero_fixer) 
                        entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
                        mean_entropy = tf.reduce_mean(entropy) 
                        entropy_loss =  mean_entropy * entropy_coeff 
                        
                        loss_actor = tf.math.reduce_mean(tf.math.maximum(r1,r2), axis=None) #- entropy_loss
                        grads = tape.gradient(loss_actor, self.actor.trainable_variables)
                        self.optimizer.apply_gradients(zip(grads,self.actor.trainable_variables))
                    '''

                    with tf.GradientTape() as tape:
                        
                        values = self.critic(states)
                        next_values = self.critic(next_states)
                        
                        loss_critic = tf.reduce_mean(tf.math.square(values-rewards))*0.5

                        grads = tape.gradient(loss_critic, self.critic.trainable_variables)
                        self.optimizer.apply_gradients(zip(grads,self.critic.trainable_variables))

                    with tf.GradientTape() as tape:

                        predictions = self.actor(states)

                        advantages = rewards - values + self.gamma*next_values*np.invert(dones)
                        advantages = tf.reshape(advantages, (-1))
                        pred = tf.reduce_sum(predictions*actions, axis=1)
                        log_pred =tf.math.log(pred + 1e-9)

                        entropy_coeff = 0.1
                        z0 = tf.reduce_sum(predictions + self.zero_fixer, axis = 1)
                        z0 = tf.stack([z0,z0,z0,z0], axis=-1)
                        p0 = predictions / z0 
                        entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
                        mean_entropy = tf.reduce_mean(entropy) 
                        entropy_loss =  mean_entropy * entropy_coeff 

                        loss_actor = - tf.reduce_mean(log_pred*advantages) + entropy_loss
                        grads = tape.gradient(loss_actor, self.actor.trainable_variables)
                        self.optimizer.apply_gradients(zip(grads,self.actor.trainable_variables))
                        '''
            if step % 5 == 0:
                
                winrate = self.average[-1]
                right_up, right_down = self.right_up_average[-1], self.right_down_average[-1] 
                self.logger.log_performance(step, winrate, loss_actor.numpy(), loss_critic.numpy(), entropy_loss.numpy(), 0, 0, \
                    self.optimizer._decayed_lr(tf.float32).numpy(), right_up, right_down, 0, 0, self.row)
                #print('values', self.values[0:5],'predictions', self.predictions)

        self.logger.close_files()

    def pre_train(self):
        
        # Higher learning rate for pretraining, since its just for scaling
        optimizer = tf.keras.optimizers.Adam(lr=3e-4)
        print('Is pretraining')
        for _ in range(self.pretraining_steps):

            with tf.GradientTape() as tape: 
                
                # Since we initialized in the beginning an actor and a critic with weights by default, we take these
                # weigths and fit for them
                w1a_true = np.concatenate(self.actor.layers[0].get_weights(),axis = None)
                w1c_true = np.concatenate(self.critic.layers[0].get_weights(),axis = None)
                w2_true = np.concatenate(self.actor.layers[1].get_weights(),axis = None)
                w3_true = np.concatenate(self.critic.layers[1].get_weights(),axis = None)
                w4_true = np.concatenate(self.actor.layers[2].get_weights(),axis = None)
                w5_true = np.concatenate(self.critic.layers[2].get_weights(),axis = None)
                
                # The batch_size has to be one, since we only fit for one set of agents
                z = np.random.uniform(low = -1, high = 1, size = [1,300])
                w1c, w1a, w2, w3, w4, w5 = self.hypernetwork(z,1)
                w1c = tf.reshape(w1c, -1)
                w1a = tf.reshape(w1a, -1)
                w2 = tf.reshape(w2, -1)
                w3 = tf.reshape(w3, -1)
                w4 = tf.reshape(w4, -1)
                w5 = tf.reshape(w5, -1)

                loss_actor = tf.losses.mse(w1a_true, w1a) + tf.losses.mse(w2_true, w2) + tf.losses.mse(w4_true, w4)
                loss_critic = tf.losses.mse(w1c_true, w1c) + tf.losses.mse(w3_true, w3) + tf.losses.mse(w5_true, w5)
                loss = loss_actor + loss_critic
                grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                optimizer.apply_gradients(zip(grads,self.hypernetwork.trainable_weights))


    def train_hypernetwork(self):
        
        # Create the enviroment
        self.graph.build()
        self.score, self.average = [], []
        self.left, self.right, self.left_average, self.right_average = [],[],[],[]

        # Do pretaining if wanted
        if self.pretraining_steps != 0:
            self.pre_train()

        # Before the training loop is entered, a first set of workers needs to be generated. Otherwise the loop for training 
        # with tf.GradientTape() can't track the gradients
        z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,self.input_noise_size])
        w1c, w1a, w2, w3, w4, w5 = self.hypernetwork(z,self.batch_size)
        print(self.hypernetwork.summary())
        # Reshape it for actor and crit        
        weights_actor = tf.concat(axis=1, values=[tf.reshape(w1a,(self.batch_size,-1)), tf.reshape(w2,(self.batch_size,-1)),tf.reshape(w4,(self.batch_size,-1))])
        weights_critic = tf.concat(axis=1, values=[tf.reshape(w1c,(self.batch_size,-1)), tf.reshape(w3,(self.batch_size,-1)),tf.reshape(w5,(self.batch_size,-1))])
        print(self.actor.summary())
        print(self.critic.summary())

        # Start the actual hypernetwork training loop
        for step in range(self.training_steps):
            
            # In the beginning always clear the memory and the score
            self.clear_memory()
            self.score = 0

            # Play a game for ever generated worker
            for thread in range(self.batch_size):
                
                self.set_weights(weights_actor, weights_critic, thread)
                self.play_game(thread)
            
            z = np.random.uniform(low = -0.1, high = 0.1, size = [self.batch_size,self.input_noise_size])

            # Now train hypernetwork for each worker for predefined number of epochs
            for e in range(self.epochs):
            
                with tf.GradientTape() as tape:
                    
                    # In the beginning of the gradient, the accuracy loss is set to zero and new weights generated
                    # Also make new lists for the cosine calculation of the probabilites
                    self.cosine_probs = [[] for _ in range(self.batch_size)] 
                    self.loss_acc = 0
                    w1c, w1a, w2, w3, w4, w5 = self.hypernetwork(z,self.batch_size)
                    
                    # Reshape it for actor and critic
                    weights_actor = tf.concat(axis=1, values=[tf.reshape(w1a,(self.batch_size,-1)), tf.reshape(w2,(self.batch_size,-1)),tf.reshape(w4,(self.batch_size,-1))])
                    weights_critic = tf.concat(axis=1, values=[tf.reshape(w1c,(self.batch_size,-1)), tf.reshape(w3,(self.batch_size,-1)),tf.reshape(w5,(self.batch_size,-1))])
                    
                    # For each worker first the weights are set and then one batch calculated and added to the overall loss
                    for num in range(self.batch_size):

                        self.set_weights(weights_actor, weights_critic, num)
                        self.update_weights(num)
                            
                    # If the number of workers generated is greater than one, diversity is calculated
                    if self.batch_size > 1:

                        # Calculate the cosine similarity of the actions and the diversity of the weights
                        # To do that, first the predictions have to be chosen for the same states
                        self.evaluate_actions(weights_actor, weights_critic)
                        cos = self.cosine_similarity_actions()
                        kl = self.KL_estimator(w1c,w1a,w2,w3,w4,w5)
                        dvd = self.div_kernel()

                        # It is checked, which diversity shall be calculated
                        if self.kl_diversity:
                            self.loss_div = kl * self.lamBda
                        elif self.cosine_diversity:
                            cos  = tf.dtypes.cast(cos, tf.float32)
                            self.loss_div = cos * self.lamBda
                        else:
                            self.loss_div = tf.constant(0.)

                        self.loss_div = dvd * self.lamBda
                        # And the overall loss is:
                        loss = self.loss_div #+ self.loss_acc    

                    # Otherwise the diversity loss term is set to zero
                    else:
                        self.loss_div = tf.constant(0.)
                        loss = self.loss_acc 
                        kl, cos = tf.constant(0.), tf.constant(0.)

                    # Finally, the hypernetwork is updated
                    grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))

            # Logging
            if step % 5 == 0:

                winrate = self.average[-1]
                left, right = self.left_average[-1], self.right_average[-1] 
                self.logger.log_performance(step, winrate, self.loss_actor.numpy(), self.loss_critic.numpy(), self.entropy_loss.numpy(), self.loss_div.numpy(), loss.numpy(), \
                    self.optimizer._decayed_lr(tf.float32).numpy(), left, right, dvd.numpy(), cos.numpy(), self.row)
                #print('values', self.values[0:5],'predictions', self.predictions)

        # Saving + End    
        self.hypernetwork.save_weights('{}/hypernetwork_{}_{}.h5'.format(self.path,self.row,self.score))
        self.logger.close_files()


    def update_weights(self, thread):

        # Get the states from the memory depending on the created worker
        self.states, self.actions, self.rewards, self.next_states, self.dones, self.predictions, self.values = self.memory[thread]
        
        # Check if the actor won the game
        if np.sum(self.rewards)>0:
            self.score += 1

        # Create the index for one episode, shuffle it and scale for the data length
        index = np.arange(len(self.rewards))
        np.random.shuffle(index)
        step_size = len(self.rewards)// self.mini_batch_size

        # Loop over the batches and calculate the loss
        for start in range(0,len(self.rewards), step_size):
            
            end = start + step_size
            idx = index[start:end]

            states = self.states[idx]
            actions = self.actions[idx]
            rewards = self.rewards[idx]
            next_states = self.next_states[idx]
            dones = self.dones[idx]
            old_prob = self.predictions[idx]
            old_values = self.values[idx]

            # Constants for PPO2 (OpenAI)
            cliprange = 0.2            
            entropy_coeff = 0.1
            value_coeff = 0.5

            # Critic part
            values = self.critic(states)
            values_next = self.critic(next_states)
            values_clipped = old_values + tf.clip_by_value(values - old_values, - cliprange, cliprange)
            values_loss_1 = tf.math.square(values - rewards)                  
            values_loss_2 = tf.math.square(values_clipped - rewards) 
            self.loss_critic = tf.reduce_mean(tf.math.maximum(values_loss_1, values_loss_2))*value_coeff

            # Actor part 
            prob = self.actor(states)
            advantages = rewards - values + self.gamma *values_next*np.invert(dones).astype(np.float32)
            advantages = np.reshape(advantages, (-1))
            
            log_prob = tf.reduce_sum(tf.math.multiply((prob+self.zero_fixer),actions),axis=1)
            log_old_prob = tf.reduce_sum(tf.math.multiply((old_prob+self.zero_fixer),actions),axis=1)
            r = log_prob/(log_old_prob+self.zero_fixer)
            r1 = - advantages * r
            r2 = - advantages * tf.clip_by_value(r, 1 - cliprange, 1 + cliprange)
            
            # Entropy of Actor
            z0 = tf.reduce_sum(prob, axis = 1)
            z0 = tf.stack([z0,z0,z0,z0], axis=-1) # TODO Change this
            p0 = prob / (z0 + self.zero_fixer) 
            entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
            mean_entropy = tf.reduce_mean(entropy) 
            self.entropy_loss =  mean_entropy * entropy_coeff 
            
            # Loss Actor
            self.loss_actor = tf.math.reduce_mean(tf.math.maximum(r1,r2), axis=None) - self.entropy_loss

            # Total loss
            self.loss_acc += self.loss_actor + self.loss_critic

    def cosine_similarity_actions(self):

        #cosine_actions = tf.zeros([self.batch_size, self.batch_size])
        cosine_actions = 0
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                score = 0
                v1, v2 = self.cosine_probs[i], self.cosine_probs[j]
        
                for x in range(5):
                    score += self.cos_between(v1[x],v2[x])
                
                cosine_actions += score
                #cosine_actions[i][j] = score
                #cosine_actions[j][i] = cosine_actions[i][j]

        return tf.reduce_sum(cosine_actions)

    def KL_estimator(self, w1c,w1a,w2,w3,w4,w5):

        flattened_network = tf.concat(axis=1,values=[\
                        tf.reshape(w1c, [self.batch_size, -1]),\
                        tf.reshape(w1a, [self.batch_size, -1]),\
                        tf.reshape(w2, [self.batch_size, -1]),\
                        tf.reshape(w3, [self.batch_size, -1]),\
                        tf.reshape(w4, [self.batch_size, -1]),\
                        tf.reshape(w5, [self.batch_size, -1])])

        # entropy estimated using  Kozachenko-Leonenko estimator, with l1 distances
        mutual_distances = tf.math.reduce_sum(tf.math.abs(tf.expand_dims(flattened_network, 0) - tf.expand_dims(flattened_network, 1)), 2)
        nearest_distances = tf.identity(-1*tf.math.top_k(-1 * mutual_distances, k=2)[0][:, 1]) 
        entropy_estimate = tf.identity(self.input_noise_size * tf.math.reduce_mean(tf.math.log(nearest_distances + self.zero_fixer)) + tf.math.digamma(tf.cast(self.batch_size, tf.float32)))
        loss_div = tf.identity( - 1 * entropy_estimate)

        return loss_div

    def cos_between(self, v1, v2):

        #v1_u = v1 / np.linalg.norm(v1)
        #v2_u = v2 / np.linalg.norm(v2)
        #np.dot(v1_u, v2_u) 

        v1_u = v1 / tf.norm(v1)
        v2_u = v2 / tf.norm(v2)
        
        return tf.tensordot(v1_u, v2_u, axes=1)

    def div_kernel(self):

        X = []
        for i in range(self.batch_size):
            X.append(tf.reshape(self.cosine_probs[i],-1))

        #X = (X - tf.math.reduce_mean(X, axis=0))/(tf.math.reduce_std(X, axis=0)+self.zero_fixer)
        #X = tf.clip_by_value(X, -2, 2)
        matrix = []
        for i in range(4):
            for j in range(4): 
                matrix.append(tf.reduce_mean(tf.exp(-tf.math.squared_difference(X[i],X[j]))) - self.zero_fixer)
        matrix = tf.reshape(matrix, (4,4))

        #print(matrix)

        kernel = rbf_kernel(X)

        expanded_a = tf.expand_dims(X, 1)
        expanded_b = tf.expand_dims(X, 0)
        #matrix = tf.reduce_sum(tf.math.exp(- tf.math.squared_difference(expanded_a, expanded_b)), 2)
        kernel2 = tfp.math.psd_kernels.ExponentiatedQuadratic()
        #value = kernel2.apply(X,X)
        dets = tf.linalg.det(matrix)
        #dets = (dets - tf.math.reduce_mean(dets))/(tf.math.reduce_std(dets) + self.zero_fixer)

        d = tf.linalg.det(kernel)

        print(dets)
        p = tf.reduce_mean(matrix, axis=None)
        return - dets

    def set_weights(self, weights_actor, weights_critic, num):
        
        # This part is used to set the weights for the Actor
        last_used = 0
        weights = weights_actor[num]
        for i in range(len(self.actor.layers)):
            if 'conv' in self.actor.layers[i].name or  'dense' in self.actor.layers[i].name: 
                weights_shape = self.actor.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                self.actor.layers[i].kernel = new_weights
                last_used += no_of_weights
                
                if self.actor.layers[i].use_bias:
                    weights_shape = self.actor.layers[i].bias.shape
                    no_of_weights = tf.reduce_prod(weights_shape)
                    new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                    self.actor.layers[i].bias = new_weights
                    last_used += no_of_weights

        # This part is the same, but for the Critic
        last_used = 0
        weights = weights_critic[num]  
        for i in range(len(self.critic.layers)):
          
            if 'conv' in self.critic.layers[i].name or  'dense' in self.critic.layers[i].name: 
                weights_shape = self.critic.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                self.critic.layers[i].kernel = new_weights
                last_used += no_of_weights
                
                if self.critic.layers[i].use_bias:
                    weights_shape = self.critic.layers[i].bias.shape
                    no_of_weights = tf.reduce_prod(weights_shape)
                    new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                    self.critic.layers[i].bias = new_weights
                    last_used += no_of_weights

    def discounted_r(self, reward):

        gamma = 0.98 
        running_add = 0
        discounted_r = np.zeros(len(reward))
        running_add = 0
        for i in reversed(range(0,len(reward))):
            
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        #discounted_r -= np.mean(discounted_r) # normalizing the result
        #discounted_r /= np.std(discounted_r) + self.zero_fixer
       
        return discounted_r

    def create_actor(self):
        kernel_init = tf.keras.initializers.glorot_uniform(self.seed)
        bias_init = tf.keras.initializers.constant(0)
        model = tf.keras.Sequential([
            Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, input_shape=(14,)),
            Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init),
            Dense(4, activation='softmax', kernel_initializer=kernel_init, bias_initializer=bias_init)
        ])  
        return model

    def create_critic(self):
        kernel_init = tf.keras.initializers.glorot_uniform(self.seed)
        bias_init = tf.keras.initializers.constant(0)

        model = tf.keras.Sequential([
            Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, input_shape=(14,)),
            Dense(64, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init),
            Dense(1, activation='linear', kernel_initializer=kernel_init, bias_initializer=bias_init)
        ])  
        return model

    def evaluate_actions(self, weights_actor, weights_critic):
        
        idx = np.random.randint(0,self.batch_size)
        states = self.memory[idx][0][0:8]

        for num in range(self.batch_size):

            self.set_weights(weights_actor, weights_critic, num)
            predictions = self.actor(states)
            actions = tf.math.argmax(predictions,axis = 1)
            actions_one_hot = tf.one_hot(actions, self.batch_size, axis=1)
            self.cosine_probs[num] = actions_one_hot