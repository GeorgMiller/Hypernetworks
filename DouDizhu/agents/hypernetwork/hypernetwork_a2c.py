import keras
import random
import numpy as np

import tensorflow as tf     
from keras.layers import Dense, Input, Flatten, BatchNormalization,Dropout
from keras.optimizers import Adam
import tensorflow.keras.initializers as init
from collections import namedtuple

from utils_global import remove_illegal
import agents.non_rl.rule_based_agent as rule_based_agent
import envs.mydoudizhu as env
import utils.logger_hypernetwork as logger
from utils_global import set_global_seed, tournament


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


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
                 kl_diversity, 
                 cosine_diversity, 
                 path, 
                 row,
                 config,
                 state_shape):

        discount_factor=0.99
        batch_size=1
        state_shape=None
        decay_steps=20 
        decay_rate=0.99998
        gamma = 0.95
        evaluate_num = 50
        evaluate_every = 100
        training_steps = 5000
        memory_length = 10000
        seed = 42

        self.discount_factor = discount_factor
        self.batch_size = 1
        self.action_num = 309
        self.train_every = 1
        self.state_shape = state_shape
        self.config = config

        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.lamBda = lamBda
        self.decay_rate = decay_rate
        self.weights_init = weights_init
        self.pretraining_steps = pretraining_steps
        self.batch_size = batch_size 
        self.kl_diversity = kl_diversity
        self.cosine_diversity = cosine_diversity
        self.path = path 
        self.row = row 

        self.seed = seed
        self.input_noise_size = 10
        self.gamma = gamma
        self.training_steps = training_steps
        self.memory_length = memory_length
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.evaluate_num = evaluate_num
        self.evaluate_every = evaluate_every
        self.state_shape = state_shape

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.decay_steps, self.decay_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic = self.create_critic(1, self.state_shape)
        self.actor = self.create_actor(self.action_num, self.state_shape)
        self.env = env.DoudizhuEnv(self.config, self.state_shape)
        self.eval_env = env.DoudizhuEnv(self.config, self.state_shape)
        self.rule_based_agent = rule_based_agent.DouDizhuRuleAgentV1(action_num=self.eval_env.action_num)
        self.hypernetwork = Hypernetwork(self.input_noise_size, self.env.action_num, self.seed)
        self.logger = logger.Logger(self.log_dir_random,self.row)

        self.total_t = 0
        self.zero_fixer = 1e-9
        self.scores, self.average = [], []
        
    def step(self, state):

        A = self.predict(state['obs'])
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
        return action


    def eval_step(self, state):

        prediction = self.actor(np.expand_dims(state['obs'], 0))[0]
        probs = remove_illegal(np.exp(prediction), state['legal_actions'])
        best_action = np.argmax(probs)
        return best_action, probs


    def predict(self, state):

        prediction = self.actor(np.expand_dims(state,0))[0].numpy()
        return prediction 

    
    def save_to_memory(self, ts, thread):

        (state, action, reward, next_state, done) = tuple(ts)
        transition = Transition(state['obs'], action, reward, next_state['obs'], done, state['legal_actions'])
        self.scores[thread].append(reward)
        self.average[thread].append(sum(self.scores[thread][-50:]) / len(self.scores[thread][-50:]))
        self.memory[thread].append(transition) 


    def clear_memory(self):

        self.memory = [[] for _ in range(self.batch_size)]
        self.scores = [[] for _ in range(self.batch_size)]
        self.average = [[] for _ in range(self.batch_size)]


    def train_hypernetwork(self):

        # Before the training loop is entered, a first set of workers needs to be generated. Otherwise the loop for training 
        # with tf.GradientTape() can't track the gradients
        z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,10])
        w1, w2, w3, w4, w5, w6 = self.hypernetwork(z,self.batch_size)

        # Reshape it for actor and critic
        weights_actor = tf.concat(concat_dim=1, values=[tf.reshape(w1,(self.batch_size,-1)), tf.reshape(w2,(self.batch_size,-1)),tf.reshape(w3,(self.batch_size,-1))])
        weights_critic = tf.concat(concat_dim=1, values=[tf.reshape(w4,(self.batch_size,-1)), tf.reshape(w5,(self.batch_size,-1)),tf.reshape(w6,(self.batch_size,-1))])

        # Start the actual hypernetwork training loop
        for episode in range(self.training_steps):
            
            # In the beginning always clear the memory and the score
            self.clear_memory()
            self.score = 0

            # Play a game for ever generated worker
            for thread in range(self.batch_size):
                
                self.set_weights(weights_actor, weights_critic, thread)
                self.play_game(thread)

            # Now train hypernetwork for each worker for predefined number of epochs
            for e in range(self.epochs):
            
                with tf.GradientTape() as tape:
            
                    # In the beginning of the gradient, the accuracy loss is set to zero and new weights generated
                    # Also make new lists for the cosine calculation of the probabilites
                    self.cosine_probs = [[] for _ in range(self.batch_size)] 
                    self.loss_acc = 0
                    z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,10])
                    w1, w2, w3, w4, w5, w6 = self.hypernetwork(z,self.batch_size)
                    
                    # Reshape it for actor and critic
                    weights_actor = tf.concat(concat_dim=1, values=[tf.reshape(w1,(self.batch_size,-1)), tf.reshape(w2,(self.batch_size,-1)),tf.reshape(w3,(self.batch_size,-1))])
                    weights_critic = tf.concat(concat_dim=1, values=[tf.reshape(w4,(self.batch_size,-1)), tf.reshape(w5,(self.batch_size,-1)),tf.reshape(w6,(self.batch_size,-1))])
                    
                    # For each worker first the weights are set and then one batch calculated and added to the overall loss
                    for thread in range(self.batch_size):

                        self.set_weights(weights_actor, weights_critic, thread)
                        self.update_weights(thread)
                            
                    # If the number of workers generated is greater than one, diversity is calculated
                    if self.batch_size > 1:

                        # Calculate the cosine similarity of the actions and the diversity of the weights
                        # To do that, first the predictions have to be chosen for the same states
                        self.evaluate_actions(weights_actor, weights_critic)
                        cos = self.cosine_similarity_actions()
                        dvd = self.kernel_similarity()

                        # It is checked, which diversity shall be calculated
                        if self.kernel_diversity:
                            self.loss_div = dvd * self.lamBda
                        elif self.cosine_diversity:
                            self.loss_div = cos * self.lamBda
                        else:
                            self.loss_div = tf.constant(0.)

                        # And the overall loss is:
                        loss = self.loss_acc + self.loss_div    

                    # Otherwise the diversity loss term is set to zero
                    else:
                        self.loss_div = tf.constant(0.)
                        loss = self.loss_acc 
                        kl, cos = tf.constant(0.), tf.constant(0.)

                    # Finally, the hypernetwork is updated
                    grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))

            
            
            # Evaluate the performance. Play with random agents.
            if episode % self.evaluate_every == 0:

                self.eval_env.set_agents([self, self.rule_based_agent, self.rule_based_agent])

                self.logger.log_performance(episode, tournament(self.eval_env, self.evaluate_num, self)[0],\
                    self.history_actor, self.history_critic, self.optimizer._decayed_lr(tf.float32).numpy(), self.actions, self.predictions)
                
                
    def evaluate_actions(self, weights_actor, weights_critic):
        
        idx = np.random.randint(0,self.batch_size)
        states = self.memory[idx][0][0:10]

        for num in range(self.batch_size):

            self.set_weights(weights_actor, weights_critic, num)
            predictions = self.actor(states)
            self.cosine_probs[num] = predictions


    def play_game(self, thread):

        self.env.set_agents([self, self.rule_based_agent, self.rule_based_agent])
        length = 0
        while length < self.memory_length:
            trajectories, _ = self.env.run(is_training=True)
            for ts in trajectories[0]:
                self.save_to_memory(ts, thread) 
                length += 1

        
    def update_weights(self, thread):
        
        self.states, self.actions, self.rewards, self.next_states, self.dones, self.legal_actions = zip(*self.memory[thread][0:self.memory_length])
        self.rewards = self.discounted_rewards(self.rewards)

        self.states = np.stack(self.states, axis=0)
        self.actions = np.stack(self.actions, axis=0)
        self.rewards = np.stack(self.rewards, axis=0)
        self.next_states = np.stack(self.next_states, axis=0)
        self.dones = np.stack(self.dones, axis=0)

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

            self.values = self.critic(states)
            values_next = self.critic(next_state)
            self.loss_critic = tf.math.reduce_mean(tf.math.square(self.values-rewards))
            gamma = 0.95
            advantages = rewards - tf.reshape(self.values,-1) + gamma*tf.reshape(values_next, -1)*np.invert(done).astype(np.float32)

            self.probs = self.actor(states)
            
            entropy_coeff = 0.1
            z0 = tf.reduce_sum(self.probs, axis = 1)
            p0 = self.probs / tf.reshape(z0, (-1,1)) 
            entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
            mean_entropy = tf.reduce_mean(entropy) 
            self.entropy_loss =  mean_entropy * entropy_coeff 

            action_one_hot = tf.one_hot(actions, 309, on_value=1,off_value=0).numpy().astype(np.float32)
            actions_prob = tf.reduce_sum(tf.multiply(self.probs,action_one_hot), axis=1)
            action_log_probs =  tf.math.log(actions_prob+self.zero_fixer)
            
            actor_loss1 = action_log_probs * advantages 
            actor_loss1 = - tf.reduce_mean(actor_loss1) 
            
            self.loss_actor = actor_loss1 + self.entropy_loss 

            # Total loss
            self.loss_acc += self.loss_actor + self.loss_critic * 0.5

        self.history_actor = self.loss_actor 
        self.history_critic =  self.loss_critic
        self.actions = actions
        self.predictions  = tf.argmax(self.probs, axis=1).numpy()


    def cosine_similarity_actions(self):

        cosine_actions = np.zeros([self.batch_size, self.batch_size])

        for i in range(self.batch_size):
            for j in range(self.batch_size):
                score = 0
                v1, v2 = self.cosine_probs[i], self.cosine_probs[j]
        
                for x in range(10):
                    score += self.cos_between(v1[x],v2[x])

                cosine_actions[i][j] = score
                cosine_actions[j][i] = cosine_actions[i][j]

        return tf.reduce_sum(cosine_actions)


    def cos_between(self, v1, v2):

        v1_u = v1 / tf.norm(v1)
        v2_u = v2 / tf.norm(v2)

        return tf.tensordot(v1_u, v2_u, axes=1)

    def kernel_similarity():

        pass

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

    def discounted_rewards(self, reward):

        gamma = 0.95  # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward, dtype='float64')
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
        
        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) + self.zero_fixer
        return discounted_r
    
    def create_actor(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        output = Dense(action_num, activation='softmax')(x)
        network = keras.Model(inputs = input_x, outputs=output)
    
        return network
        
    def create_critic(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        output = Dense(action_num)(x)
        network = keras.Model(inputs = input_x, outputs=output)
        
        return network


class Hypernetwork(keras.Model):

    def __init__(self, input_size, output_size, seed):
        super().__init__()


        self.input_size = input_size 
        self.output_size = output_size
        self.seed = seed

        x = (512/2 + 512/2 +309/3 + 512/2 + 512/2 + 1)*15

        kernel_init = tf.keras.initializers.glorot_uniform(seed = self.seed)
        bias_init = tf.keras.initializers.constant(0)

        self.dense_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_2 = Dense(x, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)


        self.dense_w1_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w1_2 = Dense(601*2, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.dense_w2_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w2_2 = Dense(513*2, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.dense_w3_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w3_2 = Dense(513*3, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)


        self.dense_w4_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w4_2 = Dense(601*2, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.dense_w5_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w5_2 = Dense(513*2, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.dense_w6_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w6_2 = Dense(513, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
    

    def call(self, inputs, batch_size):

        layer_1 = 256 #512/2
        layer_2 = 256 #512/2
        layer_3 = 103 #309/3
        layer_4 = 1

        embedding = 15

        index1 = layer_1*embedding
        index2 = index1 +layer_2*embedding
        index3 = index2 + layer_3*embedding
        index4 = index3 + layer_1*embedding
        index5 = index4 + layer_2*embedding
        #index6 = index5 + layer_4*embedding

        x = self.dense_1(inputs)
        x = self.dense_2(x)

        input_w1 = x[:,:index1]
        input_w1 = tf.reshape(input_w1,(batch_size,layer_1,-1))
        w1 = self.dense_w1_1(input_w1)
        w1 = self.dense_w1_2(w1)

        input_w2 = x[:,index1:index2]
        input_w2 = tf.reshape(input_w2,(batch_size,layer_2,-1))
        w2 = self.dense_w2_1(input_w2)
        w2 = self.dense_w2_2(w2)
        
        input_w3 = x[:,index2:index3]
        input_w3 = tf.reshape(input_w3,(batch_size,layer_3,-1))
        w3 = self.dense_w3_1(input_w3)
        w3 = self.dense_w3_2(w3)

        input_w4 = x[:,index3:index4]
        input_w4 = tf.reshape(input_w4,(batch_size,layer_1,-1))
        w4 = self.dense_w4_1(input_w4)
        w4 = self.dense_w4_2(w4)

        input_w5 = x[:,index4:index5]
        input_w5 = tf.reshape(input_w5,(batch_size,layer_2,-1))
        w5 = self.dense_w5_1(input_w5)
        w5 = self.dense_w5_2(w5)
        
        input_w6 = x[:,index5]
        input_w6 = tf.reshape(input_w6,(batch_size,layer_4,-1))
        w6 = self.dense_w6_1(input_w6)
        w6 = self.dense_w6_2(w6)

        return w1, w2, w3, w4, w5, w6

