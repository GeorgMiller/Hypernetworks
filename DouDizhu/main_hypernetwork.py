from tensorflow.python.keras.layers.core import Lambda
import agents.hypernetwork.hypernetwork_ddqn as Agent
import agents.hypernetwork.hypernetwork_a2c as A2C

import tensorflow as tf
import keras 
import numpy as np 
from keras.layers import Dense

import os
import timeit
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

from envs.mydoudizhu import DoudizhuEnv as Env
from utils.logger import Logger
from utils_global import tournament

### random or rule_based agents for evaluating ###
from agents.non_rl.rule_based_agent import DouDizhuRuleAgentV1 as RuleAgent

class HyperAgent(object):

    def __init__(self,
                training_steps,
                dvd,
                cos,
                batch_size,
                lr,
                path,
                lamBda):

        # Make environments to train and evaluate models
        config = {
                'seed': None,
                'use_conv': False,
                'allow_step_back': True,
                'allow_raw_data': False,
                'record_action': True,
                'single_agent_mode': False,
                'active_player': None,
                }
        self.zero_fixer = 1e-9
        self.total_t = 0
        self.train_t = 0
        self.training_steps = training_steps

        self.kernel_diversity = dvd
        self.cosine_diversity = cos
    
        self.input_noise_size = 4
        self.batch_size = batch_size
        self.seed = 42
        self.action_num = 309
        self.learning_rate = lr
        self.decay_steps = self.batch_size
        self.decay_rate = 0.999998
        self.update_every = 200

        self.eval_every = 100
        self.eval_num = 100

        self.state_shape = [9, 5, 15]

        self.env = Env(config, state_shape=self.state_shape, type='cooperation')
        self.eval_env = Env(config, state_shape=self.state_shape, type='cooperation')
        
        #self.agent = A2C.Agent(action_num=self.eval_env.action_num)
        self.agent = Agent.DQNAgent(action_num=self.eval_env.action_num)
        self.rule_agent = RuleAgent(action_num=self.eval_env.action_num)

        self.hypernetwork = Hypernetwork(self.input_noise_size,self.action_num,self.seed)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate, decay_steps=self.batch_size, decay_rate=self.decay_rate)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_schedule)

        self.row = 0
        self.path = path
        self.lamBda = lamBda
        self.logger = Logger(self.path, self.row)

    def train_hypernetwork(self):     

        # Start the actual hypernetwork training loop
        for step in range(self.training_steps):
                                 
            with tf.GradientTape() as tape:
                
                # In the beginning of the gradient, the accuracy loss is set to zero and new weights generated
                # Also make new lists for the cosine calculation of the probabilites
                z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,self.input_noise_size])
                self.embedding = [[] for _ in range(self.batch_size)] 
                self.loss_acc = 0
                w1, w2, w3 = self.hypernetwork(z,self.batch_size)
                
                # Reshape it for the online network and eventually update the target networks
                weights_online_network = tf.concat(axis=1, values=[tf.reshape(w1,(self.batch_size,-1)), tf.reshape(w2,(self.batch_size,-1)),tf.reshape(w3,(self.batch_size,-1))])
                if step % self.update_every == 0:
                    weights_target_network = weights_online_network
                
                self.env.set_agents([self.agent, self.rule_agent, self.rule_agent])
                trajectories, _ = self.env.run(is_training=True)
                _ = self.agent.target_network(np.expand_dims(trajectories[0][0][0]['obs'], 0))

                # For each worker first the weights are set and then one batch calculated and added to the overall loss
                for thread in range(self.batch_size):

                    self.agent.set_weights(weights_online_network, thread)
                    self.agent.set_target_weights(weights_target_network, thread)
                    self.env.set_agents([self.agent, self.rule_agent, self.rule_agent])
                    trajectories, _ = self.env.run(is_training=True)

                    for ts in trajectories[0]:
                        self.agent.feed(ts, thread) 
                   
                    if step >= 10:
                        self.loss_acc += self.agent.update_weights(thread)
                
                if step >= 10:
                    # If the number of workers generated is greater than one, diversity is calculated
                    if self.batch_size >= 1:

                        # Calculate the cosine similarity of the actions and the diversity of the weights
                        # To do that, first the predictions have to be chosen for the same states
                        self.evaluate_actions(weights_online_network)
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
                        loss = self.loss_div + tf.dtypes.cast(self.loss_acc, tf.float64)

                    # Otherwise the diversity loss term is set to zero
                    else:
                        self.loss_div = tf.constant(0.)
                        loss = self.loss_acc 
                        dvd, cos = tf.constant(0.), tf.constant(0.)

                    # Finally, the hypernetwork is updated
                    grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))


                if step % self.eval_every == 0 and step >= 10:

                    # Set agent's online network to evaluation mode
                    self.eval_env.set_agents([self.agent, self.rule_agent, self.rule_agent])

                    result, _ = tournament(self.eval_env, self.eval_num, self.agent)
                    self.logger.log_performance(step, result[0], self.loss_div.numpy(), self.loss_acc.numpy(), self.lamBda, loss.numpy(), \
                    self.optimizer._decayed_lr(tf.float32).numpy(),  dvd.numpy(), cos.numpy(), self.row)


        
        # Saving + End    
        #self.hypernetwork.save_weights('{}/hypernetwork_{}_{}.h5'.format(self.path,self.row,self.score))
        self.logger.close_files()

    
    def evaluate_actions(self, weights_online_network):

        e_states = []
        # This function is calculating the action embeddings for the cosine similarity and the kernel similarity
        # Select randomly a replay buffer batch_size x times and append mini_batch_size x times many states to the evaluation list
        for _ in range(self.batch_size):
            thread = np.random.randint(0,self.batch_size)
            states, _ , _ , _ , _, _  = self.agent.memories[thread].sample()
            e_states.append(states)

        # Then collect all the actions of all the agents for the states and save them to the embeddings-list
        for thread in range(self.batch_size):

            self.agent.set_weights(weights_online_network, thread)
            predictions = []
            for batch in e_states:
                predictions.append(self.agent.online_network(batch))
            
            predictions = tf.concat(predictions,axis=0)
            self.embedding[thread] = predictions



    def cosine_similarity_actions(self):

        cosine_actions = []
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                score = 0
                v1, v2 = self.embedding[i], self.embedding[j]
        
                for x in range(self.batch_size):
                    score += tf.dtypes.cast(self.cos_between(v1[x],v2[x])/self.batch_size, tf.float64)
                
                cosine_actions.append(score)

        return tf.reduce_mean(cosine_actions)


    def cos_between(self, v1, v2):

        v1_u = v1 / tf.norm(v1)
        v2_u = v2 / tf.norm(v2)
        
        return tf.tensordot(v1_u, v2_u, axes=1)


    def kernel_similarity(self):

        X = []
        for i in range(self.batch_size):
            X.append(tf.dtypes.cast(tf.reshape(self.embedding[i],-1), tf.float64))

        matrix = []
        for i in range(self.batch_size):
            for j in range(self.batch_size): 
                z = -tf.math.squared_difference(X[i],X[j])
                xz = tf.exp(z)
                matrix.append(tf.dtypes.cast(tf.reduce_mean(xz),tf.float64))
                    
        matrix = tf.reshape(matrix, (self.batch_size,self.batch_size))  
        determinant = tf.linalg.det(matrix)

        return - tf.math.log(determinant + self.zero_fixer)


class Hypernetwork(keras.Model):

    def __init__(self, input_size, output_size, seed):
        super().__init__()


        self.input_size = input_size 
        self.output_size = output_size
        self.seed = seed

        x = (512 + 512 + 309)*15

        kernel_init = tf.keras.initializers.glorot_uniform(seed = self.seed)
        bias_init = tf.keras.initializers.constant(0)

        self.dense_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_2 = Dense(x, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.dense_w1_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w1_2 = Dense(676, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.dense_w2_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w2_2 = Dense(513, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.dense_w3_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_w3_2 = Dense(513, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)


    def call(self, inputs, batch_size):

        layer_1 = 512 
        layer_2 = 512 
        layer_3 = 309 

        embedding = 15

        index1 = layer_1*embedding
        index2 = index1 +layer_2*embedding
        index3 = index2 + layer_3*embedding 

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

        return w1, w2, w3


training_steps = 30_000
dvd = True
cos = False
batch_size = 2 
lr = 1e-5
lamBda = 0.2

lamBdas = [1e-3, 5e-3]

for lamBda in lamBdas:
    #path = "exp_new_5e6_dvd_{}".format(lamBda)
    #path = "exp_new_try{}".format(lamBda)

    agent = HyperAgent(training_steps, dvd, cos, batch_size, lr, path, lamBda)
    agent.train_hypernetwork()

training_steps = 30_000
dvd = False
cos = True
batch_size = 2 
lr = 1e-5
lamBda = 0.2

lamBdas = [1e-2, 1e-3]

#for lamBda in lamBdas:
    #path = "exp_new_5e6_cos_{}".format(lamBda)
    #agent = HyperAgent(training_steps, dvd, cos, batch_size, lr, path, lamBda)
    #agent.train_hypernetwork()

def plot():

    ddqn1_path = 'exp_new_lr_1e-05/0_performance.csv'
    ddqn2_path = 'experiments/1/ddqn_new_75e5/0_performance.csv'
    algorithm1a = 'A2C reward'
    algorithm2a = 'DDQN reward'
    algorithm1b = 'DvD loss'
    algorithm2b = 'DDQN percentage right'

    with open(ddqn1_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['value_loss']))
            ys2.append(float(row['value_loss']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn2_path) as csvfile:#
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        for row in reader:
            zs1.append(float(row['reward']))
            zs2.append(float(row['dvd']))

    ax.plot(xs, ys1, label=algorithm1a, c='b')
    #ax.plot(xs, zs1, label=algorithm2a, c='g')

    ax2.plot(xs, ys2, label=algorithm1b, c='r')
    #ax2.plot(xs, zs2, label=algorithm2b, c='y')
    
    ax.set(xlabel='episode', ylabel='reward')
    ax2.set(xlabel='episode', ylabel='cosine loss')
    ax.legend(loc='upper left')
    #ax2.legend(loc='lower right')
    ax.grid()
    #ax2.grid()
    save_path = 'experiments' + '/hyper_losses.png'
    plt.savefig(save_path)
    plt.show()


plot()