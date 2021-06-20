import numpy as np
import random
import os
import tensorflow as tf

from utils_global import remove_illegal
import agents.non_rl.rule_based_agent as rule_based_agent
import envs.mydoudizhu as env
import utils.logger_hypernetwork as logger
from utils_global import set_global_seed, tournament
from agents.policy_based.a2c_tf import Actor_Critic
from agents.value_based.ddqn_tf import DQNAgent


class Genetic_Algorithm():

    def __init__(self, 
                 population_size, 
                 elite_workers_num,
                 evaluation_num,
                 generations,
                 epsilon, 
                 state_shape,
                 config,
                 model_load_path,
                 model_save_path,
                 model_save_path_dir,
                 log_dir,
                 training,
                 _type,
                 train_episode_num,
                 num_threads,
                 load_data,
                 row,
                 weight_space):
        
        self.config = config
        self.model_load_path = model_load_path
        self.model_save_path = model_save_path
        self.model_save_path_dir = model_save_path_dir
        self.log_dir = log_dir

        if not os.path.exists(self.model_save_path_dir):
            os.makedirs(self.model_save_path_dir)

        self.training = training
        self.random = random
        self.type = _type
        self.train_episode_num = train_episode_num
        self.num_threads = num_threads
        self.load_data = load_data
        self.row = row
        self.state_shape = state_shape
        self.population_size = population_size
        self.elite_workers_num = elite_workers_num
        self.evaluate_num = evaluation_num
        self.generations = generations
        self.epsilon = epsilon
        self.weight_space = weight_space
        
        self.logger = logger.Logger(self.log_dir, self.row)      
        self.env = env.DoudizhuEnv(self.config, self.state_shape)
        self.dqn_agent = DQNAgent(action_num=self.env.action_num)
        self.a2c_agent = Actor_Critic(action_num=self.env.action_num)
        self.rule_based_agent = rule_based_agent.DouDizhuRuleAgentV1(action_num=self.env.action_num)
        self.workers = []
        self.sigma = 0.01
        self.lamBda = 0.0
        self.learning_rate = 0.05
        self.eps = population_size

    def set_parameters_dqn(self, weights):
        
        last_used = 0

        for i in range(len(self.dqn_agent.q_estimator.layers)):

            if 'dense' in self.dqn_agent.q_estimator.layers[i].name:
                weights_shape = self.dqn_agent.q_estimator.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.dqn_agent.q_estimator.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.dqn_agent.q_estimator.layer[i].set_weigths([new_weights, new_weights_bias])
                last_used += no_of_weights_bias

        self.dqn_agent.target_estimator.set_weights(self.dqn_agent.q_estimator.get_weights())

    def set_parameters_a2c(self, weights):

        last_used = 0
                
        for i in range(len(self.a2c_agent.actor.layers)):

            if 'dense' in self.a2c_agent.actor.layers[i].name:
                weights_shape = self.a2c_agent.actor.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.a2c_agent.actor.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.a2c_agent.actor.layers[i].set_weights([new_weights,new_weights_bias])
                last_used += no_of_weights_bias

        for i in range(len(self.a2c_agent.critic.layers)):

            if 'dense' in self.a2c_agent.critic.layers[i].name:
                weights_shape = self.a2c_agent.critic.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.a2c_agent.critic.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.a2c_agent.critic.layers[i].set_weights([new_weights,new_weights_bias])
                last_used += no_of_weights_bias
        
    def initial_population(self):

        # These are hyperparameters and can be changed
        fan_in = 6400*4
        np.random.seed(42)

        for _ in range(self.population_size):
            
            z = np.random.uniform(low=-np.sqrt(6/fan_in), high=np.sqrt(6/fan_in), size=self.weight_space)
            self.workers.append(z)
        
    def evaluate_population(self):

        rewards = []
        
        self.epsilons = self.sigma**2 * np.random.normal(0,1,(self.population_size,self.weight_space))

        for idx, worker in enumerate(self.workers):
            
            if self.type == 'dqn':
                
                self.set_parameters_dqn(worker + self.epsilons[idx])
                self.env.set_agents([self.dqn_agent, self.rule_based_agent, self.rule_based_agent])
            
                if self.training == True:
                    for _ in range(self.train_episode_num):
                        trajectories, _ = self.env.run(is_training=True)
                    for ts in trajectories[0]:
                        self.dqn_agent.feed(ts)

                payoff = tournament(self.env, self.evaluate_num,self.dqn_agent)[0]
                rewards.append(payoff)
            
            else: 
                self.set_parameters_a2c(worker + self.epsilons[idx])
                self.env.set_agents([self.a2c_agent, self.rule_based_agent, self.rule_based_agent])

                if self.training == True:
                    for _ in range(self.train_episode_num):
                        trajectories, _ = self.env.run(is_training=True)
                    for ts in trajectories[0]:
                        self.a2c_agent.feed(ts)

                payoff = tournament(self.env, self.evaluate_num,self.a2c_agent)[0]
                rewards.append(payoff[0])
        
        rewards = np.array(rewards)
        diversity = 0

        return rewards, diversity

    def update_population(self):

        for idx in range(self.population_size):
            
            gradient = np.mean((1-self.lamBda)*self.rewards[idx]) 
            gradient = (1/(self.sigma*self.eps))*gradient*self.epsilons[idx]# + self.lamBda * self.dvd[i])*e
            self.workers[idx] = self.workers[idx] + gradient * self.learning_rate
        
    def run(self):

        random.seed(42)
        self.initial_population()
        max_reward = 0
        
        for i in range(self.generations):
            
            self.rewards, self.diversity = self.evaluate_population()

            self.update_population()

            print('these are the scores', self.rewards, 'and this is the generation', i , 'and diversity', self.diversity)
            
            #self.logger.log_performance(i, rewards.mean(), self.agent.history_actor, self.agent.history_critic, self.agent.optimizer._decayed_lr(tf.float32).numpy(), self.agent.actions, self.agent.predictions)

            
            if i % 1 == 0:

                max_reward = self.rewards.mean()
                '''
                path = '{}{:.3f}.txt'.format(self.model_save_path,max_reward)
                with open(path, 'w') as data:
                    for x in elite_workers:
                        data.write('{}\n'.format(x))
                data.close()

                
                path_a = '{}_actor_{:.3f}.h5'.format(self.model_save_path,max_reward)
                path_c = '{}_critic_{:.3f}.h5'.format(self.model_save_path,max_reward)


                if self.type == 'a2c':
                    self.a2c_agent.critic.save(path_a)
                    self.a2c_agent.actor.save(path_c)
                else:
                    self.dqn_agent.q_estimator.save(path)
                '''
            if i == 30:
                self.random = False
            
            elite_workers, rewards = [],[]

