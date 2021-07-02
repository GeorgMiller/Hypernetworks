import tensorflow as tf 
import keras

import agent_a2c
import agent_ddqn
import agent_genetic
import agent_ga
import logger 
import numpy as np 
import tensorflow.keras.initializers as init

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

epochs = 4
mini_batch_size = 2
decay_rate = 0.99998
learning_rate = 5e-4
lamBda = 0.2
weights_init = init.glorot_normal(42)
pretraining_steps = 0
batch_size = 2
kl_diversity = False
cosine_diversity = True

# For dvd diversity lambda 0.02 and learningrate 5e-4 it kinda worked. 

def normal_run_learning_rate():
    # This is the config to run the tests. First, four agents are initialized and trained for different learning_rates
    learning_rates = [1e-4]#, 6e-6, 3e-6]

    path = 'experiment/normal_run_learning_rate_dqn_new'

    for row, learning_rate in enumerate(learning_rates):

        worker = agent_ddqn.Agent(epochs,
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
                            row)
        worker.train_normal()

    
    path = 'experiment/normal_run_learning_rate_a2c_new'

    for row, learning_rate in enumerate(learning_rates):

        worker = agent_a2c.Agent(epochs,
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
                            row)
        worker.train_normal()



    

#normal_run_learning_rate()



def hypernetwork_learning_rate():
    # Then Hypernetwork is initialized and also trained for different learning rates with batch_size 1
    lamBdas = [1.5, 1.3, 1.15, 1, 0.85, 0.75]
    path = 'experiments/NEWWW_hypernetwork_learning_rate_ddqn_cos_lambdas_batch_size_2'

    for row, lamBda in enumerate(lamBdas):
        worker = agent_ddqn.Agent(epochs,
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
                            row)
        worker.train_hypernetwork()

hypernetwork_learning_rate()

def hypernetwork_weight_init():
    # The best learningrate is selected and different weight initializations tested
    weights_inits_num = [init.VarianceScaling(1, seed=seed), init.VarianceScaling(0.1,seed=seed), init.VarianceScaling(0.01,seed=seed), init.glorot_uniform(seed)]
    path = 'experiments/hypernetwork_weight_init_ddqn'

    for row, weights_init in enumerate(weights_inits_num):
        worker = agent_ddqn.Agent(epochs,
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
                            row)
        worker.train_hypernetwork()


def hypernetwork_pretraining():
    # All of them are also tested with pretraining
    pretraining_num = [100, 500, 1000]
    path = 'experiments/hypernetwork_pretraining'

    for row, pretraining_steps in enumerate(pretraining_num):
        worker = agent.Agent(epochs,
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
                            row)
        worker.train_hypernetwork()


def hypernetwork_batch_size():
    # Evaluate different batch_sizes
    batch_sizes = [4, 8, 16]
    path = 'experiments/hypernetwork_batch_size_ddqn'

    for row, batch_size in enumerate(batch_sizes):
        worker = agent_ddqn.Agent(epochs,
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
                            row)
        worker.train_hypernetwork()


def hypernetwork_kl_diversity():
    # When the best network type is found, it is used to evaluate the different diversity terms
    lamBdas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    path = 'experiments/hypernetwork_kl_diversity'
    kl_diversity = True

    for i, lamBda in enumerate(lamBdas):
        worker = agent.Agent(epochs,
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
                            i)
        worker.train_hypernetwork()

def hypernetwork_cosine_diversity():
# When the best network type is found, it is used to evaluate the different diversity terms
    lamBdas = [0.1, 0.5, 1]
    path = 'experiments/hypernetwork_cosine_diversity_ddqn'
    cosine_diversity = True

    for row, lamBda in enumerate(lamBdas):
        worker = agent_ddqn.Agent(epochs,
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
                            row)
        worker.train_hypernetwork()

def genetic_run(path):

    agent = agent_ga.Agent(path)
    agent.init_population()

    generations = 3000

    for i in range(generations):

        agent.train_generation()

path = "experiments_ga"
#genetic_run(path)

#hypernetwork_learning_rate()
#hypernetwork_weight_init()
#hypernetwork_pretraining()
#hypernetwork_kl_diversity()
#hypernetwork_cosine_diversity()
#hypernetwork_batch_size()

# Plotting
path = 'experiments/hypernetwork_learning_rate_ddqn_2'
logger.plot_diversity(path, 'A2C PPO entropy_loss', 4, 'steps', 'actor_loss')