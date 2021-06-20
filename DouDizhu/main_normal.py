import os
import timeit
from datetime import datetime

from envs.mydoudizhu import DoudizhuEnv as Env
from utils.logger import Logger
from utils_global import tournament

from agents.non_rl.rule_based_agent import DouDizhuRuleAgentV1 as RuleAgent
from agents.non_rl.dummyrule_agent import DummyRuleAgent as RandomAgent


from agents.policy_based.a2c_tf import Actor_Critic
from agents.value_based.ddqn_tf import DQNAgent


config = {
    'seed': None,
    'use_conv': False,
    'allow_step_back': True,
    'allow_raw_data': True,
    'record_action': True,
    'single_agent_mode': False,
    'active_player': None,
}

replay_memory_size=200
replay_memory_init_size=100
update_target_estimator_every=1000,
discount_factor=0.99
epsilon_start=1.0
epsilon_end=0.1
epsilon_decay_steps=20000,
batch_size=64
action_num=2
state_shape=None
train_every=256
mlp_layers=None
initial_learning_rate=5e-5
decay_steps=20
decay_rate=0.99998

state_shape = [9, 5, 15]
env = Env(config, state_shape=state_shape, type='cooperation')
eval_env = Env(config, state_shape=state_shape, type='cooperation')
rule_agent = RuleAgent(action_num=eval_env.action_num)
random_agent = RandomAgent(action_num=eval_env.action_num)

# initialize rl_agents

lamBda = 0.2
batch_size = 2
loss = "dvd" # cos , none
lr = 1e-5

eval_every = 100
eval_num = 100
episode_num = 30_000

which_run = '1'
which_agent = 'a2c_new_1e5'
agent = Actor_Critic(state_shape=state_shape, initial_learning_rate = lr)


save_dir = f'./experiments/{which_run}/{which_agent}/'

def train_agent(agent, save_dir):
    
    logger = Logger(save_dir, 0)

    env.set_agents([agent, rule_agent, rule_agent])
    eval_env.set_agents([agent, rule_agent, rule_agent])


    for episode in range(episode_num + 1):

        # get transitions by playing an episode in envs
        trajectories, _ = env.run(is_training=True)

        for ts in trajectories[0]:
            agent.feed(ts) 

        # evaluate against random agent
        if episode % eval_every == 0:

            # Set agent's online network to evaluation mode
            result, states = tournament(eval_env, eval_num, agent)
            
            logger.log_performance(episode, result[0], agent.loss, 0, #states[0][-1][0]['raw_obs'],
                                agent.actions, agent.predictions, 0, 0, 0, 0)
            
            #logger.log_performance(episode, result, agent.history_actor.numpy(), agent.history_critic.numpy(), #states[0][-1][0]['raw_obs'],
             #                   agent.actions, agent.predictions, 0, 0, 0, 0)
            
            print(f'\nepisode: {episode}, result: {result}, '
                #f'epsilon: {agent.epsilon}, '
                #f'lr: {agent.lr_scheduler.get_lr()}'
            )

    # Close files in the logger and plot the learning curve
    logger.close_files()

train_agent(agent, save_dir)

lr = 7.5e-5
which_run = '1'
which_agent = 'ddqn_new_75e5'
agent = DQNAgent(state_shape=state_shape, learning_rate = lr)

save_dir = f'./experiments/{which_run}/{which_agent}/'


train_agent(agent, save_dir)


