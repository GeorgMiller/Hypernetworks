import numpy as np
from random import sample
import torch
import torch.nn as nn

from utils_global import remove_illegal
from agents.networks import DRQN

"""
Deep Recurrent Q-Learning agent.
parameters:
    num_actions (int): number of possible actions
    state_shape (list): tensor shape of the state
    recurrent_layer_size (int): size of LSTM hidden states
    recurrent_layers_num (int): number of LSTM layers to use in recurrent network
    hidden_layers (list): fc hidden layer size 
    lr (float): learning rate
    memory_size (int): max number of experiences to store in memory buffer
    update_every (int): how often to copy parameters to target network
    epsilons (list): list of epsilon values to using over training period
    shift_epsilon_every (int): how often to shift epsilon value
    gamma (float): discount parameter
    device (torch.device): device to put models on
"""


class DRQNAgent:
    def __init__(self,
                 num_actions,
                 state_shape,
                 recurrent_layer_size=256,
                 recurrent_layers_num=2,
                 mlp_layers=None,
                 lr=0.0001,
                 batch_size=32,
                 memory_init_size=100,
                 memory_size=50000,
                 update_every=100,
                 train_every=1,
                 epsilons=None,
                 epsilon_decay_steps=2000,
                 gamma=0.99,
                 device=None,
                 ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.update_every = update_every
        self.train_every = train_every
        self.memory_init_size = memory_init_size
        self.epsilons = np.linspace(0.5, 0.1, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.discount_factor = gamma
        self.device = device
        self.use_raw = False

        # initialize learner and target networks
        self.q_net = DRQN(state_shape=state_shape,
                          num_actions=num_actions,
                          recurrent_layer_size=recurrent_layer_size,
                          recurrent_layers_num=recurrent_layers_num,
                          mlp_layers=mlp_layers,
                          ).to(self.device)
        self.q_net.eval()
        self.target_net = DRQN(state_shape=state_shape,
                               num_actions=num_actions,
                               recurrent_layer_size=recurrent_layer_size,
                               recurrent_layers_num=recurrent_layers_num,
                               mlp_layers=mlp_layers,
                               ).to(self.device)
        self.target_net.eval()

        # initialize optimizer for learner q network
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # initialize loss func for network
        self.loss = nn.MSELoss(reduction='mean').to(self.device)

        # timesteps
        self.timestep = 0

        # initialize memory buffer
        self.memory_buffer = SequentialMemory(memory_size, batch_size)
        self.softmax = nn.Softmax(dim=-1)

        self.device = device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def reset_hidden(self):
        self.q_net.init_hidden(size=1)
        self.target_net.init_hidden(size=1)

    def predict(self, state):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            legal_actions = state['legal_actions']
            # calculate a softmax distribution over the q_values for all actions
            # q_vals = self.online_net(state_obs).cpu().detach().numpy()
            softmax_q_vals = self.softmax(self.q_net(state_obs))[0][0].cpu().detach().numpy()
            predicted_action = np.argmax(softmax_q_vals)
            probs = remove_illegal(softmax_q_vals, legal_actions)
            max_action = np.argmax(probs)

        return probs, max_action

    def step(self, state):
        """
        Given state, produce actions to generate training data. Use epsilon greedy action selection.
        Should be separate from compute graph as we only update through the feed function.
        Uses epsilon greedy methods in order to produce the action.
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
            Output:
                action (int) : integer representing action id
        """

        epsilon = self.epsilons[min(self.timestep, self.epsilon_decay_steps - 1)]

        legal_actions = state['legal_actions']
        max_action = self.predict(state)[1]
        if np.random.uniform() < epsilon:
            probs = remove_illegal(np.ones(self.num_actions), legal_actions)
            action = np.random.choice(self.num_actions, p=probs)
        else:
            action = max_action

        return action

    def eval_step(self, state, use_max=True):
        """
        Pick an action given a state. This is to be used during evaluation, so no epsilon greedy.
        Makes call to eval_pick_action to actually select the action
        Pick an action given a state according to max q value.
        Input:
            state (dict)
                'obs' : actual state representation
                'legal_actions' : possible legal actions to be taken from this state
        Output:
            action (int) : integer representing action id
            probs (np.array) : softmax distribution over the actions
        """

        probs, max_action = self.predict(state)
        if use_max:
            action = max_action
        else:
            action = np.random.choice(self.num_actions, p=probs)

        return action, probs

    def add_transition(self, sequence):
        """
        add transition to memory buffer and train the network one batch
        Input:
            transition (tuple): tuple representation of a transition --> (state, action, reward, next_state, done)
        Output:
            Nothing.
            stores transition in buffer, updates network using memory buffer,
            and updates target network denpending on timesteps
        """

        # store sequence of transitions in memory
        if len(sequence) > 0:
            self.memory_buffer.add_seq_transition(sequence)
            self.timestep += 1

            # once we have enough samples, get a sample from stored memory and train the network
            if self.timestep >= self.memory_init_size and self.timestep % self.train_every == 0:
                batch_loss = self.train()
                print(f'\rstep: {self.timestep}, loss on batch: {batch_loss}', end='')

        # copy parameters over target network once in a while
        if self.timestep % self.update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()
            print(f'target_network parameters updated at step {self.timestep}')

    def train(self):
        """
        sample from memory buffer and train the network one step
        Input:
            Nothing.
            Draw sample of sequence from memory buffer to train the network
        Output:
            loss (float): loss on training batch
        """

        sequences = self.memory_buffer.sample()

        self.q_net.train()
        self.optim.zero_grad()

        batch_loss = 0
        for sequence in sequences:
            self.reset_hidden()
            states = [t[0]['obs'] for t in sequence] + [sequence[-1][3]['obs']]
            actions = [t[1] for t in sequence]
            rewards = [t[2] for t in sequence]
            states = torch.FloatTensor(states).view(len(states), 1, -1).to(self.device)

            actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
            rewards = torch.FloatTensor(rewards).view(-1).to(self.device)

            # with no gradient, calculate 'true values' for each state in sequence using target network
            with torch.no_grad():
                target_q_values = self.target_net(states).detach()
                print(target_q_values)

                target_q_values, max_actions = target_q_values.max(-1)
                target_q_values = rewards + self.discount_factor * target_q_values[1:].view(-1)
                target_q_values[-1] = 0.0

            q_values = self.q_net(states).squeeze(1)[:-1]
            q_values = q_values.gather(-1, actions).view(-1)
            batch_loss += self.loss(q_values, target_q_values)

        batch_loss.backward()
        self.optim.step()
        self.q_net.eval()
        return batch_loss.item()

    def save_state_dict(self, filepath):
        """
        save state_dict for networks for dqn agents
        input:
            filepath (str): string filepath to save agent parameters at
        """
        state_dict = dict()
        state_dict['online_net'] = self.q_net.state_dict()
        state_dict['target_net'] = self.target_net.state_dict()

        torch.save(state_dict, filepath)

    def load_from_state_dict(self, filepath):
        """
        load agent parameters from filepath
        input:
            filepath (str): string filepath to load agent parameters from
        """
        state_dict = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(state_dict['online_net'])
        self.target_net.load_state_dict(state_dict['target_net'])


class SequentialMemory(object):
    """
        sequential memory implementation for recurrent q network
        save a series of transitions to use as training examples for the recurrent network
    """

    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.memory = []

    def add_seq_transition(self, seq):
        if len(self.memory) == self.max_size:
            self.memory.pop(0)
        self.memory.append(seq)

    def sample(self):
        return sample(self.memory, self.batch_size)
