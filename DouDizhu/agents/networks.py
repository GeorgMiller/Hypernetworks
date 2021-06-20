import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.nn.functional as F

from functools import reduce
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class cReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)


class DQN(nn.Module):
    def __init__(self, state_shape, num_actions, use_conv=False):
        super(DQN, self).__init__()

        self.flatten = Flatten()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.use_conv = use_conv
        if self.use_conv:

            self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 32, kernel_size=4, stride=1),
                cReLU(),
                #nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 32, kernel_size=1, stride=1),
                #nn.LeakyReLU(),
                cReLU(),
                nn.BatchNorm2d(64),
            )

            self.fc = nn.Sequential(
                nn.Linear(self._feature_size(), 512),
                nn.ReLU(),
                #nn.Linear(512, 512),
                #nn.ReLU(),
                nn.Linear(512, self.num_actions),
            )
        else:
            flattened_state_shape = reduce(lambda x, y: x * y, self.state_shape)
            self.fc = nn.Sequential(
                nn.Linear(flattened_state_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
            )

    def forward(self, state):
        if self.use_conv:
            x = self.features(state)
            x = self.flatten(x)

        else:
            x = torch.flatten(state, start_dim=1)

        q_values = self.fc(x)
        return q_values

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)


class DuelingDQN(nn.Module):
    """
    Duelling networks predicting state values and advantage functions to form the Q-values
    the advantage Aπ(s,a) of an action: Aπ(s,a) = Qπ(s,a) − Vπ(s)
    the advantage function has less variance than the Q-values,
    only tracks the relative change of the value of an action compared to its state, much more stable over time.
    """

    def __init__(self, state_shape, num_actions, use_conv):
        super(DuelingDQN, self).__init__()

        self.flatten = Flatten()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.use_conv = use_conv

        if self.use_conv:

            self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 32, kernel_size=4, stride=1),
                cReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                cReLU(),
                nn.BatchNorm2d(128),
            )

            self.fc = nn.Sequential(
                nn.Linear(self._feature_size(), 512),
                nn.ReLU(),
            )
        else:
            flattened_state_shape = reduce(lambda x, y: x * y, state_shape)
            self.fc = nn.Sequential(
                nn.Linear(flattened_state_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
            )

        self.advantage = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        if self.use_conv:
            x = self.features(state)
            x = self.flatten(x)
        else:
            x = torch.flatten(state, start_dim=1)
        x = self.fc(x)
        advantage = self.advantage(x)
        value = self.value(x)
        adv_average = torch.mean(advantage, dim=1, keepdim=True)
        q_values = advantage + value - adv_average

        return q_values

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.output_dim, self.input_dim))

        # extra parameter for the bias and register buffer for the bias parameter
        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.output_dim))

        # reset parameter as initialization of the layer
        self.reset_parameter()
        self.reset_noise()

    def forward(self, x):
        """
        sample random noise in sigma weight buffer and bias buffer
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        mu_range = 1 / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))

    def _scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))


class DQNNoisy(nn.Module):

    def __init__(self, state_shape, num_actions):
        super(DQNNoisy, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        flattened_state_shape = reduce(lambda x, y: x * y, self.state_shape)
        # Apply noisy network on fully connected layers
        self.noisy1 = NoisyLinear(512, 512)
        self.noisy2 = NoisyLinear(512, self.num_actions)

        self.fc = nn.Sequential(
            nn.Linear(flattened_state_shape, 512),
            nn.ReLU(),
            self.noisy1,
            nn.ReLU(),
            self.noisy2)

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        x = self.fc(state)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class CategoricalDQN(nn.Module):
    """
     the output layer predicts the distribution of the returns for each action a in state s, instead of its mean Qπ(s,a)
    """

    def __init__(self, state_shape, num_actions, num_atoms=51, v_min=-10., v_max=10., use_conv=False):
        """Initialize parameters and build model.
                Params
                    state_size (int): Dimension of each state
                    action_size (int): Dimension of each action
        """
        super(CategoricalDQN, self).__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.use_conv = use_conv
        self.flatten = Flatten()
        self.softmax = nn.Softmax(dim=-1)

        flattened_state_shape = reduce(lambda x, y: x * y, state_shape)

        if self.use_conv:
            self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 32, kernel_size=4, stride=1),
                nn.BatchNorm2d(32),
                cReLU(),
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.BatchNorm2d(64),
                cReLU(),
            )

            self.fc = nn.Sequential(
                nn.Linear(self._feature_size(), 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions * self.num_atoms)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(flattened_state_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions * self.num_atoms),
            )

    def forward(self, state):
        """
        the Q-value is easy to compute as the mean of the distribution
        """

        if self.use_conv:
            x = self.features(state)
            x = self.flatten(x)

        else:
            x = torch.flatten(state, start_dim=1)

        # [batch_size, num_actions, num_atoms)
        dist = self.fc(x)
        dist = F.softmax(dist.view(-1, self.num_atoms), 1).view(-1, self.num_actions, self.num_atoms)
        return dist

    """ 
    def act(self, state):
        dist = self.forward(state)
        dist = dist.detach()
        dist = dist.mul(torch.linspace(self.v_min, self.v_max, self.num_atoms))
        action = dist.sum(2).max(1)[1].detach()[0].item()
        return dist, action
    """

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)


class RainbowDQN(nn.Module):
    def __init__(self, state_shape, num_actions, num_atoms=51, v_min=-10., v_max=10., use_conv=False):
        super(RainbowDQN, self).__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.use_conv = use_conv
        self.flatten = Flatten()
        self.softmax = nn.Softmax(dim=-1)
        self.adv_noisy1 = NoisyLinear(512, 512)
        self.adv_noisy2 = NoisyLinear(512, self.num_actions * self.num_atoms)
        self.value_noisy1 = NoisyLinear(512, 512)
        self.value_noisy2 = NoisyLinear(512, self.num_atoms)

        flattened_state_shape = reduce(lambda x, y: x * y, state_shape)
        if self.use_conv:
            self.features = nn.Sequential(
                nn.Conv2d(self.state_shape[0], 32, kernel_size=4, stride=1),
                nn.BatchNorm2d(32),
                cReLU(),
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.BatchNorm2d(64),
                cReLU(),
            )

            self.fc = nn.Sequential(
                nn.Linear(self._feature_size(), 512),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(flattened_state_shape, 512),
                nn.ReLU(),
            )

        self.advantage = nn.Sequential(
            self.adv_noisy1,
            nn.ReLU(),
            self.adv_noisy2,
        )

        self.value = nn.Sequential(
            self.value_noisy1,
            nn.ReLU(),
            self.value_noisy2,
        )

    def forward(self, state):
        if self.use_conv:
            x = self.features(state)
            x = self.flatten(x)
        else:
            x = torch.flatten(state, start_dim=1)

        x = self.fc(x)
        advantage = self.advantage(x)
        value = self.value(x)
        advantage = advantage.view(-1, self.num_actions, self.num_atoms)
        value = value.view(-1, 1, self.num_atoms)
        adv_average = torch.mean(advantage, dim=1, keepdim=True)

        dist = advantage + value - adv_average
        dist = self.softmax(dist)

        return dist

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)

    def reset_noise(self):
        self.adv_noisy1.reset_noise()
        self.adv_noisy2.reset_noise()
        self.value_noisy1.reset_noise()
        self.value_noisy2.reset_noise()


class DRQN(nn.Module):
    """
    Recurrent Q network in pytorch
    parameters:
        state_shape (list of int): shape of the state
        num_actions (int) : number of possible actions that an agent can take
        recurrent_layer_size (int) : size of hidden state of recurrent layer
        recurrent_layers_num (int) : number of recurrent layers to use
        mlp_layers (list): list of mlp hidden layer sizes
        describing the fully connected network from the hidden state to output
        activation (str): which activation func to use? 'tanh' or 'relu'
    """

    def __init__(self, state_shape, num_actions,
                 recurrent_layer_size, recurrent_layers_num,
                 mlp_layers, activation='relu'):
        super(DRQN, self).__init__()

        # initialize lstm layers
        self.flattened_state_size = reduce(lambda x, y: x * y, state_shape)
        self.recurrent_layer_size = recurrent_layer_size
        self.recurrent_layers_num = recurrent_layers_num
        self.lstm_layers = LSTM(input_size=self.flattened_state_size,
                                hidden_size=recurrent_layer_size,
                                num_layers=recurrent_layers_num,
                                )

        layer_dims = [recurrent_layer_size] + mlp_layers + [num_actions]
        fc_layers = []

        for i in range(len(layer_dims) - 2):
            fc_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if activation == 'relu':
                fc_layers.append(nn.ReLU())
            else:
                fc_layers.append(nn.Tanh())

        fc_layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.fc_layers = nn.Sequential(*fc_layers)
        self.init_hidden(1)

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        x, (self.hidden, self.cell) = \
            self.lstm_layers(state.view(-1, 1, self.flattened_state_size), (self.hidden, self.cell))

        q_values = self.fc_layers(x)

        return q_values

    def init_hidden(self, size):
        self.hidden = torch.zeros(self.recurrent_layers_num, size, self.recurrent_layer_size)
        self.cell = torch.zeros(self.recurrent_layers_num, size, self.recurrent_layer_size)


class AveragePolicyNet(DQN):
    def __init__(self, state_shape, num_actions, use_conv=True):
        super(AveragePolicyNet, self).__init__(state_shape, num_actions)
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.use_conv = use_conv

        if self.use_conv:
            self.fc = nn.Sequential(
                nn.Linear(self._feature_size(), 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
                nn.Softmax(dim=-1)
            )
        else:
            flattened_state_shape = reduce(lambda x, y: x * y, self.state_shape)
            self.fc = nn.Sequential(
                nn.Linear(flattened_state_shape, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
                nn.Softmax(dim=-1)
            )


class DeepConvNet(nn.Module):
    def __init__(self, state_shape, action_num):
        self.state_shape = state_shape
        self.action_num = action_num
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv2d(self.state_shape[0], 64, (1, 1), (4, 1))
        self.conv2 = nn.Conv2d(self.state_shape[0], 64, (2, 1), (4, 1))
        self.conv3 = nn.Conv2d(self.state_shape[0], 64, (3, 1), (4, 1))
        self.conv4 = nn.Conv2d(self.state_shape[0], 64, (4, 1), (4, 1))
        self.conv_single_to_bomb = (self.conv1, self.conv2, self.conv3, self.conv4)
        self.conv_straight = nn.Conv2d(state_shape[0], 64, (1, 15), 1)
        self.pool = nn.MaxPool2d((4, 1))
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * (4 + 15), 512)
        self.fc2 = nn.Linear(512, self.action_num)

    def forward(self, state):
        # 64 * 4 * 15
        x = torch.cat([f(state) for f in self.conv_single_to_bomb], -2)
        # 64 * 1 * 15
        x = self.pool(x)
        x = x.view(state.shape[0], -1)

        # 64 * 4 * 1
        x_straight = self.conv_straight(state)
        x_straight = x_straight.view(state.shape[0], -1)

        x = torch.cat([x, x_straight], -1)

        #x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
