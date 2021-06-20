import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.nn.functional as F

from functools import reduce
import numpy as np


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class cReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)


class NoisyLinear(nn.Module):
    # NoisyNet layer with factorized gaussian noise
    def __init__(self, input_dim, output_dim, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init
        # set this to false in the act method after the training is over
        self.training = True

        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        # Adding noise to weight that will not be trained
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.output_dim, self.input_dim))

        # extra parameter for the bias and register buffer for the bias parameter
        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output_dim))
        # Adding noise to bias that will not be trained
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.output_dim))

        # reset mu and sigma
        self.reset_parameter()
        # reset epsilon
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
        # Initialize mu and sigma for weight and bias

        mu_range = 1 / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))

    def reset_noise(self):
        # Generate noise for epsilon. These noise weights are not trainable by the model
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        # .ger() gives the matrix out of 2 vector multiplication
        # epsilon_out is out*1, epsilon_in is in*1, the out of .ger() is out*in
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        # do not use the same random value for bias. call scale_noise once again for bias
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))

    def _scale_noise(self, size):
        # Gives the noise output of a particular size after scaling
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise


class DQN(nn.Module):
    def __init__(self, state_shape, num_actions, use_conv=False, layer_type=None):
        super(DQN, self).__init__()

        self.use_conv = use_conv

        flattened_state_shape = reduce(lambda x, y: x * y, state_shape)

        self.fc1 = nn.Linear(flattened_state_shape, 512)
        if layer_type == 'noisy':

            self.fc2 = NoisyLinear(512, 512)
            self.fc3 = NoisyLinear(512, num_actions)

        else:

            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, num_actions)

    def forward(self, state):

        x = torch.flatten(state, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def reset_noise(self):
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class DuelingDQN(nn.Module):
    """
    Duelling networks predicting state values and advantage functions to form the Q-values
    the advantage Aπ(s,a) of an action: Aπ(s,a) = Qπ(s,a) − Vπ(s)
    the advantage function has less variance than the Q-values,
    only tracks the relative change of the value of an action compared to its state, much more stable over time.
    """

    def __init__(self, state_shape, num_actions, use_conv=False, layer_type=None):
        super(DuelingDQN, self).__init__()

        self.use_conv = use_conv

        flattened_state_shape = reduce(lambda x, y: x * y, state_shape)

        self.fc1 = nn.Linear(flattened_state_shape, 512)
        if layer_type == 'noisy':

            self.fc2_adv = NoisyLinear(512, 512)
            self.fc2_val = NoisyLinear(512, 512)
            self.advantage = NoisyLinear(512, num_actions)
            self.value = NoisyLinear(512, 1)

        else:

            self.fc2_adv = nn.Linear(512, 512)
            self.fc2_val = nn.Linear(512, 512)
            self.advantage = nn.Linear(512, num_actions)
            self.value = nn.Linear(512, 1)

    def forward(self, state):
        x = torch.flatten(state, start_dim=1)
        x = F.relu(self.fc1(x))
        x_a = F.relu(self.fc2_adv(x))
        x_v = F.relu(self.fc2_val(x))
        advantage = self.advantage(x_a)
        value = self.value(x_v)

        adv_average = torch.mean(advantage, dim=1, keepdim=True)
        q_values = advantage + value - adv_average

        return q_values

    def reset_noise(self):

        self.fc2_val.reset_noise()
        self.fc2_adv.reset_noise()
        self.advantage.reset_noise()
        self.value.reset_noise()


class C51DQN(nn.Module):
    """
     the output layer predicts the distribution of the returns for each action a in state s, instead of its mean Qπ(s,a)
    """

    def __init__(self, state_shape, num_actions, num_atoms=51, use_conv=False, layer_type=None):
        """Initialize parameters and build model.
                Params
                    state_size (int): Dimension of each state
                    action_size (int): Dimension of each action
        """
        super(C51DQN, self).__init__()

        self.use_conv = use_conv
        self.num_atoms = num_atoms
        self.num_actions = num_actions
        self.flatten = Flatten()

        flattened_state_shape = reduce(lambda x, y: x * y, state_shape)

        self.fc1 = nn.Linear(flattened_state_shape, 512)
        if layer_type == 'noisy':

            self.fc2 = NoisyLinear(512, 512)
            self.fc3 = NoisyLinear(512, self.num_actions * self.num_atoms)

        else:
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, self.num_actions * self.num_atoms)

    def forward(self, state):
        x = torch.flatten(state, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dist = self.fc3(x)

        # [batch_size, num_actions, num_atoms)
        dist = F.softmax(dist.view(-1, self.num_atoms), 1).view(-1, self.num_actions, self.num_atoms)

        return dist

    def reset_noise(self):
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class C51DuelDQN(nn.Module):
    def __init__(self, state_shape, num_actions, num_atoms=51, use_conv=False, layer_type='noisy'):
        super(C51DuelDQN, self).__init__()

        self.use_conv = use_conv
        self.num_atoms = num_atoms
        self.num_actions = num_actions

        flattened_state_shape = reduce(lambda x, y: x * y, state_shape)

        self.fc1 = nn.Linear(flattened_state_shape, 512)
        if layer_type == 'noisy':

            self.fc2_adv = NoisyLinear(512, 512)
            self.fc2_val = NoisyLinear(512, 512)
            self.advantage = NoisyLinear(512, self.num_actions * self.num_atoms)
            self.value = NoisyLinear(512, self.num_atoms)
        else:
            self.fc2_adv = nn.Linear(512, 512)
            self.fc2_val = nn.Linear(512, 512)
            self.advantage = nn.Linear(512, self.num_actions * self.num_atoms)
            self.value = nn.Linear(512, self.num_atoms)

    def forward(self, state):
        x = torch.flatten(state, start_dim=1)
        x = F.relu(self.fc1(x))
        x_a = F.relu(self.fc2_adv(x))
        x_v = F.relu(self.fc2_val(x))

        advantage = self.advantage(x_a)
        value = self.value(x_v)
        advantage = advantage.view(-1, self.num_actions, self.num_atoms)
        value = value.view(-1, 1, self.num_atoms)
        adv_average = torch.mean(advantage, dim=1, keepdim=True)

        dist = advantage + value - adv_average
        dist = F.softmax(dist.view(-1, self.num_atoms), 1).view(-1, self.num_actions, self.num_atoms)

        return dist

    def reset_noise(self):

        self.fc2_val.reset_noise()
        self.fc2_adv.reset_noise()
        self.advantage.reset_noise()
        self.value.reset_noise()


class DeepConvNet(nn.Module):
    def __init__(self, state_shape, action_num, kernels):

        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], kernels, (1, 1), (4, 1))
        self.conv2 = nn.Conv2d(state_shape[0], kernels, (2, 1), (4, 1))
        self.conv3 = nn.Conv2d(state_shape[0], kernels, (3, 1), (4, 1))
        self.conv4 = nn.Conv2d(state_shape[0], kernels, (4, 1), (4, 1))
        self.conv_single_to_bomb = (self.conv1, self.conv2, self.conv3, self.conv4)
        self.conv_straight = nn.Conv2d(state_shape[0], kernels, (1, 15), 1)
        self.pool = nn.MaxPool2d((4, 1))
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(kernels * (4 + 15), 512)
        self.fc2 = nn.Linear(512, action_num)

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

        # x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AveragePolicyNet(nn.Module):

    def __init__(self, state_shape, num_actions, use_conv=False):
        super(AveragePolicyNet, self).__init__()
        self.use_conv = use_conv

        flattened_state_shape = reduce(lambda x, y: x * y, state_shape)
        self.fc1 = nn.Linear(flattened_state_shape, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_actions)

    def forward(self, state):
        x = torch.flatten(state, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), -1)

        return x

