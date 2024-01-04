import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import os


def MSELoss(input, target):
    return torch.sum((input - target) ** 2) / input.data.nelement()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Policy(nn.Module):

    def __init__(self, hidden_size, num_inputs, num_actions):
        super(Policy, self).__init__()
        self.num_actions = num_actions

        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.mu = nn.Linear(hidden_size, self.num_actions)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

        self.L = nn.Linear(hidden_size, self.num_actions ** 2)
        self.L.weight.data.mul_(0.1)
        self.L.bias.data.mul_(0.1)

        self.tril_mask = Variable(torch.tril(torch.ones(
            self.num_actions, self.num_actions), diagonal=-1).unsqueeze(0)).cuda()  # 下三角矩阵
        # [0 ...0;
        # 1 0...0;
        # 1 1 0...0;
        # ...;
        # 1...1 0]
        # unsqueeze : 在axis=0处插入一个维度
        # Variable ： 自动微分
        self.diag_mask = Variable(torch.diag(torch.diag(
            torch.ones(self.num_actions, self.num_actions))).unsqueeze(0)).cuda()
        # 和eye(self.num_acitons)一样

    def forward(self, inputs):
        x, u = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        V = self.V(x)
        mu = F.tanh(self.mu(x))

        Q = None
        if u is not None:
            num_outputs = mu.size(1)
            L = self.L(x).view(-1, num_outputs, num_outputs)
            L = L * \
                self.tril_mask.expand_as(
                    L) + torch.exp(L) * self.diag_mask.expand_as(L)
            # 把tril_mask扩展成L的形状
            P = torch.bmm(L, L.transpose(2, 1))  # 矩阵乘法 (batch_size, n, m) 和 (batch_size, m, p)
            # transpose : 将axis=2和axis=1转置

            u_mu = (u - mu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]

            Q = A + V

        return mu, Q, V


class NAF:

    def __init__(self, gamma, tau, yita, hidden_size, num_inputs, action_space, device):
        self.action_space = action_space
        self.num_inputs = num_inputs

        self.model = Policy(hidden_size, num_inputs, action_space)
        self.target_model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = Adam(self.model.parameters(), lr=yita)

        self.gamma = gamma
        self.tau = tau

        self.device = device

        hard_update(self.target_model, self.model)

    def model_to_device(self):
        self.model.to(self.device)
        self.target_model.to(self.device)

    def select_action(self, state, action_noise=None, flag_device=False, param_noise=None):
        state = state.unsqueeze(0)
        self.model.eval()
        mu, _, _ = self.model((Variable(state), None))
        self.model.train()
        mu = mu.data
        if action_noise is not None:
            if flag_device:
                mu.to(self.device)
                mu += torch.Tensor(action_noise.noise()).to(self.device)
            else:
                mu.cpu()
                mu += torch.Tensor(action_noise.noise())

        return mu

    def update_parameters(self, transition_dict):
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards'].view(-1, 1)
        next_states = transition_dict['next_states']
        dones = transition_dict['dones'].view(-1, 1)

        _, _, next_state_values = self.target_model((next_states, None))  # V

        # rewards = rewards.unsqueeze(1)
        # dones = dones.unsqueeze(1)
        expected_state_action_values = rewards + self.gamma * next_state_values

        _, state_action_values, _ = self.model((states, actions))  # Q

        loss = MSELoss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        # return loss.item(), 0

    def save_model(self, env_name, suffix="", model_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if model_path is None:
            model_path = "models/naf_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(model_path))
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        print('Loading model from {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))
