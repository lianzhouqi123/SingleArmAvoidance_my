import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.action_bound
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim+20)
        self.fc2 = nn.Linear(hidden_dim+20, hidden_dim+20)
        self.fc3 = nn.Linear(hidden_dim+20, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class KTMTD3:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 policy_noise, noise_clip, exploration_noise, gamma, policy_delay, tau, actor_lr,
                 critic_lr,
                 device):

        #  define the net (6 nets needed)
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, hidden_dim, action_dim).to(device)

        # make the start parameter of "target q net" same as "q net"
        self.hard_update()

        # optimizer options
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.action_bound = action_bound
        self.policy_delay = policy_delay
        self.tau = tau

    def load_net_para(self, actor_path=None, critic_1_path=None, critic_2_path=None):
        if actor_path:
            actor_load = torch.load(actor_path)
            self.actor.load_state_dict(actor_load)
        if critic_1_path:
            critic_1_load = torch.load(critic_1_path)
            self.critic_1.load_state_dict(critic_1_load)
        if critic_2_path:
            critic_2_load = torch.load(critic_2_path)
            self.critic_2.load_state_dict(critic_2_load)

    def select_action(self, state):
        state = state.unsqueeze(0)
        self.actor.eval()
        action = self.actor(state)
        self.actor.train()
        action = action + self.exploration_noise*torch.randn([1, 6]).cuda()
        action = action.clip(-self.action_bound, self.action_bound)
        return action

    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update_parameters(self, transition_dict, iidx, actor_expert, beta1, beta2):
        states = transition_dict['states'].clone().detach()
        actions = transition_dict['actions'].clone().detach()
        rewards = transition_dict['rewards'].view(-1, 1).clone().detach()
        next_states = transition_dict['next_states'].clone().detach()
        dones = transition_dict['dones'].view(-1, 1).to(torch.float32).clone().detach()

        # select the next action with noise
        noise = torch.ones_like(actions).data.normal_(0, self.policy_noise).to(self.device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        self.actor_target.eval()
        next_actions = (self.actor_target(next_states) + noise)
        self.actor_target.train()
        next_actions = next_actions.clamp(-self.action_bound, self.action_bound)

        # compute target q
        self.critic_1_target.eval()
        self.critic_2_target.eval()
        target_q1 = self.critic_1_target(next_states, next_actions)
        target_q2 = self.critic_2_target(next_states, next_actions)
        self.critic_1_target.train()
        self.critic_2_target.train()
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + ((1 - dones) * self.gamma * target_q).detach()

        # Optimize Critic 1 net
        self.critic_1.eval()
        current_q1 = self.critic_1(states, actions)
        self.critic_1.train()
        loss_q1 = F.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2 net
        self.critic_2.eval()
        current_q2 = self.critic_2(states, actions)
        self.critic_2.train()
        loss_q2 = F.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_2_optimizer.step()

        if iidx % self.policy_delay == 0:
            # relay update
            # compute actor loss
            actor_loss1 = - self.critic_1(states, self.actor(states)).mean()
            actor_loss2 = - self.critic_1(states, actor_expert(states)).mean()
            actor_loss = beta1 * actor_loss1 + beta2 * actor_loss2

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_1_target)
            self.soft_update(self.critic_2, self.critic_2_target)

    def update_parameters_offline(self, transition_dict, actor_expert, critic_1_expert, critic_2_expert):
        states = transition_dict['states'].clone().detach()
        actions = transition_dict['actions'].clone().detach()
        rewards = transition_dict['rewards'].view(-1, 1).clone().detach()
        next_states = transition_dict['next_states'].clone().detach()
        dones = transition_dict['dones'].view(-1, 1).to(torch.float32).clone().detach()

        # compute expert q
        expert_q1 = critic_1_expert(states, actions)
        expert_q2 = critic_2_expert(states, actions)

        # Optimize Critic 1 net
        current_q1 = self.critic_1(states, actions)
        loss_q1 = F.mse_loss(current_q1, expert_q1)
        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2 net
        current_q2 = self.critic_2(states, actions)
        loss_q2 = F.mse_loss(current_q2, expert_q2)
        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_2_optimizer.step()

        # compute actor loss
        actor_loss = - self.critic_1(states, actor_expert(states)).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)