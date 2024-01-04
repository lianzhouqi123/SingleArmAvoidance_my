import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Q_net(nn.Module):
    def __init__(self, n_action, n_feature):
        self.n_feature = n_feature
        self.n_action = n_action
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(self.n_feature, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.act1 = nn.LeakyReLU(0.01)
        self.lin2 = nn.Linear(120, 80)
        self.bn2 = nn.BatchNorm1d(80)
        self.act2 = nn.LeakyReLU(0.01)
        self.lin3 = nn.Linear(80, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.act3 = nn.LeakyReLU(0.01)
        self.lin4 = nn.Linear(16, self.n_action)

        nn.init.kaiming_uniform_(self.lin1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.lin2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.lin3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.lin4.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        if x.shape[0] > 1:
            x = self.act1(self.bn1(self.lin1(x)))
            x = self.act2(self.bn2(self.lin2(x)))
            x = self.act3(self.bn3(self.lin3(x)))
            x = self.lin4(x)
        else:
            x = self.act1(self.lin1(x))
            x = self.act2(self.lin2(x))
            x = self.act3(self.lin3(x))
            x = self.lin4(x)

        return x


class DQN:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,
                 tau=0.01, device="cpu"
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.tau = tau
        self.device = device

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory_state = torch.zeros([self.memory_size, self.n_features], dtype=torch.float32).cuda()
        self.memory_state_next = torch.zeros([self.memory_size, self.n_features], dtype=torch.float32).cuda()
        self.memory_action = torch.zeros([self.memory_size, 1], dtype=torch.int32).cuda()
        self.memory_reward = torch.zeros([self.memory_size, 1], dtype=torch.float32).cuda()
        self.memory_next_act_avail = torch.zeros([self.memory_size, self.n_actions], dtype=torch.int32).cuda()

        self.loss_func = nn.MSELoss()  # def MSELoss(pred,target):	return (pred-target)**2
        self.cost_his = np.array([])

        self._build_net()

    def _build_net(self):
        self.q_eval = Q_net(self.n_actions, self.n_features).to(self.device)
        self.q_target = Q_net(self.n_actions, self.n_features).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

    def net2cuda(self):
        if next(self.q_eval.parameters()).device != torch.device("cuda:0"):
            self.q_eval.cuda()
        if next(self.q_target.parameters()).device != torch.device("cuda:0"):
            self.q_target.cuda()

    def net2cpu(self):
        if next(self.q_eval.parameters()).device == torch.device("cuda:0"):
            self.q_eval.cpu()
        if next(self.q_target.parameters()).device == torch.device("cuda:0"):
            self.q_target.cpu()

    def choose_action(self, state, random=True):  # epsilon用于测试用
        state = torch.Tensor(state[np.newaxis, :]).cuda()  # 在原矩阵第一个维度前加一个维度
        if np.random.uniform() < self.epsilon or not random:
            actions_value = self.q_eval(state).data.cpu().numpy().reshape([self.n_actions])
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)#.view(-1, 1)
        actions = actions.view(1, -1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_next, q_eval = self.q_target(next_states), self.q_eval(states)

        q_target = torch.clone(q_eval.data)

        # q_next = torch.reshape(torch.min(q_next, 1).values, [-1, 1]) * torch.ones(q_next.shape).cuda() \
        #          * (1 - batch_memory_next_act_avail) + q_next * batch_memory_next_act_avail

        batch_index = np.arange(self.batch_size, dtype=np.int32).reshape([-1, 1])
        eval_act_index = actions.type(torch.int).cpu().numpy()  # 每组数据的action
        q_target[batch_index, eval_act_index.reshape([-1, 1])] = rewards + self.gamma \
            * torch.reshape(torch.max(q_next, 1).values, [self.batch_size, 1])  # Q-learning的典型算法

        # 标准步骤
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()  # 清空过往梯度
        loss.backward()  # 反向传播，计算当前梯度
        self.optimizer.step()  # 根据梯度更新网络参数

        # increase epsilon
        self.cost_his = np.append(self.cost_his, loss.cpu().detach().numpy())  # 记录损失
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)


    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
