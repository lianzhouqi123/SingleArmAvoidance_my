from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from manipulator_mujoco.rl_program.TD3 import *
from manipulator_mujoco.GAN.GAN import GAN
import matplotlib.pyplot as plt
import csv


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # 超出最大容量则剔除另一端

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state = torch.stack(state)
        action = torch.stack(action).reshape([batch_size, 6])
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.stack(next_state)
        done = torch.tensor(done)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)


class GoalCollection:
    def __init__(self, dim_goal, distance_threshold=None):
        self.buffer = torch.tensor([], dtype=torch.float32)
        self.dim_goal = dim_goal
        self.distance_threshold = distance_threshold

    def add(self, new_goals):
        num_goals = new_goals.shape[0]
        if self.distance_threshold <= 0 or self.distance_threshold is None:  # 判断是否超距离限制
            for ii in range(num_goals):
                goal = new_goals[ii, :].reshape([1, -1])
                distance = torch.norm(self.buffer - goal, dim=1)
                min_distance = torch.min(distance)
                if min_distance > self.distance_threshold:  # 超过阈值的才存
                    self.buffer = torch.cat([self.buffer, goal], dim=0)
        else:
            self.buffer = torch.cat([self.buffer, new_goals], dim=0)

    def sample(self, batch_size):
        batch_index = random.sample(range(self.size), batch_size)
        sample_goals = self.buffer[batch_index, :]

        return sample_goals

    @property
    def size(self):
        return self.buffer.shape[0]


class Goal_Label_Collection:
    def __init__(self, dim_goal, distance_threshold=None, R_min=0.22, R_max=0.8):
        self.buffer = []
        self.dim_goal = dim_goal
        self.distance_threshold = distance_threshold
        self.R_min = R_min
        self.R_max = R_max

    def reset(self):
        self.buffer = []

    def add(self, goal, label_good):
        if not torch.is_tensor(goal):
            goal = torch.tensor(goal, dtype=torch.float32)
        goal = goal.reshape([1, -1])
        self.buffer.append((goal, label_good))

    def sample(self, batch_size):
        if batch_size > self.size:
            batch_size = self.size

        # 采样
        tran_idx = random.sample(range(self.size), batch_size)  # 取数组
        transitions = [self.buffer[i] for i in tran_idx]  # 取样
        goals, labels_good = zip(*transitions)  # 把tuple解压成list
        self.buffer = [trans for idx, trans in enumerate(self.buffer) if idx not in tran_idx]  # 删掉取出的数

        # 将tuple转成tensor
        goals = torch.stack(goals, dim=0).reshape([-1, self.dim_goal])
        labels_good = torch.tensor(labels_good, dtype=torch.float32).reshape([-1, 1])

        # 初始化保存数组
        goals_save = torch.tensor([]).reshape([-1, self.dim_goal])
        labels_save = torch.tensor([]).reshape([-1, 1])

        # 将相近的goals的labels取平均
        if self.distance_threshold is not None and self.distance_threshold > 0:
            goals_temp = goals
            labels_temp = labels_good
            while goals_temp.shape[0] > 0:
                # 取一个目标(取第一个)
                goal = goals_temp[0, :].reshape([1, self.dim_goal])
                label_of_goal = labels_temp[0, :].reshape([1, -1])
                # 计算距离，并取小于距离限制的数
                dis = torch.norm(goals_temp - goal, dim=1)  # [n, ]
                flag = torch.lt(dis, self.distance_threshold)  # [n, ]

                # 计算label
                n_state = torch.sum(flag)  # 总数
                if n_state >= 5:  # 数量够多才认为数据有效
                    label_good = (torch.sum(labels_temp * flag)/n_state).reshape([1, 1])  # 计算成功率
                    # 不大不小才是真的好
                    if self.R_min <= label_good <= self.R_max:
                        label = torch.tensor([[1]], dtype=torch.float32)
                    else:
                        label = torch.tensor([[0]], dtype=torch.float32)

                    # 存数据
                    goals_save = torch.cat([goals_save, goal], dim=0)
                    labels_save = torch.cat([labels_save, label], dim=0)

                    # 删数据
                    goals_temp = goals_temp[~flag]
                    labels_temp = labels_temp[~flag]

                else:  # 不符合要求，放回buffer
                    self.add(goal, label_of_goal)

                    # 删数据
                    goals_temp = goals_temp[1:]
                    labels_temp = labels_temp[1:]

        else:
            raise ValueError("Goal_Label_Collection，取样时，错误distance_threshold")

        return goals_save, labels_save

    @property
    def size(self):
        return len(self.buffer)


class GoalGAN:
    """ 将GAN应用在生成目标上"""

    def __init__(self, state_size, evaluator_size, state_noise_level, goal_low, goal_high, gen_n_hiddens,
                 gen_n_outputs, discr_n_hiddens, gen_lr, discr_lr, device):
        self.device = device
        self.gan = GAN(state_size,  gen_n_hiddens, gen_n_outputs, discr_n_hiddens,
                       evaluator_size, gen_lr, discr_lr, device)
        self.state_size = state_size
        self.evaluator_size = evaluator_size
        self.state_noise_level = state_noise_level
        self.goal_low = torch.tensor(goal_low, dtype=torch.float32).reshape([1, -1])
        self.goal_high = torch.tensor(goal_high, dtype=torch.float32).reshape([1, -1])
        self.goal_center = (self.goal_high + self.goal_low)/2

    # 预训练
    def pretrain(self, states, outer_iters):
        labels = torch.ones((states.shape[0], self.gan.discr_n_outputs))
        return self.gan.train(states, labels, outer_iters)

    def sample_states(self, size):
        normalized_goals, noise = self.gan.sample_generator(size)
        goals = self.goal_center + normalized_goals.cpu() * (self.goal_high - self.goal_center)
        return goals, noise

    def add_noise_to_states(self, goals):
        noise = torch.randn_like(goals)
        goals += noise
        return torch.clamp(goals, min=self.goal_low, max=self.goal_high)

    def sample_states_with_noise(self, size):
        goals, noise = self.sample_states(size)
        goals = self.add_noise_to_states(goals)
        return goals, noise

    def train(self, goals_input, labels_input, batch_size, outer_iters=1):
        return self.gan.train(goals_input, labels_input, batch_size, outer_iters)


def generate_initial_goals(env, agent, horizon=500, size=1e4):
    current_goal = env.get_current_goal()  # 要求是1维的
    goals_dim = current_goal.shape[0]
    goals = np.array([], dtype=np.float32).reshape(-1, goals_dim)  # 创建存goals的数组

    state = env.reset()[0].cuda()
    done = False
    flag_cont_epi = False
    steps = 0
    while goals.shape[0] < size:
        steps += 1
        if done or steps >= horizon:
            steps = 0
            done = False
            flag_cont_epi = False
            # 随机新生成一个目标，未完成
            env.new_target()
            state = env.reset()[0].cuda()
        else:
            action = agent.select_action(state)  # 使用策略
            next_state, _, done, terminate, info = env.step(action)
            if env.flag_cont or flag_cont_epi:  # 判断是否在此步之前发生碰撞
                flag_cont_epi = True
            state = next_state.cuda()
            # 记录目标是好目标还是坏目标
            if ~flag_cont_epi:  # 若没撞，则当前状态标为1
                label_good = 1
            else:
                label_good = 0
            if label_good:
                goal_from_state = env.state2goal().reshape([1, -1])  # 将当前状态变为目标
                if np.all((env.goal_low <= goal_from_state) & (goal_from_state <= env.goal_high)):
                    goals = np.append(goals, goal_from_state, axis=0)  # 存数

    return goals


def add_goal_from_state(env, label_state, collection):
    goal = env.state2goal()
    if np.all((env.goal_low <= goal) & (goal <= env.goal_high)):
        collection.add(goal, label_state)


def run_train(env, agent, gan, goals_buffer, replay_buffer, goal_label_buffer, num_episodes, minimal_size,
              batch_size_rl, batch_size_gan, num_iteration, num_new_goals, num_old_goals, num_rl, num_gan, save_file):
    return_list = []
    discri_list = []
    # discri_list = np.array([], dtype=np.float32)
    env.reset()
    pretrain_goals = generate_initial_goals(env, agent)  # 获取预训练的目标
    pretrain_goals = torch.tensor(pretrain_goals, dtype=torch.float32)
    goals_buffer.add(pretrain_goals)
    pretrain_goals = pretrain_goals.cuda()
    gan.pretrain(pretrain_goals, 500)  # 预训练

    i_terminate = 0
    i_episode_all = 0
    i_contact = 0
    n_section = 10
    for i in range(n_section):
        with tqdm(total=int(num_episodes / n_section), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / n_section)):
                raw_goals, _ = gan.sample_states_with_noise(num_new_goals)  # 生成新目标
                old_goals = goals_buffer.sample(num_old_goals)  # 从全部目标集取老目标
                goals_epi = torch.cat([raw_goals, old_goals], dim=0)
                goal_label_buffer.reset()  # 重置

                env.update_goals(goals_epi)  # 更新环境的可选目标
                discri = torch.mean(gan.gan.discriminator_predict(goals_epi.cuda()).cpu()).numpy()
                discri_list.append(discri)

                # 跑环境并用RL训练agent
                for jj in range(int(num_rl)):
                    goal = env.set_goal()
                    goal = torch.tensor(goal, dtype=torch.float32)
                    episode_return = 0
                    env.change_goal = False
                    state = env.reset()[0].cuda()
                    done = False
                    flag_cont_epi = False
                    iidx = 0
                    while not done:
                        iidx += 1
                        action = agent.select_action(state)  # 使用策略
                        next_state, reward, done, terminate, info = env.step(action)  # 使用动作走一步
                        if env.flag_cont or flag_cont_epi:  # 判断是否在此步之前发生碰撞
                            flag_cont_epi = True
                        add_goal_from_state(env, flag_cont_epi, goal_label_buffer)  # 只能在当前状态下调用!!!
                        replay_buffer.add(state, action, reward, next_state.cuda(), done)  # 加入RL的buffer
                        episode_return += reward
                        state = next_state.cuda()
                        # 训练RL
                        if replay_buffer.size() > minimal_size:
                            for j in range(num_iteration):
                                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size_rl)  # 采样
                                transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(),
                                                   'next_states': b_ns.cuda(),
                                                   'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                                agent.update_parameters(transition_dict, iidx)  # 训练agent

                    if terminate:
                        i_terminate += 1
                    if flag_cont_epi:
                        i_contact += 1
                    # 记录目标是好目标还是坏目标
                    if ~flag_cont_epi and terminate:  # 若没撞且到达，则当前状态标为1
                        label_good = 1
                    else:
                        label_good = 0
                    goal_label_buffer.add(goal, label_good)  # 存goal, label
                    return_list = np.append(return_list, episode_return)

                # 训练gan
                goals_with_label, labels_of_goals = goal_label_buffer.sample(goal_label_buffer.size)  # 全取
                for kk in range(int(num_gan)):
                    gan.train(goals_with_label, labels_of_goals, batch_size_gan)

                flag_goals = (labels_of_goals == 1).reshape([-1])
                goals_buffer.add(goals_with_label[flag_goals])

                if (i_episode + 1) % 1 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / n_section * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-1:]),
                                      'terminate': '%d' % i_terminate,
                                      'contact': '%d' % i_contact})
                pbar.update(1)
                i_episode_all += 1
        time = i
        torch.save(agent.actor.state_dict(), f'{save_file}/actor_{time}.pth')
        torch.save(agent.critic_1.state_dict(), f'{save_file}/critic_1_{time}.pth')
        torch.save(agent.critic_2.state_dict(), f'{save_file}/critic_2_{time}.pth')
        torch.save(gan.gan.gen.state_dict(), f'{save_file}/gen_{time}.pth')
        torch.save(gan.gan.discr.state_dict(), f'{save_file}/discr_{time}.pth')
        with open('{}/return_list_{}.csv'.format(save_file, time), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(return_list)
        with open('{}/discri_list_{}.csv'.format(save_file, time), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(discri_list)

    return return_list, discri_list


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
