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
    def __init__(self, dim_goal, distance_threshold=None):
        self.buffer = []
        self.dim_goal = dim_goal
        self.distance_threshold = distance_threshold

    def reset(self):
        self.buffer = []

    def add(self, goal, label):
        self.buffer.append((goal, label))

    def sample(self, batch_size):
        if batch_size > self.size:
            batch_size = self.size
        transitions = random.sample(self.buffer, batch_size)  # 取样
        goals, labels = zip(*transitions)  # 把tuple解压成list
        # 将tuple转成tensor
        goals = torch.stack(goals, dim=0).reshape([-1, self.dim_goal])
        labels = torch.tensor(labels, dtype=torch.float32).reshape([-1, 1])
        goals_save = torch.tensor([]).reshape([-1, self.dim_goal])
        labels_save = torch.tensor([]).reshape([-1, 1])
        # 将相近的goals的labels取平均
        if self.distance_threshold is not None and self.distance_threshold > 0:
            n_goals = goals.shape[0]  # goals的数目
            flag_mat = torch.ones([n_goals, 1])
            goals_temp = goals
            labels_temp = labels
            while goals_temp.shape[0] > 0:
                # 取一个目标
                goal = goals_temp[0, :].reshape([1, self.dim_goal])
                # 计算距离，并取小于距离限制的数
                dis = torch.norm(goals_temp - goal, dim=1)  # [n, ]
                flag = torch.lt(dis, self.distance_threshold)  # [n, ]
                # 计算label
                n_state = torch.sum(flag)
                label = torch.tensor(torch.sum(labels_temp * flag)/n_state).reshape([1, 1])
                # 存数据
                goals_save = torch.cat([goals_save, goal], dim=0)
                labels_save = torch.cat([labels_save, label], dim=0)
                # 删数据





        return goals, labels

    @property
    def size(self):
        return len(self.buffer)


class GoalGAN:
    """ 将GAN应用在生成目标上"""

    def __init__(self, state_size, evaluator_size, state_noise_level, goal_low, goal_up, *args, **kwargs):
        self.gan = GAN(gen_n_outputs=state_size, discr_n_outputs=evaluator_size, *args, **kwargs)
        self.state_size = state_size
        self.evaluator_size = evaluator_size
        self.state_noise_level = state_noise_level
        self.goal_low = goal_low.reshape([1, 4])
        self.goal_up = goal_up.reshape([1, 4])
        self.goal_center = (self.goal_up + self.goal_low)/2

    # 预训练
    def pretrain(self, states, outer_iters):
        labels = torch.ones((states.shape[0], self.gan.discr_n_outputs))
        return self.gan.train(states, labels, outer_iters)

    def sample_states(self, size):
        normalized_goals, noise = self.gan.sample_generator(size)
        goals = self.goal_center + normalized_goals * (self.goal_up - self.goal_center)
        return goals, noise

    def add_noise_to_states(self, goals):
        noise = torch.randn_like(goals)
        goals += noise
        return torch.clamp(goals, min=self.goal_low, max=self.goal_up)

    def sample_states_with_noise(self, size):
        goals, noise = self.sample_states(size)
        goals = self.add_noise_to_states(goals)
        return goals, noise


def generate_initial_goals(env, agent, horizon=500, size=1e4):
    current_goal = env.get_current_goal()  # 要求是1维的
    goals_dim = current_goal.shape[0]
    goals = np.array([], dtype=np.float32).reshape(-1, goals_dim)  # 创建存goals的数组

    state = env.reset()[0].cuda()
    done = False
    steps = 0
    while goals.shape[0] < size:
        steps += 1
        if done or steps >= horizon:
            steps = 0
            done = False
            # 随机新生成一个目标，未完成
            env.target_sample()

            state = env.reset()[0].cuda()
        else:
            action = agent.select_action(state)  # 使用策略
            next_state, _, done, terminate, info = env.step(action)
            state = next_state.cuda()
            if info["good_state"]:
                goal_from_state = env.state2goal(state)  # 将当前状态变为目标
                goals = np.append(goals, goal_from_state, axis=0)  # 存数

    return goals


def run_train(env, agent, gan, goals_buffer, replay_buffer, goal_label_buffer, num_episodes, minimal_size, batch_size,
              num_iteration, num_new_goals, num_old_goals, num_rl, num_gan, save_file):
    pretrain_goals = generate_initial_goals(env, agent)  # 获取预训练的目标
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
                goal_label_buffer.reset()  # 重置

                env.update_goals(raw_goals, old_goals, mode="uniform")  # 更新环境的可选目标

                # 跑环境并用RL训练agent
                for jj in range(int(num_rl)):
                    episode_return = 0
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
                        replay_buffer.add(state, action, reward, next_state.cuda(), done)  # 加入RL的buffer
                        if ~flag_cont_epi:  # 若没撞，则当前状态标为1
                            label = 1
                        else:
                            label = 0
                        goal_label_buffer.add(state, label)  # 存goal, label
                        episode_return += reward
                        state = next_state.cuda()
                        if replay_buffer.size() > minimal_size:
                            for j in range(num_iteration):
                                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)  # 采样
                                transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(),
                                                   'next_states': b_ns.cuda(),
                                                   'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                                agent.update_parameters(transition_dict, iidx)  # 训练agent

                    if terminate:
                        i_terminate += 1
                    if flag_cont_epi:
                        i_contact += 1
                    return_list = np.append(return_list, episode_return)

                # 训练gan
                for kk in range(int(num_gan)):
                    goals_labels = goal_label_buffer.sample(goal_label_buffer.size)  # 全取

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
        with open('{}/return_list_{}.csv'.format(save_file, time), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(return_list)

    return return_list
