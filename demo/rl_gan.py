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
    def __init__(self, dim_goal, distance_threshold=None, R_min=0.22, R_max=0.9):
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

    def sample(self, delete=False, pre=False, output_type="tensor"):
        # 采样
        goals, labels_good = zip(*self.buffer)  # 把tuple解压成list
        if delete:
            self.reset()

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
                    label_good = (torch.sum(labels_temp[flag]) / n_state).reshape([1, 1])  # 计算成功率
                    # 不大不小才是真的好
                    if self.R_min <= label_good and (label_good <= self.R_max or pre):
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
                    if delete:  # 没删就不用放了
                        self.add(goal, label_of_goal)

                    # 删数据
                    goals_temp = goals_temp[1:]
                    labels_temp = labels_temp[1:]

        else:
            raise ValueError("Goal_Label_Collection，取样时，错误distance_threshold")

        if output_type == "ndarray":
            goals_save = goals_save.numpy()
            labels_save = labels_save.numpy()

        return goals_save, labels_save

    # def sample_goals(self, batch_size=0, all_get=False):
    #     goals_all, labels_all = self.sample(self.size, delete=False)
    #     goals_good_label = np.array([trans for idx, trans in enumerate(goals_all) if labels_all[idx, :] == 1])
    #     if goals_good_label.shape[0] > 0:
    #         if all_get:
    #             batch_size = goals_good_label.shape[0]
    #         batch_index = random.sample(range(goals_good_label.shape[0]), batch_size)
    #         sample_goals = goals_good_label[batch_index, :]
    #     else:
    #         sample_goals = np.array([]).reshape([-1, self.dim_goal])
    #
    #     return sample_goals

    @property
    def size(self):
        return len(self.buffer)


class GoalGAN:
    """ 将GAN应用在生成目标上"""

    def __init__(self, state_size, evaluator_size, state_noise_level, goal_low, goal_high, gen_n_hiddens,
                 gen_n_outputs, discr_n_hiddens, gen_lr, discr_lr, device):
        self.device = device
        self.gan = GAN(state_size, gen_n_hiddens, gen_n_outputs, discr_n_hiddens,
                       evaluator_size, gen_lr, discr_lr, device)
        self.state_size = state_size
        self.evaluator_size = evaluator_size
        self.state_noise_level = state_noise_level
        self.goal_low = torch.tensor(goal_low, dtype=torch.float32).reshape([1, -1])
        self.goal_high = torch.tensor(goal_high, dtype=torch.float32).reshape([1, -1])
        self.goal_center = (self.goal_high + self.goal_low) / 2

    # 预训练
    def pretrain(self, states, outer_iters):
        labels = torch.ones((states.shape[0], self.gan.discr_n_outputs))
        return self.gan.train(states, labels, outer_iters)

    def sample_states(self, size):
        normalized_goals, noise = self.gan.sample_generator(size)
        goals = self.goal_center + normalized_goals.cpu() * (self.goal_high - self.goal_center)
        return goals, noise

    def add_noise_to_states(self, goals):
        noise = torch.randn_like(goals) * self.state_noise_level
        goals += noise
        return torch.clamp(goals, min=self.goal_low, max=self.goal_high)

    def sample_states_with_noise(self, size):
        goals, noise = self.sample_states(size)
        goals = self.add_noise_to_states(goals)
        return goals, noise

    def train(self, goals_input, labels_input, batch_size, outer_iters=1):
        goals_adj = (goals_input - self.goal_center)/(self.goal_high - self.goal_center)
        return self.gan.train(goals_adj, labels_input, batch_size, outer_iters)


def generate_initial_goals(env, agent, GL_buffer, horizon=500, size=0.3e3):
    current_goal = env.get_current_goal().reshape([-1])  # 要求是1维的
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
            # 随机新生成一个目标
            env.set_goal(mode="fixed_sample_mode")
            state = env.reset()[0].cuda()
            goals_sample, labels_sample = GL_buffer.sample(delete=True, pre=True, output_type="ndarray")
            if goals_sample.shape[0] > 0:
                goals_good_label = np.array([trans for idx, trans in enumerate(goals_sample)
                                             if labels_sample[idx, 0] == 1])
                if goals_good_label.shape[0] > 0:
                    goals = np.append(goals, goals_good_label, axis=0)
            print(goals.shape[0], GL_buffer.size)
        else:
            action = agent.select_action(state)  # 使用策略
            next_state, _, done, terminate, info = env.step(action)
            if env.flag_cont or flag_cont_epi:  # 判断是否在此步之前发生碰撞
                flag_cont_epi = True
            add_goal_from_state(env, not flag_cont_epi, GL_buffer)  # 只能在当前状态下调用!!!
            state = next_state.cuda()

    return goals


def add_goal_from_state(env, label_state, collection):
    goal = env.state2goal()
    if np.all((env.goal_low <= goal) & (goal <= env.goal_high)):
        collection.add(goal, label_state)


def run_train(env, agent, gan, goals_buffer, replay_buffer, goal_label_buffer, num_episodes, minimal_size,
              batch_size_rl, batch_size_gan, num_iteration, num_new_goals, num_old_goals, num_rl, num_gan, save_file):
    return_list = []
    discri_list = []
    dis_loss_save = []
    gen_loss_save = []
    actor_loss_save = []
    actor_loss = 0
    env.set_goal(mode="fixed_sample_mode")
    env.reset()
    pretrain_goals = generate_initial_goals(env, agent, goal_label_buffer)  # 获取预训练的目标
    pretrain_goals = torch.tensor(pretrain_goals, dtype=torch.float32)
    goals_buffer.add(pretrain_goals)
    pretrain_goals = pretrain_goals.cuda()
    gan.pretrain(pretrain_goals, 300)  # 预训练

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

                env.update_goals(goals_epi)  # 更新环境的可选目标
                # env.update_goals(old_goals)  # 更新环境的可选目标
                discri = torch.mean(gan.gan.discriminator_predict(goals_epi.cuda()).cpu()).numpy()
                discri_list = np.append(discri_list, discri)

                # 跑环境并用RL训练agent
                for jj in range(int(num_rl)):
                    goal = env.set_goal(mode="point_set")
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
                        add_goal_from_state(env, not flag_cont_epi, goal_label_buffer)  # 只能在当前状态下调用!!!
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
                                actor_loss = agent.update_parameters(transition_dict, iidx)  # 训练agent
                                # actor_loss_save.append(actor_loss)

                    if terminate:
                        i_terminate += 1
                    if flag_cont_epi:
                        i_contact += 1

                    return_list = np.append(return_list, episode_return)

                # 训练gan
                goals_with_label, labels_of_goals = goal_label_buffer.sample(delete=True)  # 全取
                # n_goals_with_label = min(goals_with_label.shape[0], goals_buffer.size)
                # goals_with_label1 = goals_buffer.sample(n_goals_with_label)
                # labels_of_goals1 = torch.ones([n_goals_with_label, 1])
                # goals_with_label_train = torch.cat([goals_with_label, goals_with_label1], dim=0)
                # labels_of_goals_train = torch.cat([labels_of_goals, labels_of_goals1], dim=0)
                for kk in range(int(num_gan)):
                    dis_loss, gen_loss = gan.train(goals_with_label, labels_of_goals, batch_size_gan)
                    # dis_loss, gen_loss = gan.train(goals_with_label_train, labels_of_goals_train, batch_size_gan)
                    # dis_loss_save.append(dis_loss)
                    # gen_loss_save.append(gen_loss)

                flag_goals = (labels_of_goals == 1).reshape([-1])
                goals_buffer.add(goals_with_label[flag_goals])

                if (i_episode + 1) % 1 == 0:
                    pbar.set_postfix({'epi': '%d' % (num_episodes / n_section * i + i_episode + 1),
                                      'rtn': '%.3f' % np.mean(return_list[-1:]),
                                      'term': '%d' % i_terminate,
                                      'cnt': '%d' % i_contact,
                                      'dscr': '%.7f' % discri,
                                      # 'actor_loss': '%.6f' % actor_loss,
                                      # 'dis_loss': '%.6f' % dis_loss
                                      })
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
        with open('{}/actor_list_{}.csv'.format(save_file, time), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(actor_loss_save)
        with open('{}/dis_list_{}.csv'.format(save_file, time), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(dis_loss_save)
        with open('{}/gen_list_{}.csv'.format(save_file, time), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(gen_loss_save)

    return return_list, discri_list, actor_loss, dis_loss_save, gen_loss_save


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
