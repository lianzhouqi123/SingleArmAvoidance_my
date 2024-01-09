from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from manipulator_mujoco.rl_program.TD3 import *
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


class ReplayBuffer_2con:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, contact):
        self.buffer.append((state, action, reward, next_state, done))  # 超出最大容量则剔除另一端
        if contact:
            self.buffer.append((state, action, reward, next_state, done))  # 碰撞则多往容器里放一组

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

def train(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = np.array([])
    step_i = 0
    i_episode_all = 0
    i_terminate = 0
    i_contact = 0
    for i in range(10):  # 循次数考虑适当增加
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 10000/10 step
                episode_return = 0
                state = env.reset()[0].cuda()
                done = False
                terminate = False
                flag_cont_eqi = False
                iidx = 0
                i_episode_all += 1
                # 更新OU噪声
                # env.unwrapped.ounoise.change_scale(env.unwrapped.ounoise_noise_scale,
                #                                    env.unwrapped.ounoise_final_noise_scale,
                #                                    env.unwrapped.exploration_end, i_episode_all)
                while not done:
                    step_i += 1
                    iidx += 1
                    # action = agent.select_action(state, env.unwrapped.ounoise, flag_device=True)  # 施加OU噪声 NAF
                    action = agent.select_action(state)
                    next_state, reward, done, terminate, flag_cont = env.step(action)
                    replay_buffer.add(state, action, reward, next_state.cuda(), done)
                    state = next_state.cuda()
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(), 'next_states': b_ns.cuda(),
                                           'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                        agent.update_parameters(transition_dict, iidx)
                        # print("lin1", agent.model.linear1.weight.grad)
                        # print("lin2", agent.model.linear2.weight.grad)
                        # print("mu", agent.model.mu.weight.grad)
                    if env.flag_cont or flag_cont_eqi:
                        flag_cont_eqi = True
                if terminate:
                    i_terminate += 1
                if flag_cont_eqi:
                    i_contact += 1
                return_list = np.append(return_list, episode_return)
                if (i_episode + 1) % 1 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-1:]),
                                      'terminate': '%d' % i_terminate,
                                      'contact': '%d' % i_contact})
                pbar.update(1)
    return return_list


def train2(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, num_iteration, save_file):
    return_list = np.array([])
    i_terminate = 0
    i_episode_all = 0
    i_contact = 0
    step_i = 0
    n_section = 10
    for i in range(n_section):
        with tqdm(total=int(num_episodes / n_section), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / n_section)):  # 10000/10 step
                episode_return = 0
                state = env.reset()[0].cuda()
                done = False
                terminate = False
                flag_cont_epi = False
                iidx = 0
                while not done:
                    step_i += 1
                    iidx += 1
                    action = agent.select_action(state)  # 使用新策略
                    next_state, reward, done, terminate, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state.cuda(), done)
                    state = next_state.cuda()
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        for j in range(num_iteration):
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(), 'next_states': b_ns.cuda(),
                                               'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                            agent.update_parameters(transition_dict, iidx)
                        # print("lin1", agent.model.linear1.weight.grad)
                        # print("lin2", agent.model.linear2.weight.grad)
                        # print("mu", agent.model.mu.weight.grad)
                    if env.flag_cont or flag_cont_epi:
                        flag_cont_epi = True
                if terminate:
                    i_terminate += 1
                if flag_cont_epi:
                    i_contact += 1
                return_list = np.append(return_list, episode_return)
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


def train_gan(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, num_iteration, save_file):
    return_list = np.array([])
    i_terminate = 0
    i_episode_all = 0
    i_contact = 0
    step_i = 0
    n_section = 10
    for i in range(n_section):
        with tqdm(total=int(num_episodes / n_section), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / n_section)):  # 10000/10 step

                goals = gan.sample_generator()
                env.set_goals(goals)


                episode_return = 0
                state = env.reset()[0].cuda()
                done = False
                terminate = False
                flag_cont_epi = False
                iidx = 0
                while not done:
                    step_i += 1
                    iidx += 1
                    action = agent.select_action(state)  # 使用新策略
                    next_state, reward, done, terminate, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state.cuda(), done)
                    state = next_state.cuda()
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        for j in range(num_iteration):
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(), 'next_states': b_ns.cuda(),
                                               'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                            agent.update_parameters(transition_dict, iidx)
                        flag_cont_epi = True
                if terminate:
                    i_terminate += 1
                if flag_cont_epi:
                    i_contact += 1
                return_list = np.append(return_list, episode_return)

                labels = gan.labels_goals(terminate)
                gan.train(goals, labels)
                replay_buffer_gan.add(goals)


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


def train2_con(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, num_iteration, save_file):
    return_list = np.array([])
    i_terminate = 0
    i_episode_all = 0
    i_contact = 0
    step_i = 0
    n_section = 10
    for i in range(n_section):
        with tqdm(total=int(num_episodes / n_section), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / n_section)):  # 10000/10 step
                episode_return = 0
                state = env.reset()[0].cuda()
                done = False
                terminate = False
                flag_cont_epi = False
                iidx = 0
                while not done:
                    step_i += 1
                    iidx += 1
                    action = agent.select_action(state)  # 使用新策略
                    next_state, reward, done, terminate, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state.cuda(), done, env.flag_cont)
                    state = next_state.cuda()
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        for j in range(num_iteration):
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(), 'next_states': b_ns.cuda(),
                                               'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                            agent.update_parameters(transition_dict, iidx)
                        # print("lin1", agent.model.linear1.weight.grad)
                        # print("lin2", agent.model.linear2.weight.grad)
                        # print("mu", agent.model.mu.weight.grad)
                    if env.flag_cont or flag_cont_epi:
                        flag_cont_epi = True
                if terminate:
                    i_terminate += 1
                if flag_cont_epi:
                    i_contact += 1
                return_list = np.append(return_list, episode_return)
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


def train_tran(env, agent, agent_expert, num_episodes, replay_buffer, minimal_size, batch_size,
               epsino_max, epsino_min, epsino_descend_ratio, save_file):
    return_list = np.array([])
    i_terminate = 0
    i_episode_all = 0
    i_contact = 0
    step_i = 0
    n_section = 50
    for i in range(n_section):
        with tqdm(total=int(num_episodes / n_section), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / n_section)):  # 10000/10 step
                episode_return = 0
                state = env.reset()[0].cuda()
                done = False
                terminate = False
                flag_cont_epi = False
                iidx = 0
                # 专家探索随机阈值
                epsino = np.max([epsino_min, epsino_max - epsino_descend_ratio * i_episode_all])
                while not done:
                    step_i += 1
                    iidx += 1
                    if np.random.rand() < epsino:
                        action = agent_expert.select_action(state)  # 使用专家策略
                    else:
                        action = agent.select_action(state)  # 使用新策略
                    next_state, reward, done, terminate, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state.cuda(), done)
                    state = next_state.cuda()
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(), 'next_states': b_ns.cuda(),
                                           'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                        agent.update_parameters(transition_dict, iidx)
                        # print("lin1", agent.model.linear1.weight.grad)
                        # print("lin2", agent.model.linear2.weight.grad)
                        # print("mu", agent.model.mu.weight.grad)
                    if env.flag_cont or flag_cont_epi:
                        flag_cont_epi = True
                if terminate:
                    i_terminate += 1
                if flag_cont_epi:
                    i_contact += 1
                return_list = np.append(return_list, episode_return)
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


def train_tran_2exp(env, agent, agent_expert1, agent_expert2, num_episodes, replay_buffer, minimal_size, batch_size,
               epsino_max, epsino_min, epsino_descend_ratio, save_file):
    return_list = np.array([])
    i_terminate = 0
    i_episode_all = 0
    i_contact = 0
    step_i = 0
    n_section = 50
    for i in range(n_section):
        with tqdm(total=int(num_episodes / n_section), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / n_section)):  # 10000/10 step
                episode_return = 0
                state = env.reset()[0].cuda()
                done = False
                terminate = False
                flag_cont_epi = False
                iidx = 0
                # 专家探索随机阈值
                epsino_flag = np.max([epsino_min, epsino_max - epsino_descend_ratio * i_episode_all])
                while not done:
                    step_i += 1
                    iidx += 1
                    # 判断策略选择
                    epsino1 = np.random.rand()
                    epsino2 = np.random.rand()
                    if np.min([epsino1, epsino2]) < epsino_flag:
                        if epsino1 <= epsino2:
                            action = agent_expert1.select_action(state)  # 使用专家1策略
                        else:
                            action = agent_expert2.select_action(state)  # 使用专家2策略
                    else:
                        action = agent.select_action(state)  # 使用新策略
                    next_state, reward, done, terminate, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state.cuda(), done)
                    state = next_state.cuda()
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(), 'next_states': b_ns.cuda(),
                                           'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                        agent.update_parameters(transition_dict, iidx)
                        # print("lin1", agent.model.linear1.weight.grad)
                        # print("lin2", agent.model.linear2.weight.grad)
                        # print("mu", agent.model.mu.weight.grad)
                    if env.flag_cont or flag_cont_epi:
                        flag_cont_epi = True
                if terminate:
                    i_terminate += 1
                if flag_cont_epi:
                    i_contact += 1
                return_list = np.append(return_list, episode_return)
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


def train_KTM(env, agent, env_expert, agent_expert, num_episodes, replay_buffer_offline, replay_buffer,
              minimal_size, batch_size, offline_episodes, beta1, beta2):
    actor_expert = agent_expert.actor
    critic_1_expert = agent_expert.critic_1
    critic_2_expert = agent_expert.critic_2
    return_list = np.array([])
    step1_i = 0
    # 重塑经验
    with tqdm(total=int(offline_episodes)) as pbar:
        while 1:
            state = env_expert.reset()[0].cuda()
            done = False
            while not done:
                action = agent_expert.select_action(state)
                next_state, reward, done, terminate, info = env_expert.step(action)
                replay_buffer_offline.add(state, action, reward, next_state.cuda(), done)
                state = next_state.cuda()
                step1_i += 1
            if step1_i > replay_buffer_offline.buffer.maxlen:
                break
            pbar.set_postfix({'重塑经验中': '_=',
                              'step': '%d' % step1_i})
            pbar.update(1)
    # 预训练
    with tqdm(total=int(offline_episodes)) as pbar:
        for ii in range(int(offline_episodes)):
            b_s, b_a, b_r, b_ns, b_d = replay_buffer_offline.sample(batch_size)
            transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(), 'next_states': b_ns.cuda(),
                               'rewards': b_r.cuda(), 'dones': b_d.cuda()}
            agent.update_parameters_offline(transition_dict, actor_expert, critic_1_expert, critic_2_expert)
            pbar.set_postfix({'预训练中': '_=',
                              'step': '%d' % ii})
            pbar.update(1)
    agent.hard_update()
    # 正式训练
    i_terminate = 0
    i_contact = 0
    step_i = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 10000/10 step
                episode_return = 0
                state = env.reset()[0].cuda()
                done = False
                flag_cont_eqi = False
                iidx = 0
                while not done:
                    step_i += 1
                    iidx += 1
                    action = agent.select_action(state)
                    next_state, reward, done, terminate, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state.cuda(), done)
                    state = next_state.cuda()
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s.cuda(), 'actions': b_a.cuda(), 'next_states': b_ns.cuda(),
                                           'rewards': b_r.cuda(), 'dones': b_d.cuda()}
                        agent.update_parameters(transition_dict, iidx, actor_expert, beta1, beta2)
                        # print("lin1", agent.model.linear1.weight.grad)
                        # print("lin2", agent.model.linear2.weight.grad)
                        # print("mu", agent.model.mu.weight.grad)
                    if env.flag_cont or flag_cont_eqi:
                        flag_cont_eqi = True
                if terminate:
                    i_terminate += 1
                if flag_cont_eqi:
                    i_contact += 1
                return_list = np.append(return_list, episode_return)
                if (i_episode + 1) % 1 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-1:]),
                                      'terminate': '%d' % i_terminate,
                                      'contact': '%d' % i_contact})
                pbar.update(1)
                # time = i
                # torch.save(agent.actor.state_dict(), f'actor_{time}.pth')
                # torch.save(agent.critic_1.state_dict(), f'critic_1_{time}.pth')
                # torch.save(agent.critic_2.state_dict(), f'critic_2_{time}.pth')
    return return_list


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
