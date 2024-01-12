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



class GoalGAN:
    """ 将GAN应用在生成目标上"""
    def __init__(self, state_size, evaluater_size, state_noise_level, *args, **kwargs):
        self.gan = GAN(gen_n_outputs=state_size, discr_n_outputs=evaluater_size, *args, **kwargs)
        self.state_size = state_size
        self.evaluater_size = evaluater_size
        self.state_noise_level = state_noise_level

    # 预训练
    def pretrain(self, states, outer_iters):
        labels = torch.ones((states.shape[0], self.gan.discr_n_outputs))
        return self.gan.train(states, labels, outer_iters)

    def add_noise_to_states(self, states):
        noise = np.random.randn




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


def run_train(env, agent, gan):
    pretrain_goals = generate_initial_goals(env, agent)
    pretrain(gan, pretrain_goals, 500)


