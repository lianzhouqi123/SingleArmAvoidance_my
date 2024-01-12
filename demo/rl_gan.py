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


# 预训练
def pretrain(gan, states, outer_iters):
    labels = torch.ones((states.shape[0], gan.discr_n_outputs))
    return gan.train(states, labels, outer_iters)


def pretrain_uniform(gan, size=1000, report)


def generate_initial_goals(env, agent, method="uniform"):
    if method == "smart_init":
        pass
    else:
        pretrain_uniform(gan)


def run_train():
    pass

