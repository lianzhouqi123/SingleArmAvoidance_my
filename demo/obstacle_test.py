import gymnasium as gym
import torch
import rl_utils_ywz as rl
import manipulator_mujoco.rl_program.DQN as DQN
import matplotlib.pyplot as plt
import numpy as np
import math as m
import time

env = gym.make('manipulator_mujoco/rm63Env-v0', render_mode='human')
timestep = np.arange(0, 3, 0.01)

qd = np.sin(2*m.pi*timestep)  # 期望转角
# qd = np.ones(len(timestep))/57.3*1  # 期望转角


angle = np.zeros([2, len(timestep)])
angle[0, :] = qd

# qt = 0
angle_single_save = np.zeros([1, len(timestep)]).reshape([len(timestep)])
current_q = np.zeros([1, len(timestep)]).reshape([len(timestep)])
env.reset()
state = env.get_obs_obstacle()
for i in range(len(timestep)):
    angle_single = angle[:, i].reshape([1, 2]) - state[0:6].reshape([1, 2])
    state = env.move456(angle_single, True)
    current_q[i] = state[0]
    angle_single_save[i] = angle_single[0, 0]
    env.show()
    time.sleep(0.01)


env.close()
fig, ax = plt.subplots()
ax.plot(timestep, current_q, label="current_q")
ax.plot(timestep, qd, label="desired_q")

ax.legend()

plt.show()
