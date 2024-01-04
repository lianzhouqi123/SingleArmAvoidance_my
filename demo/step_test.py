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

qd = 1*np.sin(2*m.pi*timestep)/57.3*25  # 期望转角
# qd = np.ones(len(timestep))/57.3*1  # 期望转角

angle = np.zeros([6, len(timestep)])
angle[0, :] = qd
angle[1, :] = qd
angle[2, :] = qd
angle = torch.tensor(angle)

# qt = 0
current_q = np.zeros([1, len(timestep)]).reshape([len(timestep)])
state, _ = env.reset()
for i in range(len(timestep)):
    angle_single = angle[:, i].reshape([1, 6]) - state[0:6].reshape([1, 6])
    state, _, done, _, _ = env.step(angle_single)
    current_q[i] = state[1]
    time.sleep(0.1)
    if done:
        break

env.close()
fig, ax = plt.subplots()
ax.plot(timestep, current_q, label="current_q")
ax.plot(timestep, qd, label="desired_q")

ax.legend()

plt.show()
