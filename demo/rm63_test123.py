import gymnasium as gym
import torch
import rl_utils
import manipulator_mujoco.rl_program.SAC as SAC_train
import matplotlib.pyplot as plt
import numpy as np
import math as m

env = gym.make('manipulator_mujoco/rm63Env-v0', render_mode='human')
timestep = np.arange(0, 1.5, 0.01)

# qd = np.sin(2*m.pi*timestep)  # 期望转角
qd = np.ones(len(timestep))/57.3*5  # 期望转角

for idx in [1]:
    angle = np.zeros([6, len(timestep)])
    angle[idx, :] = qd

    # qt = 0
    current_q = np.zeros([1, len(timestep)]).reshape([len(timestep)])
    current_v = np.zeros([1, len(timestep)]).reshape([len(timestep)])
    current_a = np.zeros([1, len(timestep)]).reshape([len(timestep)])

    env.reset()
    state = env.get_obs()
    vel, acc = env.get_rm63_vel_acc()
    flag = 0
    idx_adj = 0
    for i in range(len(timestep)):
        angle_single = angle[:, i].reshape([1, 6]) - state[0:6].reshape([1, 6])
        current_q[i] = state[idx]
        current_v[i] = vel[idx]
        current_a[i] = acc[idx]
        print(i, acc)
        state = env.move123(angle_single, True)
        vel, acc = env.get_rm63_vel_acc()
        if flag == 0 and np.abs(angle_single[0, idx]) < 1e-4 and np.abs(state[idx + 6]):
            idx_adj = i
            flag = 1


    env.close()
    fig, ax = plt.subplots()
    ax.plot(timestep, current_a, label="current_q")
    ax.plot(timestep, qd, label="desired_q")
    ax.legend()

    plt.show()

    t_adj = timestep[idx_adj]
    print("idx = " + np.str_(idx) + "调节时间为" + np.str_(t_adj))

# ax.legend()
#
# plt.show()
