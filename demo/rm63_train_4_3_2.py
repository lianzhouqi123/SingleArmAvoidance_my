import gymnasium as gym
import torch
import rl_utils_ywz as rl
import manipulator_mujoco.rl_program.TD3 as TD3
import manipulator_mujoco.rl_program.KTMTD3 as KTMTD3
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

my_render_mode = 'human'
my_render_mode = "rgb_array"
env = gym.make('Rm63Env-s4-3', render_mode=my_render_mode)
n_features = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]  # 正负
action_bound = env.action_space.high[0]  # 动作最大值
np.random.seed(0)
torch.manual_seed(0)

gamma = 0.98  # 更新学习率
tau = 0.003  # 软更新参数
actor_lr = 1e-4  # actor优化器学习率
critic_lr = 1e-4  # critic优化器学习率
n_hiddens = 128  # 隐藏层宽度
policy_noise = 0.01
noise_clip = 0.01
exploration_noise = 0.008  # 噪声标准差
policy_delay = 2

num_episodes = 8000  # 总训练循环数
buffer_size = 2 ** 15  # 样本缓存数目
minimal_size = 10000  # 最小训练总样本数
batch_size = 64  # 取样样本数
num_iteration = 3  # 一回合训练次数

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

replay_buffer = rl.ReplayBuffer(buffer_size)
# 待训练的新模型
agent = TD3.TD3Continuous(n_features, n_hiddens, n_actions, action_bound, policy_noise, noise_clip,
                          exploration_noise, gamma, policy_delay, tau, actor_lr, critic_lr, device)
actor_path_2 = "actor_result_2_2_3.pth"
critic_1_path_2 = "critic_1_result_2_2_3.pth"
critic_2_path_2 = "critic_2_result_2_2_3.pth"
agent.load_net_para(actor_path=actor_path_2, critic_1_path=critic_1_path_2, critic_2_path=critic_2_path_2)

save_file = "4_3_3"
if not os.path.exists(save_file):
    os.mkdir(save_file)

return_list = rl.train2(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, num_iteration, save_file)

torch.save(agent.actor.state_dict(), 'actor_result_4_3_3.pth')
torch.save(agent.critic_1.state_dict(), 'critic_1_result_4_3_3.pth')
torch.save(agent.critic_2.state_dict(), 'critic_2_result_4_3_3.pth')

with open('return_list_4_3_3.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(return_list)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 tran 4-3-3 on {}'.format(env))
plt.show()

mv_return = rl.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 tran 4-3-3 on {}'.format(env))
plt.show()
