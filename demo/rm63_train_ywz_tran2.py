import gymnasium as gym
import torch
import rl_utils_ywz as rl
import manipulator_mujoco.rl_program.TD3 as TD3
import manipulator_mujoco.rl_program.KTMTD3 as KTMTD3
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# my_render_mode = 'human'
my_render_mode = "rgb_array"
env_expert = gym.make('Rm63Env-s1', render_mode=my_render_mode)
env = gym.make('Rm63Env-s2-1', render_mode=my_render_mode)
n_features = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]  # 正负
action_bound = env.action_space.high[0]  # 动作最大值
np.random.seed(0)
torch.manual_seed(0)

gamma = 0.95  # 更新学习率
tau = 0.002  # 软更新参数
actor_lr = 1e-4  # actor优化器学习率
critic_lr = 1e-3  # critic优化器学习率
n_hiddens = 128  # 隐藏层宽度
policy_noise = 0.01
noise_clip = 0.01
exploration_noise = 0.01  # 噪声标准差
policy_delay = 2

num_episodes = 50000  # 总训练循环数
buffer_size = 2 ** 18  # 样本缓存数目
minimal_size = 10000  # 最小训练总样本数
batch_size = 512  # 取样样本数
epsino_max = 1
epsino_min = 0.0
epsino_descend_ratio = 0.009
offline_episodes = 1e6  # 离线训练循环数
beta1 = 1  # KTM update 参数
beta2 = 1  # KTM update 参数

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

replay_buffer = rl.ReplayBuffer(buffer_size)
replay_buffer_offline = rl.ReplayBuffer(buffer_size)
# 训练好的专家模型
agent_expert = TD3.TD3Continuous(n_features, n_hiddens, n_actions, action_bound, policy_noise, noise_clip,
                                 exploration_noise, gamma, policy_delay, tau, actor_lr, critic_lr, device)
# 待训练的新模型
agent = TD3.TD3Continuous(n_features, n_hiddens, n_actions, action_bound, policy_noise, noise_clip,
                          exploration_noise, gamma, policy_delay, tau, actor_lr, critic_lr, device)
actor_path_1 = "actor_result_1_obs1.pth"
critic_1_path_1 = "critic_1_result_1_obs1.pth"
critic_2_path_1 = "critic_2_result_1_obs1.pth"
agent_expert.load_net_para(actor_path=actor_path_1, critic_1_path=critic_1_path_1, critic_2_path=critic_2_path_1)

save_file = "2_3_tran1_obs1"
if not os.path.exists(save_file):
    os.mkdir(save_file)

return_list = rl.train_tran(env, agent, agent_expert, num_episodes, replay_buffer, minimal_size, batch_size,
                            epsino_max, epsino_min, epsino_descend_ratio, save_file)

torch.save(agent.actor.state_dict(), 'actor_result_2_3_tran1_obs1.pth')
torch.save(agent.critic_1.state_dict(), 'critic_1_result_2_3_tran1_obs1.pth')
torch.save(agent.critic_2.state_dict(), 'critic_2_result_2_3_tran1_obs1.pth')

with open('return_list_2_3_tran1_obs1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(return_list)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 tran 2 on {}'.format(env))
plt.show()

mv_return = rl.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 tran 2 on {}'.format(env))
plt.show()
