import gymnasium as gym
import torch
import rl_gan as rl
import manipulator_mujoco.rl_program.TD3 as TD3
import manipulator_mujoco.rl_program.KTMTD3 as KTMTD3
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# 环境参数
my_render_mode = 'human'
my_render_mode = "rgb_array"
env = gym.make('Rm63Env-s2-5', render_mode=my_render_mode)
state_size = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]  # 正负
action_bound = env.action_space.high[0]  # 动作最大值
np.random.seed(0)
torch.manual_seed(0)
# 目标设定
goal_size = env.goal_space.shape[0]
goal_low, goal_high = env.goal_space.low, env.goal_space.high

# RL参数
gamma = 0.98  # 更新学习率
tau = 0.003  # 软更新参数
actor_lr = 1e-4  # actor优化器学习率
critic_lr = 3e-4  # critic优化器学习率
n_hiddens = 128  # 隐藏层宽度
policy_noise = 0.01
noise_clip = 0.01
exploration_noise = 0.008  # 噪声标准差
policy_delay = 2

# gan 参数
evaluator_size = 1
state_noise_level = 0.01
gen_n_hiddens = 256  # 生成器隐含层
discr_n_hiddens = 128  # 判别器隐含层
gen_lr = 1e-3
discr_lr = 1e-3
distance_threshold = 3e-3
R_min = 0.1
R_max = 0.9

num_episodes = 30  # 总训练循环数
buffer_size = 2 ** 15  # 样本缓存数目
minimal_size = 10000  # 最小训练总样本数

batch_size_rl = 128
batch_size_gan = 64
num_new_goals = 200
num_old_goals = 100
num_rl = 500
num_gan = 200
num_iteration = 1  # 一回合训练次数

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# 待训练的新模型+

agent = TD3.TD3Continuous(state_size, n_hiddens, n_actions, action_bound, policy_noise, noise_clip,
                          exploration_noise, gamma, policy_delay, tau, actor_lr, critic_lr, device)
actor_path_2 = "1_2_1/actor_3.pth"
critic_1_path_2 = "1_2_1/critic_1_3.pth"
critic_2_path_2 = "1_2_1/critic_2_3.pth"
agent.load_net_para(actor_path=actor_path_2, critic_1_path=critic_1_path_2, critic_2_path=critic_2_path_2)

gan = rl.GoalGAN(state_size, evaluator_size, state_noise_level, goal_low, goal_high, gen_n_hiddens,
                 goal_size, discr_n_hiddens, gen_lr, discr_lr, device)

goals_buffer = rl.GoalCollection(goal_size, distance_threshold)

replay_buffer = rl.ReplayBuffer(buffer_size)

goal_label_buffer = rl.Goal_Label_Collection(goal_size, goal_low, goal_high, distance_threshold, R_min, R_max)

save_file = "2_2_5_2"
if not os.path.exists(save_file):
    os.mkdir(save_file)

# return_list, discri_list, actor_loss_save, dis_loss_save, gen_loss_save \
return_list, discri_list \
    = rl.run_train(env, agent, gan, goals_buffer, replay_buffer, goal_label_buffer,
                   num_episodes, minimal_size, batch_size_rl, batch_size_gan, num_iteration,
                   num_new_goals, num_old_goals, num_rl, num_gan, save_file)

torch.save(agent.actor.state_dict(), f'actor_result_{save_file}.pth')
torch.save(agent.critic_1.state_dict(), f'critic_1_result_{save_file}.pth')
torch.save(agent.critic_2.state_dict(), f'critic_2_result_{save_file}.pth')
torch.save(gan.gan.gen.state_dict(), f'gen_result_{save_file}.pth')
torch.save(gan.gan.discr.state_dict(), f'discr_result_{save_file}.pth')

with open('return_list_{}.csv'.format(save_file), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(return_list)
with open('discri_list_{}.csv'.format(save_file), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(discri_list)
# with open('actor_list_{}.csv'.format(save_file), 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(actor_loss_save)
# with open('dis_list_{}.csv'.format(save_file), 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(dis_loss_save)
# with open('gen_list_{}.csv'.format(save_file), 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(gen_loss_save)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('gan TD3 tran 2-5-2')
plt.show()

mv_return = rl.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('gan TD3 tran 2-5-2')
plt.show()

# episodes_dis_list = list(range(len(discri_list)))
# plt.plot(episodes_dis_list, discri_list)
# plt.xlabel('Episodes')
# plt.ylabel('discri')
# plt.title('gan tran 2-5-2')
# plt.show()
#
# mv_discri = rl.moving_average(discri_list, 9)
# plt.plot(episodes_dis_list, mv_discri)
# plt.xlabel('Episodes')
# plt.ylabel('discri')
# plt.title('gan tran 2-5-2')
# plt.show()
