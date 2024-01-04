import time
from manipulator_mujoco.rl_program.TD3 import Actor
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
# 验证已训练好的模型

my_render_mode = 'human'
# my_render_mode = "rgb_array"
env = gym.make('Rm63Env-s4-2', render_mode=my_render_mode)
action_bound = env.action_space.high[0]
actor = Actor(22, 128, 6, action_bound)
# actor.load_state_dict(torch.load('actor_result_3_5_tran1_obs1.pth'))
actor.load_state_dict(torch.load('4_1_1tran2_3_act2/actor_8.pth'))
actor.eval()

n_episodes = 1000
i_terminate = 0
episode_reward = 0
reward_list = []
# test the model
for ii in range(n_episodes):
    state = env.reset()[0]
    episode_reward = 0
    with torch.no_grad():
        done = False
        terminate = False
        while not done:
            state = state.unsqueeze(0)
            state = torch.tensor(state, dtype=torch.float)
            action = actor(state)
            next_state, reward, done, terminate, info = env.step(action)
            episode_reward += reward
            if env.contact_detect != 0:
                terminate = False
                break
            time.sleep(0.03)
            state = next_state
        reward_list = np.append(reward_list, episode_reward)
        print(terminate)
        if terminate:
            i_terminate += 1
        time.sleep(0.3)
print("成功率为：{:.2f}%".format(i_terminate/n_episodes*100))

episodes_list = list(range(len(reward_list)))
plt.plot(episodes_list, reward_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('actor_result_3_1_2_tran3_5_tran1_obs1 test on s3-1')
plt.show()
