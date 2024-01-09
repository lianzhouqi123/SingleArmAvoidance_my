import time
from manipulator_mujoco.rl_program.TD3 import Actor
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
# 验证已训练好的模型

# 测试参数
env_id = "Rm63Env-s4-3"
actor_path = "actor_result_4_3_1.pth"
n_episodes = 1000  # 测试循环数
test_mode = "correct_ratio"  # 正确率模式，不显示图形，不进行延时
test_mode = "show"  # 展示模式，显示图形，并对每步进行延时

if test_mode == "show":
    my_render_mode = "human"
else:
    my_render_mode = "rgb_array"
env = gym.make(env_id, render_mode=my_render_mode)
action_bound = env.action_space.high[0]
actor = Actor(22, 128, 6, action_bound)
actor.load_state_dict(torch.load(actor_path))
actor.eval()

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
            if test_mode == "show":
                time.sleep(0.03)
            state = next_state
        reward_list = np.append(reward_list, episode_reward)
        print(terminate)
        if terminate:
            i_terminate += 1
        if test_mode == "show":
            time.sleep(0.3)
print("成功率为：{:.2f}%".format(i_terminate/n_episodes*100))

episodes_list = list(range(len(reward_list)))
plt.plot(episodes_list, reward_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('actor_result_3_1_2_tran3_5_tran1_obs1 test on s3-1')
plt.show()
