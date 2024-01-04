from manipulator_mujoco.rl_program.SAC import PolicyNetContinuous
import gymnasium as gym
import torch
import numpy as np
# 验证已训练好的模型
# 验证已训练好的模型

env = gym.make('manipulator_mujoco/rm63Env-v0', render_mode='human')
model = PolicyNetContinuous(22, 128, 6, 0.01)
model.load_state_dict(torch.load('test1.pth'))
model.eval()
state = env.reset()[0]
# test the model
# 1107_actor结果，有点飘起来，能接近，需要进一步优化rewards
with torch.no_grad():
    for i in range(600):
        state = torch.tensor(state, dtype=torch.float)
        action = model(state)[0]
        action = np.array([action.tolist()])
        next_state, reward, done, terminate, info = env.step(action)
        state = next_state
        print(reward)





