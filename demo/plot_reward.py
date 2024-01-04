import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rl_utils_ywz as rl

# filename = 'return_list_3_5_tran1_obs1.csv'
filename = '2_3_tran1_obs1/return_list_11.csv'

df = pd.read_csv(filename, header=None)
return_list = df.values.reshape(-1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TD3 on {}'.format(filename))
plt.show()

# mv_return = rl.moving_average(return_list, 9)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('TD3 on s1')
# plt.show()