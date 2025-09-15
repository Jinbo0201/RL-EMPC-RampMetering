from src.mpc.mpcOpt import *
import pickle
import numpy as np
import matplotlib.pyplot as plt


def discretize_state(state):
    discretized_state = []

    for i, element in enumerate(state):
        if i == 0:
            transformed_state = int(element // 10)
        elif i == 1:
            transformed_state = int(element // 10)
        elif i == 2:
            transformed_state = int(element // 10)
            if transformed_state > 9:
                transformed_state = 9
            elif transformed_state < 0:
                transformed_state = 0
        else:
            transformed_state = int(element // 10)
            if transformed_state > 9:
                transformed_state = 9
            elif transformed_state < 0:
                transformed_state = 0
        discretized_state.append(transformed_state)

    return tuple(discretized_state)


mpc_env = MPCEnv()
mpc_env.reset()

density_list_0 = []
density_list_1 = []
density_list_2 = []
queue_list_o = []
queue_list_r = []
action_list_o = []
action_list_r = []
queue_list_o_over = []
queue_list_r_over = []

with open("../models/q_table_2024-02-22_13-18-33.pkl", "rb") as f:
    q_table = pickle.load(f)

flag = 0

# plt.figure()
# 参数配置
for k in range(1000):

    # # case-1
    # mpc_env.step(0)
    # self.simu.state['density'][1], self.simu.state['density'][2],
    # self.simu.state['queue_length_origin'], self.simu.state['queue_length_onramp']
    state = discretize_state(
        [mpc_env.simu.state['density'][1], mpc_env.simu.state['density'][2], mpc_env.simu.state['queue_length_origin'],
         mpc_env.simu.state['queue_length_onramp']])
    # case-2
    action_opt = np.argmax(q_table[state])
    print(action_opt)
    if flag >= 5 and np.argmax(q_table[state]):
        mpc_env.step(1)
        flag = 0
    else:
        mpc_env.step(0)

    flag += 1
    # # case-3
    # if k % (2*M) == 0:
    #     mpc_env.step(1)
    # else:
    #     mpc_env.step(0)

    # # case-4
    # if k % (3*M) == 0:
    #     mpc_env.step(1)
    # else:
    #     mpc_env.step(0)

    # case-5
    # if k % (4*M) == 0:
    #     mpc_env.step(1)
    # else:
    #     mpc_env.step(0)

    density_list_0.append(mpc_env.simu.state['density'][0])
    density_list_1.append(mpc_env.simu.state['density'][1])
    density_list_2.append(mpc_env.simu.state['density'][2])
    queue_list_o.append(mpc_env.simu.state['queue_length_origin'])
    queue_list_r.append(mpc_env.simu.state['queue_length_onramp'])
    action_list_o.append(mpc_env.simu.state['action'][0])
    action_list_r.append(mpc_env.simu.state['action'][1])
    queue_list_o_over.append(mpc_env.simu.state['queue_length_origin'] - QUEUE_MAX if mpc_env.simu.state[
                                                                                          'queue_length_origin'] - QUEUE_MAX > 0 else 0)
    queue_list_r_over.append(mpc_env.simu.state['queue_length_onramp'] - QUEUE_MAX if mpc_env.simu.state[
                                                                                          'queue_length_onramp'] - QUEUE_MAX > 0 else 0)

obj_value = (sum(density_list_0) + sum(density_list_1) + sum(density_list_2)) * L * LAMBDA * T + (
        sum(queue_list_o) + sum(queue_list_r)) * T + (sum(queue_list_o_over) + sum(queue_list_r_over)) * XI_W
print('obj_value', obj_value)

plt.figure()
plt.plot(density_list_0, label='S-1')
plt.plot(density_list_1, label='S-2')
plt.plot(density_list_2, label='S-3')
plt.legend()

plt.figure()
plt.plot(queue_list_o, label='w-o')
plt.plot(queue_list_r, label='w-r')
plt.legend()

plt.figure()
plt.plot(queue_list_o_over, label='w-o-over')
plt.plot(queue_list_r_over, label='w-r-over')
plt.legend()

plt.figure()
plt.plot(action_list_o, label='a-o')
plt.plot(action_list_r, label='a-r')
plt.legend()

plt.show()
