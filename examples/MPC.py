from src.mpc.mpcOpt import *
import matplotlib.pyplot as plt



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

# plt.figure()
# 参数配置
for k in range(1000):

    # # case-1
    # mpc_env.step(0)

    # case-2
    if k % M == 0:
        mpc_env.step(1)
    else:
        mpc_env.step(0)


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
    queue_list_o_over.append(mpc_env.simu.state['queue_length_origin'] - QUEUE_MAX if mpc_env.simu.state['queue_length_origin'] - QUEUE_MAX > 0 else 0)
    queue_list_r_over.append(mpc_env.simu.state['queue_length_onramp'] - QUEUE_MAX if mpc_env.simu.state['queue_length_onramp'] - QUEUE_MAX > 0 else 0)


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