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
event_data = []


# plt.figure()
# 参数配置
for k in range(1000):

    # # case-1
    # mpc_env.step(0)

    # case-2
    # if flag >= 5 and (mpc_env.simu.state['queue_length_origin'] > QUEUE_MAX or mpc_env.simu.state[
    #     'queue_length_onramp'] > QUEUE_MAX or mpc_env.simu.state['density'][1] > DENSITY_CRIT or
    #                   mpc_env.simu.state['density'][2] > DENSITY_CRIT):
    if (mpc_env.simu.state['queue_length_onramp'] > QUEUE_MAX or mpc_env.simu.state['density'][1] > DENSITY_CRIT):

        mpc_env.step(1)

        event_data.append(1)
        # event_data.append(0)
        # event_data.append(0)
        # event_data.append(0)
        # event_data.append(0)

    else:
        mpc_env.step(0)

        event_data.append(0)

    print('action is', event_data[-1])

    # flag += 1
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

ttt = (sum(density_list_0) + sum(density_list_1) + sum(density_list_2)) * L * LAMBDA * T + (
        sum(queue_list_o) + sum(queue_list_r)) * T
ttt = ttt * M
print('ttt', ttt)

print('sum_event', sum(event_data))

plt.figure(figsize=(4, 1.5))
plt.plot(density_list_0)
plt.plot(density_list_1)
plt.plot(density_list_2)
plt.axhline(y=33.5, color='lightgray', linestyle='--')
plt.xlim(0, 1000)
plt.ylim(-5, 120)
plt.xlabel('time step', fontsize=10, fontname='Times New Roman')  # X轴标签
plt.ylabel('traffic density', fontsize=10, fontname='Times New Roman')  # Y轴标签
plt.xticks(fontsize=8)  # X轴刻度字号
plt.yticks(fontsize=8)  # Y轴刻度字号
plt.legend(['s-1','s-2','s-3'], loc='best', fontsize=8, frameon=False)
# plt.savefig('../resources/empc_p.png', bbox_inches='tight', dpi=600)

plt.figure(figsize=(4, 1.5))
plt.plot(queue_list_o, label='w-1')
plt.plot(queue_list_r, label='w-3')
plt.axhline(y=50, color='lightgray', linestyle='--')
plt.xlim(0, 1000)
plt.ylim(-5, 200)
plt.xlabel('time step', fontsize=10, fontname='Times New Roman')  # X轴标签
plt.ylabel('queue length', fontsize=10, fontname='Times New Roman')  # Y轴标签
plt.xticks(fontsize=8)  # X轴刻度字号
plt.yticks(fontsize=8)  # Y轴刻度字号
plt.legend(['w-1','w-3'], loc='best', fontsize=8, frameon=False)
# plt.savefig('../resources/empc_w.png', bbox_inches='tight', dpi=600)

plt.figure(figsize=(4, 1.5))
plt.axhline(y=0, color='lightgray', linestyle='--')
plt.axhline(y=1, color='lightgray', linestyle='--')
plt.plot(event_data, 'o--')
plt.xlim(0, 1000)
plt.ylim(-0.1, 1.1)
plt.xlabel('time step', fontsize=10, fontname='Times New Roman')  # X轴标签
plt.ylabel('triggering command', fontsize=10, fontname='Times New Roman')  # Y轴标签
plt.xticks(fontsize=8)  # X轴刻度字号
plt.yticks(fontsize=8)  # Y轴刻度字号
# plt.savefig('../resources/empc_e.png', bbox_inches='tight', dpi=600)

# plt.figure()
# plt.plot(queue_list_o_over, label='w-o-over')
# plt.plot(queue_list_r_over, label='w-r-over')
# plt.legend()

# plt.figure()
# plt.plot(action_list_o, label='a-o')
# plt.plot(action_list_r, label='a-r')
# plt.legend()

plt.show()
