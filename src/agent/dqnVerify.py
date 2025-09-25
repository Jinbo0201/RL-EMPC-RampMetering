from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn
import torch.optim as optim

from src.mpc.mpcOpt import *
import numpy as np
import matplotlib.pyplot as plt

import os

from src.utils.discrete_state import cal_obser2state
from src.agent.dqnTrain import DQNAgent

import pickle

def replace_last_chars(text, new_suffix, count):
    """
    替换文本最后count个字符
    :param text: 原始文本
    :param new_suffix: 新的后缀内容
    :param count: 要替换的字符数量
    :return: 替换后的文本
    """
    if len(text) <= count:
        # 如果文本长度小于等于要替换的数量，直接返回新内容
        return new_suffix
    # 保留除最后count个字符外的部分，再拼接新内容
    return text[:-count] + new_suffix

def verify_dqn_agent(agent_path):


    model_path = agent_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    state_size = 2
    action_size = 2
    dqn_model = DQNAgent(state_size, action_size)

    dqn_model.load_model(model_path)

    dqn_model.model.eval()
    dqn_model.target_model.eval()

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


    mpc_env = MPCEnv()

    obser = mpc_env.reset()
    state = cal_obser2state(obser)



    for k in range(DONE_STEP_CONTROL):


        action_opt = dqn_model.act_real(state)
        print('step', k, 'action_opt', action_opt)
        # print('state', state)



        event_data.append(action_opt)

        obser = mpc_env.step(action_opt)
        state = cal_obser2state(obser)


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
    obj_value = obj_value * M
    print('obj_value', obj_value)


    ttt = (sum(density_list_0) + sum(density_list_1) + sum(density_list_2)) * L * LAMBDA * T + (
            sum(queue_list_o) + sum(queue_list_r)) * T
    ttt = ttt * M
    print('ttt', ttt)

    queue_over = (sum(queue_list_o_over) + sum(queue_list_r_over)) * XI_W
    queue_over = queue_over * M
    print('queue_over', queue_over)

    sum_event = sum(event_data)
    print('sum_event', sum_event)

    results = {
        'obj_value': obj_value,
        'ttt': ttt,
        'queue_over': queue_over,
        'event_data': event_data,
        'R_action': R_ACTION,
    }

    results_path = replace_last_chars(str(model_path), '-result.pkl', 4)

    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"结果已保存到: {results_path}")


    plt.figure(figsize=(4, 1.5))
    plt.plot(density_list_0)
    plt.plot(density_list_1)
    plt.plot(density_list_2)
    plt.axhline(y=33.5, color='lightgray', linestyle='--')
    plt.xlim(0, DONE_STEP_CONTROL)
    plt.ylim(-5, 120)
    plt.xlabel('time step', fontsize=10, fontname='Times New Roman')  # X轴标签
    plt.ylabel('traffic density', fontsize=10, fontname='Times New Roman')  # Y轴标签
    plt.xticks(fontsize=8)  # X轴刻度字号
    plt.yticks(fontsize=8)  # Y轴刻度字号
    plt.legend(['s-1','s-2','s-3'], loc='best', fontsize=8, frameon=False)
    # plt.savefig('../resources/rlempc_p.png', bbox_inches='tight', dpi=600)

    plt.figure(figsize=(4, 1.5))
    plt.plot(queue_list_o, label='w-1')
    plt.plot(queue_list_r, label='w-3')
    plt.axhline(y=50, color='lightgray', linestyle='--')
    plt.xlim(0, DONE_STEP_CONTROL)
    plt.ylim(-5, 200)
    plt.xlabel('time step', fontsize=10, fontname='Times New Roman')  # X轴标签
    plt.ylabel('queue length', fontsize=10, fontname='Times New Roman')  # Y轴标签
    plt.xticks(fontsize=8)  # X轴刻度字号
    plt.yticks(fontsize=8)  # Y轴刻度字号
    plt.legend(['w-1','w-3'], loc='best', fontsize=8, frameon=False)
    # plt.savefig('../resources/rlempc_w.png', bbox_inches='tight', dpi=600)

    plt.figure(figsize=(4, 1.5))
    plt.axhline(y=0, color='lightgray', linestyle='--')
    plt.axhline(y=1, color='lightgray', linestyle='--')
    plt.plot(event_data, 'o--')
    plt.xlim(0, DONE_STEP_CONTROL)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('time step', fontsize=10, fontname='Times New Roman')  # X轴标签
    plt.ylabel('triggering command', fontsize=10, fontname='Times New Roman')  # Y轴标签
    plt.xticks(fontsize=8)  # X轴刻度字号
    plt.yticks(fontsize=8)  # Y轴刻度字号
    # plt.savefig('../resources/rlempc_e.png', bbox_inches='tight', dpi=600)


    plt.show()


if __name__ == "__main__":

    agent_path = "../../models/dqn_2025-09-25_14-51-20.pth"
    verify_dqn_agent(agent_path)