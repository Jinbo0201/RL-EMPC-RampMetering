import numpy as np
import pickle
import datetime
from src.utils.discrete_state import cal_obser2state_ql
from src.mpc.mpcOpt import *

import os
from pathlib import Path

# 定义超参数

# EPI 训练回合数 1000~5000
# LR 学习率 0.1~0.5
# DIS_FACTOR 折扣因子 0.8~0.99
# EPS_START 探索率初始值 1
# EPS_END 探索率最小值 0.01
# EPS_DECAY 探索率衰减 0.995

def train_ql_agent(EPI = 20, LR = 0.1, DIS_FACTOR = 0.9, EPS_START = 1, EPS_END = 0.01, EPS_DECAY = 0.999):

    EPS = EPS_START

    num_episodes = EPI

    # 定义状态空间和动作空间的大小
    state_space_size = (18, 10)
    action_space_size = 2
    # 初始化Q值表
    Q = np.zeros(state_space_size + (action_space_size,))
    # 定义环境
    env = MPCEnv()

    rewards = []

    # Q-learning算法
    for episode in range(num_episodes):
        # 初始化环境

        observation = env.reset()
        state = cal_obser2state_ql(observation)

        done = False
        total_reward = 0

        action_list = []
        state_list = []


        while not done:

            # 选择动作
            if np.random.uniform(0, 1) < EPS:
                action = np.random.randint(action_space_size)
            else:
                action = np.argmax(Q[state])

            # 执行动作
            observation_next, reward, done, _ = env.step_train(action)
            state_next = cal_obser2state_ql(observation_next)

            Q[state][action] += LR * (reward + DIS_FACTOR * np.max(Q[state_next]) - Q[state][action])

            # 更新状态
            state = state_next
            total_reward += reward

            action_list.append(action)
            state_list.append(state)

            # 探索率更新
            if EPS > EPS_END:
                EPS = EPS * EPS_DECAY
            else:
                EPS = EPS_END

        rewards.append(total_reward)
        # print('step ', episode, 'total reward ', total_reward)
        # print('total_simulation_step', env.simu_step, env.control_step)
        print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {total_reward}, SUM of Action: {sum(action_list)}, EPS: {EPS}")
        # print(action_list)
        # print('sum of action', sum(action_list))

    time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"q_table_fewer_{time_string}.pkl"

    # 构建上上级目录下的models文件夹路径
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir.parent.parent / "models"

    # 确保models目录存在，不存在则创建
    os.makedirs(models_dir, exist_ok=True)

    # 完整的保存路径
    save_path = models_dir / filename

    with open(save_path, "wb") as f:
        pickle.dump(Q, f)

    print(f"Q表已保存到: {save_path}")

    return save_path


if __name__ == "__main__":
    print(train_ql_agent(10, LR = 0.1))