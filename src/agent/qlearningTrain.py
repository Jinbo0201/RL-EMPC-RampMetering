import numpy as np
from src.mpc.mpcOpt import *
import pickle
import datetime
import os
from pathlib import Path

from src.utils.discrete_state import discretize_state

# 定义状态空间和动作空间的大小
state_space_size = (18, 18, 10, 10)
action_space_size = 2

# 初始化Q值表
Q = np.zeros(state_space_size + (action_space_size,))

# 定义超参数
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 5

# 定义环境
env = MPCEnv()


# print(env.reset())


rewards = []

# Q-learning算法
for episode in range(num_episodes):
    # 初始化环境

    state = discretize_state(env.reset())
    done = False
    total_reward = 0

    while not done:
        # 选择动作
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(action_space_size)
        else:
            action = np.argmax(Q[state])

        # 执行动作
        next_state, reward, done, _ = env.step_q(action)
        # print('next_state', next_state)
        # print('reward', reward)
        # print(next_state)
        next_state = discretize_state(next_state)
        # print(next_state)

        # print('action:', action)

        # 更新Q值
        Q[state][action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state][action])

        # 更新状态
        state = next_state

        total_reward += reward

    rewards.append(total_reward)
    print('step ', episode, 'total reward ', total_reward)

time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

filename = f"q_table_{time_string}.pkl"

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