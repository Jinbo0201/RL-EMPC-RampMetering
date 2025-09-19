import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import datetime
import os
from pathlib import Path
import pickle

from src.mpc.mpcOpt import *
from src.utils.discrete_state import discretize_fewerstate



# 定义简单的DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 定义智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 超参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # self.epsilon_decay = 0.999

        self.learning_rate = 0.001
        self.batch_size = 64

        # 经验回放缓冲区
        self.memory = deque(maxlen=10000)

        # 主网络和目标网络
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        action = torch.argmax(act_values).item()
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([i[0] for i in minibatch])
        actions = torch.LongTensor([i[1] for i in minibatch])
        rewards = torch.FloatTensor([i[2] for i in minibatch])
        next_states = torch.FloatTensor([i[3] for i in minibatch])
        dones = torch.FloatTensor([i[4] for i in minibatch])

        # 获取当前预测
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 获取目标值
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失并优化
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target)
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # 保存模型
    def save_model(self, path):
        """保存模型参数到指定路径"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"模型已保存到 {path}")

    # 加载模型
    def load_model(self, path):
        """从指定路径加载模型参数"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"模型已从 {path} 加载")



# 训练智能体
def train_agent():
    env = MPCEnv()
    state_size = 2
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    episodes = 40
    for e in range(episodes):
        state = discretize_fewerstate(env.reset())
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step_train(action)
            next_state = discretize_fewerstate(next_state)
            agent.remember(state, action, reward, next_state, done)

            # print('step', state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            total_reward += reward

        # 每100个episode更新目标网络
        if e % 20 == 0:
            agent.update_target_model()

        print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"dqn_{time_string}.pth"
    # 构建上上级目录下的models文件夹路径
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir.parent.parent / "models"
    # 确保models目录存在，不存在则创建
    os.makedirs(models_dir, exist_ok=True)
    # 完整的保存路径
    save_path = models_dir / filename

    agent.save_model(save_path)







if __name__ == "__main__":
    train_agent()