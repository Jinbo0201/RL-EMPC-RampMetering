import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from src.mpc.mpcOpt import *
from src.utils.discrete_state import discretize_fewerstate

class Actor(nn.Module):
    """策略网络，输入状态，输出动作概率分布"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)  # 输出动作概率
        return x


class Critic(nn.Module):
    """价值网络，输入状态，输出状态价值"""

    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出状态价值

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOAgent:
    """PPO智能体"""

    def __init__(self, state_dim=2, action_dim=2, hidden_dim=64,
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, K_epochs=10):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.gae_lambda = gae_lambda  # GAE参数
        self.clip_epsilon = clip_epsilon  # PPO剪辑参数
        self.K_epochs = K_epochs  # 每轮更新的epochs数

        # 初始化网络
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        # 优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def select_action(self, state):
        """根据当前策略选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)  # 转换为张量并增加批次维度
        action_probs = self.actor(state)
        dist = Categorical(action_probs)  # 分类分布
        action = dist.sample()  # 采样动作
        log_prob = dist.log_prob(action)  # 计算动作的对数概率

        return action.item(), log_prob.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """存储转换样本"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def compute_gae(self):
        """计算广义优势估计(GAE)"""
        states = torch.FloatTensor(self.states)
        next_states = torch.FloatTensor(self.next_states)
        dones = torch.FloatTensor(self.dones).unsqueeze(1)
        rewards = torch.FloatTensor(self.rewards).unsqueeze(1)

        # 计算状态价值
        values = self.critic(states)
        next_values = self.critic(next_states)

        # 计算TD误差
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        deltas = deltas.detach().numpy()

        # 计算GAE
        advantages = np.zeros_like(deltas)
        advantage = 0.0
        for t in reversed(range(len(deltas))):
            # print(deltas[t])
            # print(type(deltas[t]))
            # print(self.gamma * self.gae_lambda * advantage * (1 - dones[t]))
            # print(type(self.gamma * self.gae_lambda * advantage * (1 - dones[t])))

            advantage = deltas[t] + self.gamma * self.gae_lambda * advantage * (1 - dones[t]).numpy()
            advantages[t] = advantage

        # 计算回报
        returns = advantages + values.detach().numpy()

        # 标准化优势
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return torch.FloatTensor(advantages), torch.FloatTensor(returns), values

    def update(self):
        """更新PPO网络"""
        # 计算优势和回报
        advantages, returns, values = self.compute_gae()

        # 转换为张量
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions).unsqueeze(1)
        old_log_probs = torch.FloatTensor(self.log_probs).unsqueeze(1)

        # 多次更新网络
        for _ in range(self.K_epochs):
            # 计算当前策略的动作概率和对数概率
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions.squeeze(1)).unsqueeze(1)

            # 计算概率比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 计算PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 更新策略网络
            self.optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optimizer_actor.step()

            # 计算价值损失
            new_values = self.critic(states)
            critic_loss = F.mse_loss(new_values, returns)

            # 更新价值网络
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        # 清空存储的轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

        return actor_loss.item(), critic_loss.item()


# 使用示例
if __name__ == "__main__":

    env = MPCEnv()

    # 初始化PPO智能体，状态空间为2，动作空间为2
    agent = PPOAgent(state_dim=2, action_dim=2)

    # 模拟训练过程（这里使用随机生成的状态和奖励作为示例）
    num_episodes = 100
    max_steps = 50

    for episode in range(num_episodes):
        # 随机初始状态（2维）
        # state = np.random.randn(2)
        state = discretize_fewerstate(env.reset())
        total_reward = 0
        done = False

        while not done:
            # 选择动作
            action, log_prob = agent.select_action(state)

            # 模拟环境反馈（这里使用随机奖励和下一状态作为示例）
            next_state, reward, done, _ = env.step_train(action)
            next_state = discretize_fewerstate(next_state)

            # 存储转换
            agent.store_transition(state, action, reward, next_state, done, log_prob)

            state = next_state
            total_reward += reward

            # 如果结束，更新网络
            if done:
                actor_loss, critic_loss = agent.update()
                break

        # 打印训练信息
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
