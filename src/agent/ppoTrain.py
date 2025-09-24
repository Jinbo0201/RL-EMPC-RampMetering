import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
import datetime
from pathlib import Path

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

    def save_model(self, path):
        """
        保存模型参数

        参数:
            path (str): 保存模型的目录路径
            episode (int, optional): 当前训练的回合数，用于在文件名中标识
        """
        # # 确保保存目录存在
        # os.makedirs(path, exist_ok=True)


        # actor_path = os.path.join(path, "actor.pth")
        # critic_path = os.path.join(path, "critic.pth")

        # print(f"模型已保存至: {path}")

        # 保存模型参数
        torch.save(self.actor.state_dict(), path)
        # torch.save(self.critic.state_dict(), critic_path)
        print(f"模型已保存至: {path}")

    def load_model(self, path):
        """
        加载模型参数

        参数:
            path (str): 模型文件所在的目录路径
            episode (int, optional): 要加载的回合数对应的模型
        """
        # # 根据是否提供回合数构建不同的文件名
        # if episode is not None:
        #     actor_path = os.path.join(path, f"actor_episode_{episode}.pth")
        #     critic_path = os.path.join(path, f"critic_episode_{episode}.pth")
        # else:
        #     actor_path = os.path.join(path, "actor.pth")
        #     critic_path = os.path.join(path, "critic.pth")

        # 加载模型参数
        self.actor.load_state_dict(torch.load(path))
        # self.critic.load_state_dict(torch.load(critic_path))
        print(f"已加载模型: {path}")

    # 在实际生产环境下产生动作
    def act_real(self, state):
        """
        用于实际部署时根据状态生成动作
        加载模型后使用，不计算或存储日志概率，仅返回动作

        参数:
            state: 环境状态

        返回:
            action: 生成的动作
        """
        # 设置为评估模式
        self.actor.eval()

        # 不追踪梯度，提高推理速度
        with torch.no_grad():
            # 转换状态为张量并增加批次维度
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # 获取动作概率分布
            action_probs = self.actor(state_tensor)
            # 创建分类分布并采样动作
            dist = Categorical(action_probs)
            action = dist.sample()

        # # 恢复训练模式（可选，根据使用场景）
        # self.actor.train()

        return action.item()


# 使用示例
if __name__ == "__main__":

    env = MPCEnv()

    # 初始化PPO智能体，状态空间为2，动作空间为2
    agent = PPOAgent(state_dim=2, action_dim=2)

    # 模拟训练过程（这里使用随机生成的状态和奖励作为示例）
    num_episodes = 20
    # max_steps = 50

    for episode in range(num_episodes):
        # 随机初始状态（2维）
        # state = np.random.randn(2)
        state = discretize_fewerstate(env.reset())
        total_reward = 0
        done = False
        action_list = []

        while not done:
            # 选择动作
            action, log_prob = agent.select_action(state)
            # print('action', 'log_prob', action, log_prob)

            # 模拟环境反馈（这里使用随机奖励和下一状态作为示例）
            next_state, reward, done, _ = env.step_train(action)
            next_state = discretize_fewerstate(next_state)

            # 存储转换
            agent.store_transition(state, action, reward, next_state, done, log_prob)

            state = next_state
            total_reward += reward

            action_list.append(action)

            # 如果结束，更新网络
            if done:
                actor_loss, critic_loss = agent.update()
                break

        # 打印训练信息
        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, "
              f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        print(action_list)
        print('sum of action', sum(action_list))

    time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"ppo_{time_string}.pth"
    # 构建上上级目录下的models文件夹路径
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir.parent.parent / "models"
    # 确保models目录存在，不存在则创建
    os.makedirs(models_dir, exist_ok=True)
    # 完整的保存路径
    save_path = models_dir / filename

    agent.save_model(save_path)
