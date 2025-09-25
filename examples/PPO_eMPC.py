from src.agent.ppoTrain import train_ppo_agent
from src.agent.ppoVerify import verify_ppo_agent

episodes = 1
agent_path = train_ppo_agent(episodes)

verify_ppo_agent(agent_path)

