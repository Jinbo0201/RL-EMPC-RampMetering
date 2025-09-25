from src.agent.dqnTrain import train_dqn_agent
from src.agent.dqnVerify import verify_dqn_agent

episodes = 100
agent_path = train_dqn_agent(episodes)
verify_dqn_agent(agent_path)

