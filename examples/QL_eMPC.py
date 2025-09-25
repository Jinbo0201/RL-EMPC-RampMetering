from src.agent import *
from src.agent.qlearningTrain import train_ql_agent
from src.agent.qlearningVerify import verify_ql_agent

episodes = 100
agent_path = train_ql_agent(episodes)
verify_ql_agent(agent_path)
