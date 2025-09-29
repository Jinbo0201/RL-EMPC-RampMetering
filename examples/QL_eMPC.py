from src.agent import *
from src.agent.qlearningTrain import train_ql_agent
from src.agent.qlearningVerify import verify_ql_agent

episodes = 20
agent_path = train_ql_agent(EPI=episodes)
# agent_path = "..\models\q_table_fewer_2025-09-26_15-46-20.pkl"

verify_ql_agent(agent_path)
