# Input Parameters
DELTA_T = 10 / 3600  # 仿真步长 h
LENGTH_H = 12  # 总时长 h
RANDOM_DEMAND_ORIGIN_CYCLE = 2  # 周期 h
RANDOM_DEMAND_ORIGIN_MAX = 2500
RANDOM_DEMAND_ORIGIN_MIN = 1500
RANDOM_DEMAND_ONRAMP_CYCLE = 2  # 周期 h
RANDOM_DEMAND_ONRAMP_MAX = 1600
RANDOM_DEMAND_ONRAMP_MIN = 500
RANDOM_DOWNSTREAM_DENSITY_CYCLE = 2  # 周期 h
RANDOM_DOWNSTREAM_DENSITY_MAX = 55
RANDOM_DOWNSTREAM_DENSITY_MIN = 20

# Network Parameters
NUM_SEGMENT = 3
ID_ONRAMP = 3 - 1
T = 10 / 3600  # 步长时间 h, 步长为10s
V_FREE = 102  # 自由速度 km/h
L = 1  # 路段长度 km
LAMBDA = 2  # 车道数
TAU = 18 / 3600  # 速度计算参数 h
ALPHA = 1.867  # 速度计算参数 常量
DENSITY_CRIT = 33.5  # 速度计算参数 vel/km/lane
NU = 60  # 速度计算参数 km^2/h
KAPPA = 40  # 速度计算参数 vel/km/lane
MU = 0.0122  # 速度计算参数 常量
CAPACITY_ORIGIN = 3500  # 入口最大容量 veh/h
CAPACITY_ONRAMP = 2000  # 上匝道最大容量 veh/h
DENSITY_MAX = 180  # 最大密度 veh/km/lane
QUEUE_MAX = 50 # 最大排队长度 veh
# PWA Parameters
PWA_MID = 75.98  # PWA转折点密度 veh/km/lane
PWA_MAX = DENSITY_MAX - PWA_MID  # PWA中最大密度 veh/km/lane
PWA_MIN = - PWA_MID  # PWA中最小密度 veh/km/lane
PWA_EPSILON = 0.000001  # PWA中极小正值
ALPHA_1 = -1.3  # PWA分段函数参数 常量
ALPHA_2 = -0.031  # PWA分段函数参数 常量
BETA_1 = 102  # PWA分段函数参数 常量
BETA_2 = 5.58  # PWA分段函数参数 常量
AUX_A_MAX = 10000000000
AUX_A_MIN = -10000000000
# MPCEnv Parameters
NP_C = 6  # 控制变量角度出发的控制步长
M = 3  # 控制量保持不变的仿真步长数
XI = 1  # 控制量变化成本系数
XI_W = 0.01 # 超长队伍惩罚系数
NP = NP_C * M  # 预测步长
DONE_STEP = int(RANDOM_DEMAND_ORIGIN_CYCLE / DELTA_T)
DONE_STEP_CONTROL = int(RANDOM_DEMAND_ORIGIN_CYCLE / DELTA_T / M)
# 标准化参数
V_MAX = 120  # 最大速度，用于标准化
FLOW_MAX = 8040  # 最大流量用于标准化
QUEUE_LENGTH_ONRAMP_MAX = 2000  # 最大匝道排队长度用于标准化

# Reward calculation
R_TTT = 1
R_QUEUE = R_TTT
R_ACTION = 0.1

