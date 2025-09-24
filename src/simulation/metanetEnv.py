import gym
from gym import spaces
import math
from src.simulation.input import Input
from src.config.constants import *


class MetanetEnv(gym.Env):
    def __init__(self):
        # 定义动作空间和状态空间
        # self.action_space = spaces.Discrete(5)  # 离散动作，分别对应[0，0.25，0.5，0.75，1]，因此需要乘以VALUE_ACTION2ACTION = 0.25
        self.action_space = spaces.Box(low=0, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,))  # flow, speed, density, queue_on_ramp
        # 定义METANET
        self.metanet = Metanet()
        # 初始化环境的内部状态
        self.action = None
        self.state = None
        self.observation = None
        self.reward = None
        self.time_step = 0

    def reset(self):
        self.metanet.init_state()
        # 重置环境的状态
        self.action = None
        self.state = self.metanet.get_state()
        self.observation = self._get_observation()
        self.reward = None
        self.time_step = 0
        return self.observation

    def set_state(self, state, step_id):
        self.metanet.state_density = state['density']
        self.metanet.state_flow = state['flow']
        self.metanet.state_v = state['v']
        self.metanet.state_queue_length_origin = state['queue_length_origin']
        self.metanet.state_queue_length_onramp = state['queue_length_onramp']
        self.metanet.state_flow_onramp = state['flow_onramp']
        self.metanet.step_id = step_id

    def step(self, action):
        # 执行动作并返回下一个状态、奖励和是否终止的标志
        # 判断action是否在范围内
        self.action = action
        # assert self.action_space.contains([self.action]), "Invalid action"
        # 输入动作，根据动作步进仿真
        self.metanet.step_state(self.action)
        # 获取状态量
        self.state = self.metanet.get_state()
        self.observation = self._get_observation()
        # 计算奖励
        self.reward = self._calculate_reward()
        # 赋值仿真步数
        self.time_step = self.metanet.step_id
        # 判断是否终止
        done = self._is_done()
        # 返回下一个状态、奖励和是否终止的标志
        return self.observation, self.reward, done, {}

    def _calculate_reward(self):
        # 根据当前状态计算奖励
        reward_online = T * sum(
            self.state['density']) * L * LAMBDA
        reward_queue = T * (self.state['queue_length_origin'] + self.state['queue_length_onramp'])
        # reward_action = 0.1 * self.action
        reward_action = 0
        return reward_online + reward_queue + reward_action

    def _is_done(self):
        # 判断是否终止
        # 训练6个小时
        return self.time_step * T > LENGTH_H

    def _get_observation(self):
        observation = [
            self.state['flow'][0] / FLOW_MAX,
            self.state['flow'][1] / FLOW_MAX,
            self.state['flow'][2] / FLOW_MAX,
            self.state['v'][0] / V_MAX,
            self.state['v'][1] / V_MAX,
            self.state['v'][2] / V_MAX,
            self.state['density'][0] / DENSITY_MAX ,
            self.state['density'][1] / DENSITY_MAX ,
            self.state['density'][2] / DENSITY_MAX ,
            self.state['queue_length_onramp'] / QUEUE_LENGTH_ONRAMP_MAX
        ]
        return observation

    def render(self):
        # 可选的渲染函数，用于可视化环境
        print('step:', self.metanet.step_id, ', action:', self.action, ', reward:', self.reward)
        print('state', self.state)


class Metanet(object):

    def __init__(self):
        # states
        self.state_density = [0] * NUM_SEGMENT
        self.state_flow = [0] * NUM_SEGMENT
        self.state_v = [V_FREE] * NUM_SEGMENT
        self.state_queue_length_origin = 0  # 入口处的队伍长度
        self.state_queue_length_onramp = 0  # 上匝道的队伍长度
        self.state_flow_onramp = [0] * NUM_SEGMENT
        # inputs
        self.input_demand_origin = 0  # 入口处的需求，即流量
        self.input_demand_onramp = 0  # 上匝道的需求，即流量
        self.input_downsteam_density = 0  # 出口处的密度
        # actions
        self.action = [0, 0]
        # step
        self.step_id = 0
        # 输入
        self.input = Input()

    # 初始化状态量
    def init_state(self):
        # states
        self.state_density = [0] * NUM_SEGMENT
        self.state_flow = [0] * NUM_SEGMENT
        self.state_v = [V_FREE] * NUM_SEGMENT
        self.state_queue_length_origin = 0  # 入口处的队伍长度
        self.state_queue_length_onramp = 0  # 上匝道的队伍长度
        self.state_flow_onramp = [0] * NUM_SEGMENT
        # inputs
        self.input_demand_origin = 0  # 入口处的需求，即流量
        self.input_demand_onramp = 0  # 上匝道的需求，即流量
        self.input_downsteam_density = 0  # 出口处的密度
        # actions
        self.action = [0, 0]
        # step
        self.step_id = 0

    # 步进仿真
    def step_state(self, action):
        self.action = action

        demand_o, demand_r, density_e = self.input.get_input(self.step_id, 1)
        self.input_demand_origin = demand_o[0]  # 入口处的需求，即流量
        self.input_demand_onramp = demand_r[0]  # 上匝道的需求，即流量
        self.input_downsteam_density = density_e[0]  # 出口处的密度

        self._cal_flow_onramp()

        self._cal_state_v()
        self._cal_state_density()
        self._cal_state_flow()

        self._cal_queue_length_onramp()
        self._cal_queue_length_origin()

        self.step_id += 1

    # 获取状态量
    def get_state(self):
        state_dict = {}
        state_dict['density'] = self.state_density
        state_dict['flow'] = self.state_flow
        state_dict['v'] = self.state_v
        state_dict['queue_length_origin'] = self.state_queue_length_origin
        state_dict['queue_length_onramp'] = self.state_queue_length_onramp
        state_dict['flow_onramp'] = self.state_flow_onramp
        state_dict['input'] = [self.input_demand_origin, self.input_demand_onramp, self.input_downsteam_density]
        state_dict['action'] = self.action
        return state_dict

    def _cal_state_flow(self):
        for id_segment in range(NUM_SEGMENT):
            self.state_flow[id_segment] = LAMBDA * self.state_density[id_segment] * self.state_v[id_segment]

    def _cal_state_v(self):
        for id_segment in range(NUM_SEGMENT):
            if id_segment == 0:
                self.state_v[id_segment] = self.state_v[id_segment] + T / TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + T / L * (
                                                   self.state_v[id_segment] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (NU * T) / (
                                                   TAU * L) * (
                                                   self.state_density[id_segment + 1] - self.state_density[
                                               id_segment]) / (
                                                   self.state_density[id_segment] + KAPPA)
            elif id_segment == NUM_SEGMENT - 1:
                self.state_v[id_segment] = self.state_v[id_segment] + T / TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + T / L * (
                                                   self.state_v[id_segment - 1] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (NU * T) / (
                                                   TAU * L) * (
                                                   self._get_destination_flow_max() - self.state_density[
                                               id_segment]) / (
                                                   self.state_density[id_segment] + KAPPA) - (
                                                   MU * T * self.state_flow_onramp[id_segment] *
                                                   self.state_v[id_segment]) / (
                                                   L * LAMBDA * (
                                                   self.state_density[id_segment] + KAPPA))
            else:
                self.state_v[id_segment] = self.state_v[id_segment] + T / TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + T / L * (
                                                   self.state_v[id_segment - 1] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (NU * T) / (
                                                   TAU * L) * (
                                                   self.state_density[id_segment + 1] - self.state_density[
                                               id_segment]) / (
                                                   self.state_density[id_segment] + KAPPA)

    def _cal_state_density(self):
        for id_segment in range(NUM_SEGMENT):
            if id_segment == 0:
                self.state_density[id_segment] = self.state_density[id_segment] + T / (
                        L * LAMBDA) * (self._get_flow_origin() - self.state_flow[id_segment])
            elif NUM_SEGMENT - 1:
                self.state_density[id_segment] = self.state_density[id_segment] + T / (
                        L * LAMBDA) * (self.state_flow[id_segment - 1] - self.state_flow[id_segment] +
                                                 self.state_flow_onramp[id_segment])
            else:
                self.state_density[id_segment] = self.state_density[id_segment] + T / (
                        L * LAMBDA) * (self.state_flow[id_segment - 1] - self.state_flow[id_segment])

    def _get_Ve(self, density):
        return V_FREE * math.exp(-1 / ALPHA * (density / DENSITY_CRIT) ** ALPHA)

    def _cal_queue_length_origin(self):
        self.state_queue_length_origin = self.state_queue_length_origin + T * (
                self.input_demand_origin - self._get_flow_origin())

    def _cal_queue_length_onramp(self):
        self.state_queue_length_onramp = self.state_queue_length_onramp + T * (
                self.input_demand_onramp - self.state_flow_onramp[ID_ONRAMP])

    def _get_flow_origin(self):
        # value = min(self.input_demand_origin + self.state_queue_length_origin / self.T, self.CAPACITY_ORIGIN,
        #             self.CAPACITY_ORIGIN * (self.DENSITY_MAX - self.state_density[0]) / (self.DENSITY_MAX - self.DENSITY_CRIT))
        return min(self.input_demand_origin + self.state_queue_length_origin / T,
                   CAPACITY_ORIGIN * (DENSITY_MAX - self.state_density[0]) / (
                           DENSITY_MAX - DENSITY_CRIT),
                   self.action[0] * CAPACITY_ORIGIN)


    def _cal_flow_onramp(self):
        self.state_flow_onramp[ID_ONRAMP] = min(self.input_demand_onramp + self.state_queue_length_onramp / T,
                                                     CAPACITY_ONRAMP * (DENSITY_MAX - self.state_density[
                                                         ID_ONRAMP]) / (DENSITY_MAX - DENSITY_CRIT),
                                                     self.action[1] * CAPACITY_ONRAMP)
        # print('print(self.action)', self.action)

    def _get_destination_flow_max(self):
        value = self.input_downsteam_density
        return value
