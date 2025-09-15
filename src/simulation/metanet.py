import math
from metanetGym.input import *


class Metanet(object):

    def __init__(self):
        # Network Parameters
        self.NUM_SEGMENT = 3
        self.ID_ONRAMP = 3 - 1
        self.T = 10 / 3600  # 步长时间 h
        self.V_FREE = 102  # 自由速度 km/h
        self.L = 1  # 路段长度 km
        self.LAMBDA = 2  # 车道数
        self.TAU = 18 / 3600  # 速度计算参数 h
        self.ALPHA = 1.867  # 速度计算参数 常量
        self.DENSITY_CRIT = 33.5  # 速度计算参数 vel/km
        self.NU = 60  # 速度计算参数 km^2/h
        self.KAPPA = 40  # 速度计算参数 vel/km
        self.MU = 0.0122  # 速度计算参数 常量
        self.CAPACITY_ORIGIN = 3500  # 入口最大容量 veh/h
        self.CAPACITY_ONRAMP = 2000  # 上匝道最大容量 veh/h
        self.DENSITY_MAX = 180  # 最大密度 veh/km
        self.QUEUE_MAX = 50
        # 标准化参数
        self.V_MAX = 120  # 最大速度，用于标准化
        self.FLOW_MAX = 8040  # 最大流量用于标准化
        self.QUEUE_LENGTH_ONRAMP_MAX = 2000  # 最大匝道排队长度用于标准化
        # states
        self.state_density = [0] * self.NUM_SEGMENT
        self.state_flow = [0] * self.NUM_SEGMENT
        self.state_v = [self.V_FREE] * self.NUM_SEGMENT
        self.state_queue_length_origin = 0  # 入口处的队伍长度
        self.state_queue_length_onramp = 0  # 上匝道的队伍长度
        self.state_flow_onramp = [0] * self.NUM_SEGMENT
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
        self.state_density = [0] * self.NUM_SEGMENT
        self.state_flow = [0] * self.NUM_SEGMENT
        self.state_v = [self.V_FREE] * self.NUM_SEGMENT
        self.state_queue_length_origin = 0  # 入口处的队伍长度
        self.state_queue_length_onramp = 0  # 上匝道的队伍长度
        self.state_flow_onramp = [0] * self.NUM_SEGMENT
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
        for id_segment in range(self.NUM_SEGMENT):
            self.state_flow[id_segment] = self.LAMBDA * self.state_density[id_segment] * self.state_v[id_segment]

    def _cal_state_v(self):
        for id_segment in range(self.NUM_SEGMENT):
            if id_segment == 0:
                self.state_v[id_segment] = self.state_v[id_segment] + self.T / self.TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + self.T / self.L * (
                                                   self.state_v[id_segment] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (self.NU * self.T) / (
                                                   self.TAU * self.L) * (
                                                   self.state_density[id_segment + 1] - self.state_density[
                                               id_segment]) / (
                                                   self.state_density[id_segment] + self.KAPPA)
            elif id_segment == self.NUM_SEGMENT - 1:
                self.state_v[id_segment] = self.state_v[id_segment] + self.T / self.TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + self.T / self.L * (
                                                   self.state_v[id_segment - 1] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (self.NU * self.T) / (
                                                   self.TAU * self.L) * (
                                                   self._get_destination_flow_max() - self.state_density[
                                               id_segment]) / (
                                                   self.state_density[id_segment] + self.KAPPA) - (
                                                   self.MU * self.T * self.state_flow_onramp[id_segment] *
                                                   self.state_v[id_segment]) / (
                                                   self.L * self.LAMBDA * (
                                                   self.state_density[id_segment] + self.KAPPA))
            else:
                self.state_v[id_segment] = self.state_v[id_segment] + self.T / self.TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + self.T / self.L * (
                                                   self.state_v[id_segment - 1] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (self.NU * self.T) / (
                                                   self.TAU * self.L) * (
                                                   self.state_density[id_segment + 1] - self.state_density[
                                               id_segment]) / (
                                                   self.state_density[id_segment] + self.KAPPA)

    def _cal_state_density(self):
        for id_segment in range(self.NUM_SEGMENT):
            if id_segment == 0:
                self.state_density[id_segment] = self.state_density[id_segment] + self.T / (
                        self.L * self.LAMBDA) * (self._get_flow_origin() - self.state_flow[id_segment])
            elif self.NUM_SEGMENT - 1:
                self.state_density[id_segment] = self.state_density[id_segment] + self.T / (
                        self.L * self.LAMBDA) * (self.state_flow[id_segment - 1] - self.state_flow[id_segment] +
                                                 self.state_flow_onramp[id_segment])
            else:
                self.state_density[id_segment] = self.state_density[id_segment] + self.T / (
                        self.L * self.LAMBDA) * (self.state_flow[id_segment - 1] - self.state_flow[id_segment])

    def _get_Ve(self, density):
        return self.V_FREE * math.exp(-1 / self.ALPHA * (density / self.DENSITY_CRIT) ** self.ALPHA)

    def _cal_queue_length_origin(self):
        self.state_queue_length_origin = self.state_queue_length_origin + self.T * (
                self.input_demand_origin - self._get_flow_origin())

    def _cal_queue_length_onramp(self):
        self.state_queue_length_onramp = self.state_queue_length_onramp + self.T * (
                self.input_demand_onramp - self.state_flow_onramp[self.ID_ONRAMP])

    def _get_flow_origin(self):
        # value = min(self.input_demand_origin + self.state_queue_length_origin / self.T, self.CAPACITY_ORIGIN,
        #             self.CAPACITY_ORIGIN * (self.DENSITY_MAX - self.state_density[0]) / (self.DENSITY_MAX - self.DENSITY_CRIT))
        return min(self.input_demand_origin + self.state_queue_length_origin / self.T,
                   self.CAPACITY_ORIGIN * (self.DENSITY_MAX - self.state_density[0]) / (
                           self.DENSITY_MAX - self.DENSITY_CRIT),
                   self.action[0] * self.CAPACITY_ORIGIN)

    # def _get_flow_onramp_min(self):
    #     value = min(self.input_demand_onramp + self.state_queue_length_onramp / self.T, self.CAPACITY_ONRAMP,
    #                 self.CAPACITY_ONRAMP * (self.DENSITY_MAX - self.state_density[self.ID_ONRAMP]) / (
    #                             self.DENSITY_MAX - self.DENSITY_CRIT))
    #     return value

    def _cal_flow_onramp(self):
        self.state_flow_onramp[self.ID_ONRAMP] = min(self.input_demand_onramp + self.state_queue_length_onramp / self.T,
                                                     self.CAPACITY_ONRAMP * (self.DENSITY_MAX - self.state_density[
                                                         self.ID_ONRAMP]) / (self.DENSITY_MAX - self.DENSITY_CRIT),
                                                     self.action[1] * self.CAPACITY_ONRAMP)
        # print('print(self.action)', self.action)

    def _get_destination_flow_max(self):
        value = self.input_downsteam_density
        return value
