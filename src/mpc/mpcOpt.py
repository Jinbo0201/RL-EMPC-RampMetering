from src.mpc.modelPyomo import *
from src.simulation.input import Input
from src.simulation.metanetEnv import MetanetEnv


class MPCEnv(object):

    def __init__(self):
        self.simu = MetanetEnv()
        self.model = mpc_model()
        self.input = Input()
        self.simu_step = 0
        self.control_step = 0
        self.action_opt = 0  # 是否进行MPC优化，0表示不进行优化，1表示进行优化
        self.action_o_list = []
        self.action_r_list = []
        self.index_action = 0
        self.reward = None
        self.observation = None

        self.opt_data_dict = {}

    def reset(self):
        self.simu.reset()
        self.simu_step = 0
        self.control_step = 0
        self.action = 0
        self.action_o_list = []
        self.action_r_list = []
        self.reward = None
        self.observation = [self.simu.state['density'][1], self.simu.state['density'][2],
                            self.simu.state['queue_length_origin'], self.simu.state['queue_length_onramp']]
        # self.observation = [self.simu.state['density'][1], self.simu.state['queue_length_onramp']]
        return self.observation

    def step(self, action):

        self.action_opt = action

        if self.action_opt == 1:
            self.action_o_list, self.action_r_list = self.solve_model()
            self.index_action = 0

            # print('self.action_o_list', self.action_o_list, 'self.action_r_list', self.action_r_list)

            for _ in range(M):
                action_o = self.action_o_list[self.index_action] if self.index_action < len(self.action_o_list) else 1
                action_r = self.action_r_list[self.index_action] if self.index_action < len(self.action_r_list) else 1
                self.simu.step([action_o, action_r])
                self.index_action += 1
                self.simu_step += 1


        else:

            for _ in range(M):
                action_o = self.action_o_list[self.index_action] if self.index_action < len(self.action_o_list) else 1
                action_r = self.action_r_list[self.index_action] if self.index_action < len(self.action_r_list) else 1
                self.simu.step([action_o, action_r])
                self.index_action += 1
                self.simu_step += 1


        # if self.action_opt == 1:
        #     self.action_o_list, self.action_r_list = self.solve_model()
        #     self.index_action = 0
        #     # print('self.action_o_list', self.action_o_list, 'self.action_r_list', self.action_r_list)
        #
        # action_o = self.action_o_list[self.index_action] if self.index_action < len(self.action_o_list) else 1
        # action_r = self.action_r_list[self.index_action] if self.index_action < len(self.action_r_list) else 1
        #
        # self.simu.step([action_o, action_r])
        # self.index_action += 1
        # self.simu_step += 1



    def step_train (self, action):
        self.action_opt = action

        reward_sum = 0

        if self.action_opt == 1:

            self.action_o_list, self.action_r_list = self.solve_model()
            self.index_action = 0


        for _ in range(M):

            action_o = self.action_o_list[self.index_action] if self.index_action < len(self.action_o_list) else 1
            action_r = self.action_r_list[self.index_action] if self.index_action < len(self.action_r_list) else 1
            self.simu.step([action_o, action_r])

            self.index_action += 1
            self.simu_step += 1

            queue_length_origin_over = self.simu.state['queue_length_origin'] - QUEUE_MAX if self.simu.state[
                                                                                                 'queue_length_origin'] - QUEUE_MAX > 0 else 0
            queue_length_onramp_over = self.simu.state['queue_length_onramp'] - QUEUE_MAX if self.simu.state[
                                                                                                 'queue_length_onramp'] - QUEUE_MAX > 0 else 0
            # reward_sum += 1/((self.simu.state['density'][0] + self.simu.state['density'][1] +
            #                 self.simu.state['density'][2]) * L * LAMBDA * T + (
            #                        self.simu.state['queue_length_origin'] + self.simu.state[
            #                    'queue_length_onramp']) * T + (queue_length_origin_over + queue_length_onramp_over) * XI_W)

            reward_ttt = (self.simu.state['density'][0] + self.simu.state['density'][1] + self.simu.state['density'][2]) * L * LAMBDA * T + (self.simu.state['queue_length_origin'] + self.simu.state['queue_length_onramp']) * T

            # 0.01是为了归一化处理, reward_over最大值在100左右
            reward_over = 0.01 * (queue_length_origin_over + queue_length_onramp_over) * XI_W

            reward_action = self.action_opt

            # print(reward_ttt, reward_over, reward_action)

            reward_sum += -(reward_ttt + 0.01* reward_over)

            self.observation = [self.simu.state['density'][1], self.simu.state['density'][2],
                                self.simu.state['queue_length_origin'], self.simu.state['queue_length_onramp']]
            # self.observation = [self.simu.state['density'][1], self.simu.state['queue_length_onramp']]

        # reward_return = reward_sum

        # else:
        #
        #     for _ in range(M):
        #
        #         action_o = self.action_o_list[self.index_action] if self.index_action < len(self.action_o_list) else 1
        #         action_r = self.action_r_list[self.index_action] if self.index_action < len(self.action_r_list) else 1
        #         self.simu.step([action_o, action_r])
        #         self.index_action += 1
        #         self.simu_step += 1
        #
        #         queue_length_origin_over = self.simu.state['queue_length_origin'] - QUEUE_MAX if self.simu.state[
        #                                                                                              'queue_length_origin'] - QUEUE_MAX > 0 else 0
        #         queue_length_onramp_over = self.simu.state['queue_length_onramp'] - QUEUE_MAX if self.simu.state[
        #                                                                                              'queue_length_onramp'] - QUEUE_MAX > 0 else 0
        #         reward_sum += - ((self.simu.state['density'][0] + self.simu.state['density'][1] +
        #                         self.simu.state['density'][2]) * L * LAMBDA * T + (
        #                                self.simu.state['queue_length_origin'] + self.simu.state[
        #                            'queue_length_onramp']) * T + (
        #                                    queue_length_origin_over + queue_length_onramp_over) * XI_W)
        #
        #     self.observation = [self.simu.state['density'][1], self.simu.state['density'][2],
        #                         self.simu.state['queue_length_origin'], self.simu.state['queue_length_onramp']]
        #
        #         # self.observation = [self.simu.state['density'][1], self.simu.state['queue_length_onramp']]
        #
        #     reward_return = reward_sum

        done = False
        if self.simu_step > 1000:
            done = True

        return self.observation, reward_sum, done, 1

    def solve_model(self):
        demand_o, demand_r, density_e = self.input.get_input(self.simu_step, NP)
        for j in range(NP):
            self.model.p_d_o[j] = demand_o[j]
            self.model.p_d_r[j] = demand_r[j]
            self.model.p_p_e[j] = density_e[j]

        for id_segment in range(NUM_SEGMENT):
            self.model.p_p[id_segment] = self.simu.state['density'][id_segment]
            self.model.p_v[id_segment] = self.simu.state['v'][id_segment]
            self.model.p_q[id_segment] = self.simu.state['flow'][id_segment]

        self.model.p_w_o = max(0, self.simu.state['queue_length_origin'])
        self.model.p_w_r = max(0, self.simu.state['queue_length_onramp'])

        # self.model.write('model.lp')
        solver = pyo.SolverFactory('ipopt')
        # print("求解器是否可用:", solver.available())
        # solver = pyo.SolverFactory('highs')

        results = solver.solve(self.model)

        # print("Step-", self.simu_step, "Optimization status:", results.solver.status,
        #       results.solver.termination_condition)
        # print("Optimal objective value:", pyo.value(self.model.obj))


        self.opt_data_dict = self.get_opt_data()

        # print(self.opt_data_dict)


        return self.opt_data_dict['q_list_o'], self.opt_data_dict['r_list']

    def get_opt_data(self):
        r_list = []
        w_list_r = []
        q_list_o = []
        w_list_o = []
        v_list_0 = []
        v_list_1 = []
        v_list_2 = []
        p_list_0 = []
        p_list_1 = []
        p_list_2 = []
        q_list_0 = []
        q_list_1 = []
        q_list_2 = []
        a_delta_list_0 = []
        a_delta_list_1 = []
        a_delta_list_2 = []
        for k in range(NP):
            # print(111)
            # print('pyo.value(self.model.x_r_r[change_NP2NC_r(k)])', pyo.value(self.model.x_r_r[change_NP2NC_r(k)]))
            r_list.append(pyo.value(self.model.x_r_r[change_NP2NC_r(k)]) / CAPACITY_ONRAMP)
            w_list_r.append(pyo.value(self.model.x_w_r[k]))
            q_list_o.append(pyo.value(self.model.x_q_o[change_NP2NC_r(k)]) / CAPACITY_ORIGIN)
            w_list_o.append(pyo.value(self.model.x_w_o[k]))
            # v_list_0.append(pyo.value(self.model.x_v[0, k + 1]))
            # v_list_1.append(pyo.value(self.model.x_v[1, k + 1]))
            # v_list_2.append(pyo.value(self.model.x_v[2, k + 1]))
            # p_list_0.append(pyo.value(self.model.x_p[0, k + 1]))
            # p_list_1.append(pyo.value(self.model.x_p[1, k + 1]))
            # p_list_2.append(pyo.value(self.model.x_p[2, k + 1]))
            # q_list_0.append(pyo.value(self.model.x_q[0, k + 1]))
            # q_list_1.append(pyo.value(self.model.x_q[1, k + 1]))
            # q_list_2.append(pyo.value(self.model.x_q[2, k + 1]))
            v_list_0.append(pyo.value(self.model.x_v[0, k]))
            v_list_1.append(pyo.value(self.model.x_v[1, k]))
            v_list_2.append(pyo.value(self.model.x_v[2, k]))
            p_list_0.append(pyo.value(self.model.x_p[0, k]))
            p_list_1.append(pyo.value(self.model.x_p[1, k]))
            p_list_2.append(pyo.value(self.model.x_p[2, k]))
            q_list_0.append(pyo.value(self.model.x_q[0, k]))
            q_list_1.append(pyo.value(self.model.x_q[1, k]))
            q_list_2.append(pyo.value(self.model.x_q[2, k]))
            a_delta_list_0.append(pyo.value(self.model.a_delta[0, k]))
            a_delta_list_1.append(pyo.value(self.model.a_delta[1, k]))
            a_delta_list_2.append(pyo.value(self.model.a_delta[2, k]))
        opt_data_dict = {}
        opt_data_dict['r_list'] = r_list
        opt_data_dict['w_list_r'] = w_list_r
        opt_data_dict['q_list_o'] = q_list_o
        opt_data_dict['w_list_o'] = w_list_o
        opt_data_dict['v_list_0'] = v_list_0
        opt_data_dict['v_list_1'] = v_list_1
        opt_data_dict['v_list_2'] = v_list_2
        opt_data_dict['p_list_0'] = p_list_0
        opt_data_dict['p_list_1'] = p_list_1
        opt_data_dict['p_list_2'] = p_list_2
        opt_data_dict['q_list_0'] = q_list_0
        opt_data_dict['q_list_1'] = q_list_1
        opt_data_dict['q_list_2'] = q_list_2
        opt_data_dict['a_delta_list_0'] = a_delta_list_0
        opt_data_dict['a_delta_list_1'] = a_delta_list_1
        opt_data_dict['a_delta_list_2'] = a_delta_list_2
        return opt_data_dict


