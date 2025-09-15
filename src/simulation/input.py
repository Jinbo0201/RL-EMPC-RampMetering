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


class Input(object):

    def __init__(self):
        # origin demand
        self.demand_o = []
        # onramp demand
        self.demand_r = []
        # end density
        self.density_e = []

        self._create_demand_o()
        self._create_demand_r()
        self._create_density_e()

    # get k-th timestep demand
    def get_input(self, k, length):
        return self.demand_o[k:k + length], self.demand_r[k:k + length], self.density_e[k:k + length]

    def _create_demand_o(self):
        delta = RANDOM_DEMAND_ORIGIN_CYCLE / 4
        k = 0
        while k * DELTA_T <= LENGTH_H:
            value = (k * DELTA_T) % RANDOM_DEMAND_ORIGIN_CYCLE
            if value < delta:
                demand = RANDOM_DEMAND_ORIGIN_MIN
            elif value < 2 * delta:
                demand = RANDOM_DEMAND_ORIGIN_MIN + (RANDOM_DEMAND_ORIGIN_MAX - RANDOM_DEMAND_ORIGIN_MIN) / delta * (
                            value - delta)
            elif value < 3 * delta:
                demand = RANDOM_DEMAND_ORIGIN_MAX
            else:
                demand = RANDOM_DEMAND_ORIGIN_MAX - (RANDOM_DEMAND_ORIGIN_MAX - RANDOM_DEMAND_ORIGIN_MIN) / delta * (
                            value - 3 * delta)
            self.demand_o.append(demand)
            k += 1

    def _create_demand_r(self):
        delta = RANDOM_DEMAND_ONRAMP_CYCLE / 4
        k = 0
        while k * DELTA_T <= LENGTH_H:
            value = (k * DELTA_T) % RANDOM_DEMAND_ONRAMP_CYCLE
            if value < delta:
                demand = RANDOM_DEMAND_ONRAMP_MIN
            elif value < 2 * delta:
                demand = RANDOM_DEMAND_ONRAMP_MIN + (RANDOM_DEMAND_ONRAMP_MAX - RANDOM_DEMAND_ONRAMP_MIN) / delta * (
                        value - delta)
            elif value < 3 * delta:
                demand = RANDOM_DEMAND_ONRAMP_MAX
            else:
                demand = RANDOM_DEMAND_ONRAMP_MAX - (RANDOM_DEMAND_ONRAMP_MAX - RANDOM_DEMAND_ONRAMP_MIN) / delta * (
                        value - 3 * delta)
            self.demand_r.append(demand)
            k += 1

    def _create_density_e(self):
        delta = RANDOM_DOWNSTREAM_DENSITY_CYCLE / 4
        k = 0
        while k * DELTA_T <= LENGTH_H:
            value = (k * DELTA_T) % RANDOM_DOWNSTREAM_DENSITY_CYCLE
            if value < delta:
                demand = RANDOM_DOWNSTREAM_DENSITY_MIN
            elif value < 2 * delta:
                demand = RANDOM_DOWNSTREAM_DENSITY_MIN + (
                            RANDOM_DOWNSTREAM_DENSITY_MAX - RANDOM_DOWNSTREAM_DENSITY_MIN) / delta * (
                                 value - delta)
            elif value < 3 * delta:
                demand = RANDOM_DOWNSTREAM_DENSITY_MAX
            else:
                demand = RANDOM_DOWNSTREAM_DENSITY_MAX - (
                            RANDOM_DOWNSTREAM_DENSITY_MAX - RANDOM_DOWNSTREAM_DENSITY_MIN) / delta * (
                                 value - 3 * delta)
            self.density_e.append(demand)
            k += 1
