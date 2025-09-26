from src.config.constants import *



# 定义辅助函数，将连续状态转换为离散状态
# 状态量依次为['density'][1], ['density'][2], ['queue_length_origin'], ['queue_length_onramp']
def cal_obser2state_ql(obser):

    state = []

    state.append(int(obser['density'][1] // 10))

    transformed_state = int(obser['queue_length_onramp'][0] // 20)
    if transformed_state > 9:
        transformed_state = 9
    elif transformed_state < 0:
        transformed_state = 0
    state.append(transformed_state)

    return tuple(state)


def cal_obser2state(obser):

    state = []

    state.append(obser['density'][1] / DENSITY_MAX )
    state.append(obser['queue_length_onramp'][0] / QUEUE_LENGTH_ONRAMP_MAX)

    return tuple(state)

