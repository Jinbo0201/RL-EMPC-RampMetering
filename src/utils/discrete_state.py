
def discretize_state(state):

    discretized_state = []

    for i, element in enumerate(state):
        if i == 0:
            transformed_state = int(element // 10)
        elif i == 1:
            transformed_state = int(element // 10)
        elif i == 2:
            transformed_state = int(element // 10)
            if transformed_state > 9:
                transformed_state = 9
            elif transformed_state < 0:
                transformed_state = 0
        else:
            transformed_state = int(element // 10)
            if transformed_state > 9:
                transformed_state = 9
            elif transformed_state < 0:
                transformed_state = 0
        discretized_state.append(transformed_state)

    return tuple(discretized_state)

# 定义辅助函数，将连续状态转换为离散状态
# 状态量依次为['density'][1], ['density'][2], ['queue_length_origin'], ['queue_length_onramp']
def discretize_fewerstate(state):
    discretized_state = []

    for i, element in enumerate(state):
        if i == 0:
            transformed_state = int(element // 10)
        elif i == 1:
            continue
        elif i == 2:
            continue
        else:
            transformed_state = int(element // 10)
            if transformed_state > 9:
                transformed_state = 9
            elif transformed_state < 0:
                transformed_state = 0
        discretized_state.append(transformed_state)

    return tuple(discretized_state)
