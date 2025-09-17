import pyomo.environ as pyo
import math
from src.config.constants import *

def mpc_model():
    model = pyo.ConcreteModel()
    # 状态变量
    model.x_p = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
    model.x_q = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
    model.x_v = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
    model.x_w_o = pyo.Var(range(NP + 1), domain=pyo.NonNegativeReals)
    model.x_w_r = pyo.Var(range(NP + 1), domain=pyo.NonNegativeReals)
    model.x_p_e = pyo.Var(range(NP), domain=pyo.NonNegativeReals)
    model.x_v_o = pyo.Var(range(NP), domain=pyo.NonNegativeReals)
    model.x_w_o_over = pyo.Var(range(NP + 1), domain=pyo.NonNegativeReals)
    model.x_w_r_over = pyo.Var(range(NP + 1), domain=pyo.NonNegativeReals)
    # 辅助变量
    model.a_delta = pyo.Var(range(NUM_SEGMENT), range(NP), domain=pyo.Binary, initialize=1)
    # model.a_a = pyo.Var(range(NP), domain=pyo.Binary, initialize=1)
    # model.a_b = pyo.Var(range(NP), domain=pyo.Binary, initialize=1)
    # model.a_q_o = pyo.Var(range(NP), domain=pyo.NonNegativeReals)
    # 决策变量
    model.x_q_o = pyo.Var(range(NP_C), domain=pyo.NonNegativeReals)
    model.x_r_r = pyo.Var(range(NP_C), domain=pyo.NonNegativeReals)
    # 参数
    model.p_d_o = pyo.Param(range(NP), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
    model.p_d_r = pyo.Param(range(NP), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
    model.p_p_e = pyo.Param(range(NP), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
    # 参数-初始状态
    model.p_q = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
    model.p_p = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
    model.p_v = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
    model.p_w_o = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
    model.p_w_r = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
    # 设定目标函数
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    # 初始化参数
    model.c_init_x_p = pyo.Constraint(range(NUM_SEGMENT), rule=constraint_init_x_p)
    model.c_init_x_q = pyo.Constraint(range(NUM_SEGMENT), rule=constraint_init_x_q)
    model.c_init_x_v = pyo.Constraint(range(NUM_SEGMENT), rule=constraint_init_x_v)
    model.c_init_x_w_o = pyo.Constraint(rule=constraint_init_x_w_o)
    model.c_init_x_w_r = pyo.Constraint(rule=constraint_init_x_w_r)
    # 状态量计算等式
    model.c_cal_q = pyo.Constraint(range(NUM_SEGMENT), range(1, NP + 1), rule=constraint_cal_q)
    model.c_cal_p = pyo.Constraint(range(NUM_SEGMENT), range(1, NP + 1), rule=constraint_cal_p)
    model.c_cal_v = pyo.Constraint(range(NUM_SEGMENT), range(1, NP + 1), rule=constraint_cal_v)
    model.c_cal_w_o = pyo.Constraint(range(1, NP + 1), rule=constraint_cal_w_o)
    model.c_cal_w_r = pyo.Constraint(range(1, NP + 1), rule=constraint_cal_w_r)
    # 状态量计算不等式
    model.c_cal_r_o = pyo.Constraint(range(NP), rule=constraint_cal_r_o)
    model.c_cal_r_o_c = pyo.Constraint(range(NP), rule=constraint_cal_r_o_c_max)
    model.c_cal_r_o_capacity = pyo.Constraint(range(NP), rule=constraint_cal_r_o_c)
    model.c_cal_r_r = pyo.Constraint(range(NP), rule=constraint_cal_r_r)
    model.c_cal_r_r_c = pyo.Constraint(range(NP), rule=constraint_cal_r_r_c_max)
    model.c_cal_r_r_capacity = pyo.Constraint(range(NP), rule=constraint_cal_r_r_c)
    # 边界量计算等式
    model.c_cal_x_v_o = pyo.Constraint(range(NP), rule=constraint_cal_x_v_o)
    model.c_cal_x_p_e = pyo.Constraint(range(NP), rule=constraint_cal_x_p_e)
    # 辅助变量约束条件
    model.c_aux_delta_1 = pyo.Constraint(range(NUM_SEGMENT), range(NP), rule=constraint_aux_delta_1)
    model.c_aux_delta_2 = pyo.Constraint(range(NUM_SEGMENT), range(NP), rule=constraint_aux_delta_2)
    # 超长队伍约束条件
    model.c_cal_x_w_o_over = pyo.Constraint(range(1, NP + 1), rule=constraint_cal_x_w_o_over)
    model.c_cal_x_w_r_over = pyo.Constraint(range(1, NP + 1), rule=constraint_cal_x_w_r_over)
    # 返回模型
    return model
# 将控制变量序号从NP转换到NP_C
def change_NP2NC_r(id_NP):
    return math.floor(id_NP / M)
# 目标函数
def obj_rule(model):
    return T * L * LAMBDA * sum(model.x_p[id_segment, k] for id_segment in range(NUM_SEGMENT) for k in
                                range(1, NP + 1)) \
        + T * sum(model.x_w_o[k] for k in range(1, NP + 1)) \
        + T * sum(model.x_w_r[k] for k in range(1, NP + 1)) \
        + XI_W * sum(model.x_w_o_over[k] for k in range(1, NP + 1)) \
        + XI_W * sum(model.x_w_r_over[k] for k in range(1, NP + 1))
        # + XI * sum(((model.x_r_r[k + 1] - model.x_r_r[k]) / CAPACITY_ONRAMP) ^ 2 for k in range(NP_C - 1)) \
        # + XI * sum(((model.x_q_o[k + 1] - model.x_q_o[k]) / CAPACITY_ORIGIN) ^ 2 for k in range(NP_C - 1)) \
        # 约束条件
def constraint_init_x_p(model, id_segment):
    return model.x_p[id_segment, 0] == model.p_p[id_segment]
def constraint_init_x_q(model, id_segment):
    return model.x_q[id_segment, 0] == model.p_q[id_segment]
def constraint_init_x_v(model, id_segment):
    return model.x_v[id_segment, 0] == model.p_v[id_segment]
def constraint_init_x_w_o(model):
    return model.x_w_o[0] == model.p_w_o
def constraint_init_x_w_r(model):
    return model.x_w_r[0] == model.p_w_r
def constraint_cal_q(model, id_segment, k):
    return model.x_q[id_segment, k] == LAMBDA * model.x_p[id_segment, k] * model.x_v[id_segment, k]
def constraint_cal_p(model, id_segment, k):
    if id_segment == 0:
        expr = model.x_p[id_segment, k] == model.x_p[id_segment, k - 1] + T / (L * LAMBDA) * (
                model.x_q_o[change_NP2NC_r(k - 1)] - model.x_q[id_segment, k - 1])
    elif id_segment == ID_ONRAMP:
        expr = model.x_p[id_segment, k] == model.x_p[id_segment, k - 1] + T / (L * LAMBDA) * (
                model.x_q[id_segment - 1, k - 1] - model.x_q[id_segment, k - 1] + model.x_r_r[change_NP2NC_r(k - 1)])
    else:
        expr = model.x_p[id_segment, k] == model.x_p[id_segment, k - 1] + T / (L * LAMBDA) * (
                model.x_q[id_segment - 1, k - 1] - model.x_q[id_segment, k - 1])
    return expr
def constraint_cal_v(model, id_segment, k):
    if id_segment == 0:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + T / TAU * (
                model.a_delta[id_segment, k - 1] * (BETA_1 + ALPHA_1 * model.x_p[id_segment, k - 1]) + (
                1 - model.a_delta[id_segment, k - 1]) * (BETA_2 + ALPHA_2 * model.x_p[id_segment, k - 1]) - model.x_v[
                    id_segment, k - 1]) + T / L * model.x_v[id_segment, k - 1] * (
                       model.x_v_o[k - 1] - model.x_v[id_segment, k - 1]) - NU * T / (
                       TAU * L) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA)
    elif id_segment == NUM_SEGMENT - 1:  # 最后路段和匝道连接
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + T / TAU * (
                model.a_delta[id_segment, k - 1] * (BETA_1 + ALPHA_1 * model.x_p[id_segment, k - 1]) + (
                1 - model.a_delta[id_segment, k - 1]) * (BETA_2 + ALPHA_2 * model.x_p[id_segment, k - 1]) - model.x_v[
                    id_segment, k - 1]) + T / L * model.x_v[id_segment, k - 1] * (
                       model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - NU * T / (
                       TAU * L) * (model.x_p_e[k - 1] - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA) - MU * T / (L * LAMBDA) * model.x_v[
                   id_segment, k - 1] / (model.x_p[id_segment, k - 1] + KAPPA) * model.x_r_r[change_NP2NC_r(k - 1)]
    else:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + T / TAU * (
                model.a_delta[id_segment, k - 1] * (BETA_1 + ALPHA_1 * model.x_p[id_segment, k - 1]) + (
                1 - model.a_delta[id_segment, k - 1]) * (BETA_2 + ALPHA_2 * model.x_p[id_segment, k - 1]) - model.x_v[
                    id_segment, k - 1]) + T / L * model.x_v[id_segment, k - 1] * (
                       model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - NU * T / (
                       TAU * L) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA)
    return expr
def constraint_aux_delta_1(model, id_segment, k):
    return model.x_p[id_segment, k] - PWA_MID <= PWA_MAX * (1 - model.a_delta[id_segment, k])
def constraint_aux_delta_2(model, id_segment, k):
    return model.x_p[id_segment, k] - PWA_MID >= PWA_EPSILON + (PWA_MIN - PWA_EPSILON) * model.a_delta[id_segment, k]
def constraint_cal_w_o(model, k):
    expr = model.x_w_o[k] == model.x_w_o[k - 1] + T * (model.p_d_o[k - 1] - model.x_q_o[change_NP2NC_r(k - 1)])
    return expr
def constraint_cal_w_r(model, k):
    expr = model.x_w_r[k] == model.x_w_r[k - 1] + T * (model.p_d_r[k - 1] - model.x_r_r[change_NP2NC_r(k - 1)])
    return expr
def constraint_cal_r_o(model, k):
    return model.x_q_o[change_NP2NC_r(k)] <= model.p_d_o[k] + model.x_w_o[k] / T
def constraint_cal_r_o_c_max(model, k):
    return model.x_q_o[change_NP2NC_r(k)] <= CAPACITY_ORIGIN
def constraint_cal_r_o_c(model, k):
    return model.x_q_o[change_NP2NC_r(k)] <= CAPACITY_ORIGIN * (DENSITY_MAX - model.x_p[0, k]) / (
            DENSITY_MAX - DENSITY_CRIT)
def constraint_cal_r_r(model, k):
    return model.x_r_r[change_NP2NC_r(k)] <= model.p_d_r[k] + model.x_w_r[k] / T
def constraint_cal_r_r_c(model, k):
    return model.x_r_r[change_NP2NC_r(k)] <= CAPACITY_ONRAMP * (DENSITY_MAX - model.x_p[ID_ONRAMP, k]) / (
            DENSITY_MAX - DENSITY_CRIT)
def constraint_cal_r_r_c_max(model, k):
    return model.x_r_r[change_NP2NC_r(k)] <= CAPACITY_ONRAMP
def constraint_cal_x_v_o(model, k):
    return model.x_v_o[k] == model.x_v[0, k]
def constraint_cal_x_p_e(model, k):
    return model.x_p_e[k] == model.p_p_e[k]
def constraint_cal_x_w_o_over(model, k):
    return model.x_w_o[k] - QUEUE_MAX <= model.x_w_o_over[k]
def constraint_cal_x_w_r_over(model, k):
    return model.x_w_r[k] - QUEUE_MAX <= model.x_w_r_over[k]
