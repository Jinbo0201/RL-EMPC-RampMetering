from src.mpc.modelPyomo import *
from src.simulation.input import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = mpc_model()
    # 打印模型
    model.pprint(verbose=True)
    input = Input()
    demand_o, demand_r, density_e = input.get_input(1000, NP)
    print(demand_o)
    for k in range(NP):
        model.p_d_o[k] = demand_o[k]
        model.p_d_r[k] = demand_r[k]
        model.p_p_e[k] = density_e[k]
    for id_segment in range(NUM_SEGMENT):
        model.p_q[id_segment] = 0
        model.p_p[id_segment] = 50
        model.p_v[id_segment] = V_FREE
    model.p_w_o = 0
    model.p_w_r = 0
    solver = pyo.SolverFactory('ipopt')
    # solver = pyo.SolverFactory('glpk')
    results = solver.solve(model)
    print("Optimization status:", results.solver.status, results.solver.termination_condition)
    print("Optimal objective value:", pyo.value(model.obj))
    print("Optimal variable values:")
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
    # a_a_list = []
    # a_b_list = []
    for k in range(NP):
        # print(k, "w =", pyo.value(model.x_w_o[k]), pyo.value(model.x_w_r[k]))
        # print(k, "r =", pyo.value(model.x_q_o[k]), pyo.value(model.x_r_r[k]))
        r_list.append(pyo.value(model.x_r_r[change_NP2NC_r(k)]))
        w_list_r.append(pyo.value(model.x_w_r[k]))
        q_list_o.append(pyo.value(model.x_q_o[change_NP2NC_r(k)]))
        w_list_o.append(pyo.value(model.x_w_o[k]))
        v_list_0.append(pyo.value(model.x_v[0, k + 1]))
        v_list_1.append(pyo.value(model.x_v[1, k + 1]))
        v_list_2.append(pyo.value(model.x_v[2, k + 1]))
        p_list_0.append(pyo.value(model.x_p[0, k + 1]))
        p_list_1.append(pyo.value(model.x_p[1, k + 1]))
        p_list_2.append(pyo.value(model.x_p[2, k + 1]))
        q_list_0.append(pyo.value(model.x_q[0, k + 1]))
        q_list_1.append(pyo.value(model.x_q[1, k + 1]))
        q_list_2.append(pyo.value(model.x_q[2, k + 1]))
        a_delta_list_0.append(pyo.value(model.a_delta[0, k]))
        a_delta_list_1.append(pyo.value(model.a_delta[1, k]))
        a_delta_list_2.append(pyo.value(model.a_delta[2, k]))
        # a_a_list.append(pyo.value(model.a_a[k]))
        # a_b_list.append(pyo.value(model.a_b[k]))
    demand_o_list = []
    for k in range(NP):
        demand_o_list.append(pyo.value(model.p_d_o[k]) + w_list_o[k] / T)
    demand_r_list = []
    for k in range(NP):
        demand_r_list.append(pyo.value(model.p_d_r[k]) + w_list_r[k] / T)
    print(a_delta_list_0)
    print(a_delta_list_1)
    print(a_delta_list_2)
    fig, axes = plt.subplots(3, 3)
    axes[0, 0].plot(r_list[:-M], label='r')
    axes[0, 0].plot(q_list_o[:-M], label='o')
    axes[0, 0].set_title('r_list')
    axes[0, 0].legend()
    axes[0, 1].plot(w_list_r[:-M], label='r')
    axes[0, 1].plot(w_list_o[:-M], label='o')
    axes[0, 1].set_title('queue_list')
    axes[0, 1].legend()
    axes[0, 2].plot(demand_r_list[:-M], label='d_r')
    axes[0, 2].plot(demand_o_list[:-M], label='d_o')
    axes[0, 2].legend()
    # axes[1, 0].plot(a_delta_list_0, label='0')
    # axes[1, 0].plot(a_delta_list_1, label='1')
    # axes[1, 0].plot(a_delta_list_2, label='2')
    axes[1, 0].plot(a_delta_list_0[:-M], label='0')
    axes[1, 0].plot(a_delta_list_1[:-M], label='1')
    axes[1, 0].plot(a_delta_list_2[:-M], label='2')
    axes[1, 0].set_title('delta_list')
    axes[1, 0].set_ylim([-1, 2])
    axes[1, 0].legend()
    # axes[1, 1].plot(a_a_list[:-M], label='aa')
    # axes[1, 1].plot(a_b_list[:-M], label='ab')
    # axes[1, 1].set_title('a_list')
    # axes[1, 1].set_ylim([-1, 2])
    # axes[1, 1].legend()
    axes[2, 0].plot(q_list_0[:-M], label='0')
    axes[2, 0].plot(q_list_1[:-M], label='1')
    axes[2, 0].plot(q_list_2[:-M], label='2')
    # axes[2, 0].plot(q_list_o[:-M], label='o')
    # axes[2, 0].plot(demand_o_list[:-M], label='d')
    axes[2, 0].set_title('flow_list')
    axes[2, 0].legend()
    axes[2, 1].plot(p_list_0[:-M], label='0')
    axes[2, 1].plot(p_list_1[:-M], label='1')
    axes[2, 1].plot(p_list_2[:-M], label='2')
    axes[2, 1].set_title('density_list')
    axes[2, 1].legend()
    axes[2, 2].plot(v_list_0[:-M], label='0')
    axes[2, 2].plot(v_list_1[:-M], label='1')
    axes[2, 2].plot(v_list_2[:-M], label='2')
    axes[2, 2].set_title('v_list')
    axes[2, 2].legend()
    plt.tight_layout()
    plt.show()
