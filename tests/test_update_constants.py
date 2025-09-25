# tests/test_onramp_demand_param.py
# -*- coding: utf-8 -*-
import sys
import importlib
import types
import pytest


# —— 配置区 ——
# 你的“被测模块”入口（请按你的工程实际修改为可导入的模块名）
# 下面写法会在导入失败时自动跳过测试，不会阻塞 CI。
SIM_MOD = pytest.importorskip("simulation", reason="请确保有 simulation 模块或把名字改成你的模块")
MODULES_TO_RELOAD = [
    "simulation",  # 把依赖 constants 的模块都列上（如有多个：env、controller、algo……）
    # "env",
    # "controller",
    # "algo",
]

# 备选参数组：依次修改 RANDOM_DEMAND_ONRAMP_MIN/MAX
# 你也可以从外部读取，或者用更细的网格
PARAM_SETS = [
    dict(RANDOM_DEMAND_ONRAMP_MIN=300,  RANDOM_DEMAND_ONRAMP_MAX=900),
    dict(RANDOM_DEMAND_ONRAMP_MIN=500,  RANDOM_DEMAND_ONRAMP_MAX=1200),
    dict(RANDOM_DEMAND_ONRAMP_MIN=800,  RANDOM_DEMAND_ONRAMP_MAX=1600),
    dict(RANDOM_DEMAND_ONRAMP_MIN=1200, RANDOM_DEMAND_ONRAMP_MAX=1800),
]


def _reload_modules(module_names):
    """按列表顺序重载模块（若不存在则忽略）。"""
    for name in module_names:
        mod = sys.modules.get(name)
        if isinstance(mod, types.ModuleType):
            importlib.reload(mod)


@pytest.fixture
def patch_onramp_constants(monkeypatch, request):
    """
    在每次测试前：
      1) monkeypatch 修改 constants 中的 ONRAMP MIN/MAX
      2) 重新加载依赖 constants 的模块，确保新值生效
    在测试结束后，monkeypatch 会自动恢复
    """
    from src.config import constants  # 仅导入一次模块对象，后续对属性打补丁
    cfg = request.param

    # 安全检查：min <= max
    assert cfg["RANDOM_DEMAND_ONRAMP_MIN"] <= cfg["RANDOM_DEMAND_ONRAMP_MAX"], \
        "RANDOM_DEMAND_ONRAMP_MIN 必须 <= RANDOM_DEMAND_ONRAMP_MAX"

    # 打补丁到 constants
    monkeypatch.setattr(constants, "RANDOM_DEMAND_ONRAMP_MIN", cfg["RANDOM_DEMAND_ONRAMP_MIN"], raising=True)
    monkeypatch.setattr(constants, "RANDOM_DEMAND_ONRAMP_MAX", cfg["RANDOM_DEMAND_ONRAMP_MAX"], raising=True)

    # 关键：重载所有依赖 constants 的模块
    _reload_modules(MODULES_TO_RELOAD)

    # 可选：把当前配置暴露给测试体使用
    return cfg


@pytest.mark.parametrize("patch_onramp_constants", PARAM_SETS, indirect=True)
def test_simulation_with_varied_onramp(patch_onramp_constants):
    """
    典型断言思路（示例）：
      1) 能顺利跑完一次仿真（基本健壮性）
      2) 产出的指标合理（如无 NaN/inf）
      3) 可选：随着 onramp 上下限升高/降低，对关键指标的单调性做轻量检查
    """
    cfg = patch_onramp_constants

    # === 你项目里的“单次运行/单步或单回合仿真”入口 ===
    # 请按你的模块实际替换；下行仅示例：
    #   - 如果你的入口是 simulation.run_once()
    #   - 或 simulation.run(horizon=...), env.reset()/step() 循环等
    #
    # 要求：被测模块在运行时是从 constants 读取
    out = SIM_MOD.run_once()  # 例如：返回 dict，包含关键指标

    # === 基本断言（示意） ===
    assert out is not None, "仿真输出为空"
    assert isinstance(out, dict), "仿真输出需为 dict（可按你项目实际修改）"
    # 例：关键指标不要为 NaN/inf，且在合理范围：
    for k in ("ttt", "queue", "reward"):
        assert k in out, f"缺少关键输出字段: {k}"
        val = float(out[k])
        assert val == val and val not in (float("inf"), float("-inf")), f"{k} 数值异常: {val}"

    # （可选）将当前用例的 onramp 配置打印出来，便于调试
    print(f"[ONRAMP] min={cfg['RANDOM_DEMAND_ONRAMP_MIN']}, max={cfg['RANDOM_DEMAND_ONRAMP_MAX']} -> out={out}")