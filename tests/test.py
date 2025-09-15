import matplotlib.pyplot as plt
import numpy as np

# 创建一个空的图表
plt.figure()

# 定义x轴的范围
x = np.linspace(0, 2 * np.pi, 100)

# 持续绘制图形
for i in range(10):
    # 生成随机的y轴数据
    y = np.sin(x + i * np.pi / 5)

    # 清除之前的图形
    # plt.clf()

    # 绘制新的图形
    plt.plot(x, y)

    # 设置标题和标签
    plt.title('Continuous Plot')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.pause(0.5)  # 暂停0.5秒

# 关闭图表
plt.close()