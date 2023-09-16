import numpy as np
import matplotlib.pyplot as plt

scale = 1.0  # 指数分布的尺度参数
size = 2000  # 生成随机数的数量

# 生成符合指数分布的随机数
random_values = np.random.exponential(scale=scale, size=size)

# 创建直方图
plt.hist(random_values, bins=300, density=True, color="g")

# 添加标签和标题
plt.xlabel("random")
plt.ylabel("count")
plt.title("exp")

# 显示图形
plt.show()
