import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv("test/rhla13_exp_2.0/K0/V0.csv")

# 提取fitness列的数据
fitness_data = df["Fitness"]


# 绘制曲线图
plt.figure(figsize=(10, 6))  # 设置图表大小（可选）
plt.hist(fitness_data, bins=80, edgecolor="k")  # bins参数控制直方图的柱数
plt.xlabel("Fitness")
plt.ylabel("Count")
plt.title("Fitness Histogram")
plt.grid(True)
plt.show()
