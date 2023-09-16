import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件加载数据
csv_file = "data.csv"  
df = pd.read_csv(csv_file)

# 提取列数据
fitness = df['Fitness'] 

# 绘制直方图
plt.figure(figsize=(8, 6))  
plt.hist(fitness, bins=100, edgecolor='k', alpha=0.7, label="Fitness")

# 设置图形标题和轴标签
plt.title("Fitness Distribution")
plt.xlabel("Fitness Value")
plt.ylabel("Frequency")

# 显示图例
plt.legend()

# 显示图形
plt.show()
