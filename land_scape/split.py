import pandas as pd

# 打开文本文件并逐行读取
with open("new_exp_zipf_data/rhla1_6_result/result.txt", "r") as file:
    lines = file.readlines()

# 初始化一个空列表，用于存储数据
data = []

# 遍历每一行
for line in lines:
    # 检查行是否包含冒号
    if ":" in line:
        # 使用冒号拆分行
        parts = line.split(":")
        # 提取冒号前的文字和冒号后的数字部分并去除前后空格
        key = parts[0].strip()
        value = float(parts[1].strip())
        # 将数据添加到列表中
        data.append([key, value])

# 创建一个 Pandas DataFrame
df = pd.DataFrame(data, columns=["Description", "Value"])

# 保存 DataFrame 到 Excel 文件
df.to_excel("new_exp_zipf_data/rhla1_6_result/result.xlsx", index=False)

print("Data saved to output.xlsx")
