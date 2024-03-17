from scipy.linalg import toeplitz
import numpy as np

# 假设 h 是你想要在Toeplitz矩阵第一行使用的值
h = np.array([0.61541569, 0.4850363, 0.40502704])

# 构造Toeplitz矩阵的第一行
first_row = np.concatenate(([h[0]], h[1:], [0, 0, 0, 0, 0]))

# 构造第一列（这里假设矩阵的其余部分你希望是零）
first_col = np.zeros(len(first_row))  # 确保第一列长度与第一行相同

# 使用 toeplitz 函数创建矩阵
toeplitz_matrix = toeplitz(first_col, first_row)

print(toeplitz_matrix)