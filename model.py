import numpy as np

def model():
    total = [[0.15, 0.85, 1],
            [0.136, 0.864, 1],
            [0.129, 0.871, 1],
            [0.102, 0.898, 1],
            [0.072, 0.928, 1],
            [0.123, 0.877, 1],
            [0.115, 0.885, 1],
            [0.125, 0.875, 1],
            [0.11, 0.89, 1],
            [0.064, 0.936, 1],
            [0.115, 0.885, 1],
            [0.124, 0.876, 1],
            [0.188, 0.812, 1],
            [0.079, 0.921, 1],
            [0.055, 0.945, 1],
            [0.066, 0.934, 1],
            [0.158, 0.842, 1],
            [0.175, 0.825, 1],
            [0.262, 0.738, 0],
            [0.235, 0.765, 0],
            [0.354, 0.646, 0],
            [0.345, 0.655, 0],
            [0.308, 0.692, 0],
            [0.325, 0.675, 0],
            [0.377, 0.623, 0],
            [0.396, 0.604, 0],
            [0.405, 0.595, 0],
            [0.318, 0.682, 0],
            [0.197, 0.803, 0],
            [0.301, 0.699, 0]]
    # 数据集
    data = np.array(total)

    # 分离特征和标签
    X = data[:, :2]
    y = data[:, 2]

    # 计算正例和反例的均值向量
    mu_0 = np.mean(X[y == 0], axis=0)
    mu_1 = np.mean(X[y == 1], axis=0)

    # 计算类内散度矩阵
    S_w = np.zeros((2, 2))
    for i in range(len(X)):
     if y[i] == 0:
            S_w += np.outer(X[i] - mu_0, X[i] - mu_0)
     else:
            S_w += np.outer(X[i] - mu_1, X[i] - mu_1)

    # 计算最优投影方向
    w_prime = np.linalg.inv(S_w).dot(mu_0 - mu_1)

    # 计算LDA分割线（决策边界），决策边界垂直于投影向量w_prime，且通过两个类别均值的中点
    mid_point = (mu_0 + mu_1) / 2
    slope = w_prime[1] / w_prime[0]

    # 计算垂直于投影向量的斜率
    slope_perpendicular = -1 / slope

    # 计算截距
    intercept_perpendicular = mid_point[1] - slope_perpendicular * mid_point[0]


    # 输出LDA分割线的方程
    # print(f'LDA分割线的方程为：y = {slope_perpendicular}x + {intercept_perpendicular}')
    return slope_perpendicular, intercept_perpendicular