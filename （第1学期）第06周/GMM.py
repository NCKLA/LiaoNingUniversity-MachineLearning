import math
import copy
import numpy as np
import matplotlib.pyplot as plt
is_debug = False
# 创建两个正态分布的数据，方差已知并同，均值不同，k是高斯分布的个数，n是样本数
def ini_data(sigma, mu1, mu2, k, n):
    global x
    global mu
    global expectations
    # 创建1*n维的0向量
    x = np.zeros((1, n))
    # 随机生成两个0-1间的随机数
    mu = np.random.random(2)
    # 生成n*k维的0向量
    expectations = np.zeros((n, k))
    # 遍历
    for i in range(0, n):
        # 如果生成一个随机数大于0.5
        if np.random.random(1) > 0.5:
            # 那么x中第0行第i列由第1个高斯分布生成
            x[0, i] = np.random.normal() * sigma + mu1
        # 如果小于0.5
        else:
            # 则由第2个高斯分布生成随机数
            x[0, i] = np.random.normal() * sigma + mu2
    if is_debug:
        print("***")
        print("初始观测值数据x:")
        print(x)
# # 计算E[Zij]隐藏参数
def e_step(sigma, k, n):
    global expectations
    global mu
    global x
    # 对每个样本做遍历
    for i in range(0, n):
        # 初始化新的样本数
        denom = 0
        # 对该样本在每个分布中做遍历
        for j in range(0, k):
            # 累加计算所有样本点在该分布中的数
            denom += math.exp((-1 / (2 * float(sigma ** 2)))) * (float(x[0, i] - mu[j]) ** 2)
        for j in range(0, k):
            # 计算该样本点在该分布中的数
            numer = math.exp((-1 / (2 * float(sigma ** 2)))) * (float(x[0, i] - mu[j]) ** 2)
            # 计算该样本点来自该分布的概率r(i,k)
            expectations[i, j] = numer / denom
    if is_debug:
        print("***")
        print("隐藏变量E(Z)")
        print(expectations)
# # M-step:求最大化E(Zij)时的参数mu
def m_step(k, N):
    global expectations
    global x
    # 遍历每个分布
    for j in range(0, k):
        # 初始化新的样本点的值
        numer = 0
        # 初始化新的样本数量
        denom = 0
        # 对每个新样本做遍历
        for i in range(0, N):
            # 累加所有样本的值
            numer += expectations[i, j] * x[0, i]
            # 累加所有样本的数量
            denom += expectations[i, j]
        # 计算新的均值
        mu[j] = numer / denom


# # 迭代收敛
def run(sigma, mu1, mu2, k, n, iter_num, epsilon):
    # 生成由两个正态分布组成的数据
    ini_data(sigma, mu1, mu2, k, n)
    print("初始化均值：", mu)
    # 根据预先设置的迭代次数循环
    for i in range(iter_num):
        # 将现有的均值赋值给老的均值变量
        old_mu = copy.deepcopy(mu)
        # 计算隐变量的期望
        e_step(sigma, k, n)
        # 求目标函数最大值时的均值
        m_step(k, n)
        print(i, mu)
        # 如果新旧均值的差值小于设定的阀值则终止循环
        if sum(abs(mu - old_mu)) < epsilon:
            break


if __name__ == '__main__':
    run(6, 40, 20, 2, 10000, 10, 0.01)
    plt.hist(x[0, :], 50)
    plt.show()