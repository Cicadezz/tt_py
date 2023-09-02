'''
单目标多变量函数 - 无约束 - （可画立体图）
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

DNA_SIZE = 24  # 一个实数DNA的长度
POP_SIZE = 200 # 种族数量
CROSSOVER_RATE = 0.8 # 交叉概率
MUTATION_RATE = 0.005 # 变异概率
N_GENERATIONS = 50 # 迭代次数
X_BOUND = [-3, 3]  # x取值范围
Y_BOUND = [-3, 3]  # y取值范围


# 适应度的目标函数
def F(x, y):
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)


def plot_3d(ax):
    # np.linspace()生成等距数组
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()


# 得到一个种群
# 设置DNA_SIZE=24（一个实数DNA长度），两个实数x , y
# x,y一共用48位二进制编码，我同时将x和y编码到同一个48位的二进制串里，每一个变量用24位表示，奇数24列为x的编码表示，偶数24列为y的编码表示
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 1::2]  # 奇数列表示X，第一维，从第二个元素起步长为2取元素
    y_pop = pop[:, ::2]  # 偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    '''
    将【0，1】区间的数映射到需要的区间：
    x_ = num * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0] #映射为x范围内的数
    y_ = num * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0] #映射为y范围内的数
    '''
    # [::-1]取从后向前的元素
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


# 求最小值适应度函数
def get_fitness(pop):
    x, y = translateDNA(pop)
    # 将可能解带入函数F中得到的预测值
    pred = F(x, y)
    # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度
    return (pred - np.min(pred)) + 1e-3


# 根据适应度选择
# 适应度越高，被选择的机会越高；适应度越低，被选择的机会越低
def select(pop, fitness):  # nature selection wrt pop's fitness
    # p描述了从np.arange(POP_SIZE)里选择每一个元素的概率，概率越高越容易被选中
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / (fitness.sum()))
    return pop[idx]


# 交叉
def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)
    return new_pop


# 变异
def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax)

    '迭代交叉算法'
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        # 特征值散点图，x,y,点的大小,颜色,标记
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o')
        # locals()以字典类型返回当前位置的所有局部变量
        if sca in locals():
            sca.remove()
        plt.show()
        # 动态绘制窗口
        plt.pause(0.1)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE)) # 种族通过交叉变异产生后代
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        fitness = get_fitness(pop) # 对种族的每个个体进行评估
        pop = select(pop, fitness)  # 选择生成新的种群

    print_info(pop)
    plt.ioff() # 画动态图
    plot_3d(ax)
    print("over")
