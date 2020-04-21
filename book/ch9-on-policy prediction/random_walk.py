#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# states except for terminal states
N_STATES = 1000

# all states
STATES = np.arange(1, N_STATES + 1)

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [0, N_STATES + 1]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100

def compute_true_value():
    # true state value, just a promising guess
    true_value = np.arange(-1001, 1003, 2) / 1001.0

    # Dynamic programming to find the true state values, based on the promising guess above
    # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states
    while True:
        old_value = np.copy(true_value)
        for state in STATES:#对每个状态，其价值等于每个可能动作的概率×该动作可能到的状态×(reward+gama×可能到的状态的价值)
            true_value[state] = 0
            for action in ACTIONS:
                for step in range(1, STEP_RANGE + 1):
                    step *= action
                    next_state = state + step
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # asynchronous update for faster convergence 总共200个动作，所以概率是1/200
                    true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
        error = np.sum(np.abs(old_value - true_value))#前后两次价值的误差
        if error < 1e-2:
            break
    # correct the state value for terminal states to 0
    true_value[0] = true_value[-1] = 0

    return true_value

# take an @action at @state, return new state and reward for this transition
def step(state, action):
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action  #乘以+-1变成向左向右
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward

# get an action, following random policy
def get_action():
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1

# a wrapper class for aggregation value function
class ValueFunction:
    # @num_of_groups: # of aggregations
    def __init__(self, num_of_groups):
        self.num_of_groups = num_of_groups
        self.group_size = N_STATES // num_of_groups #先做除法，然后向下取整

        # thetas
        self.params = np.zeros(num_of_groups)   #w的长度和组数相同

    # get the value of @state
    def value(self, state):
        if state in END_STATES:
            return 0
        group_index = (state - 1) // self.group_size    #对组尺寸求模
        return self.params[group_index] #直接按索引找价值

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    #状态聚合法，w就是对应各组的价值，偏导向量就是状态所属组为1，其他组为0的向量，所以只有对应组的w变了
    def update(self, delta, state):
        group_index = (state - 1) // self.group_size
        self.params[group_index] += delta

# a wrapper class for tile coding value function
class TilingsValueFunction:
    # @num_of_tilings: # of tilings
    # @tileWidth: each tiling has several tiles, this parameter specifies the width of each tile
    # @tilingOffset: specifies how tilings are put together
    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        self.numOfTilings = numOfTilings
        self.tileWidth = tileWidth
        self.tilingOffset = tilingOffset

        # To make sure that each sate is covered by same number of tiles,
        # we need one more tile for each tiling #多了一个网格以覆盖所有状态
        self.tilingSize = N_STATES // tileWidth + 1 #有点像神经网络的卷积

        # weight for each tile
        self.params = np.zeros((self.numOfTilings, self.tilingSize))

        # For performance, only track the starting position for each tiling
        # As we have one more tile for each tiling, the starting position will be negative
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)

    # get the value of @state
    def value(self, state):
        stateValue = 0.0
        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):#每个状态，在每个tiling中都能找到自己对应的网格
            # find the active tile in current tiling    #先看属于哪个tiling，再看属于该tiling的哪个tile
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth   #减去开始划分的位置，将状态位置转到以0为原点的坐标系
            stateValue += self.params[tilingIndex, tileIndex]#对应tiling对应tile的参数，就是状态值，和状态聚合是类似的，不过是更多类的有重叠区域的状态聚合
        return stateValue   #状态值是各个tiling之和

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):

        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling (tile)
        delta /= self.numOfTilings#在算delta的时候，是reward-各个tiling的状态值之和，但下面更新的时候是对每个tiling进行更新

        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            self.params[tilingIndex, tileIndex] += delta    #对每个tiling，都要更新与该状态对应的tile的价值

# a wrapper class for polynomial / Fourier -based value function
POLYNOMIAL_BASES = 0
FOURIER_BASES = 1
class BasesValueFunction:
    # @order: bases, each function also has one more constant parameter (called bias in machine learning)
    # @type: polynomial bases or Fourier bases
    def __init__(self, order, type):
        self.order = order
        self.weights = np.zeros(order + 1)

        # set up bases function
        self.bases = []
        if type == POLYNOMIAL_BASES:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: pow(s, i)) #给定s，i（就等于i),返回s^i
        elif type == FOURIER_BASES:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))#返回cos(i*pi*s)

    # get the value of @state
    def value(self, state):
        # map the state space into [0, 1]   #根据状态在所有状态中的位置，将状态转化为0->1的实数，但不是所有的问题中状态都是这样有顺序和位置关系的啊
        state /= float(N_STATES)    #而且书中也是傅里叶级数的说了s要在0->1，多项式的没说
        # get the feature vector   该问题状态对应的状态向量是一维的，就是顺序s #特征向量就是1 s^1 s^2 .. s^5
        feature = np.asarray([func(state) for func in self.bases])  #bases里面有6个lambda函数，循环取出来，带入state，再倒回lambda定义的地方求出来
        return np.dot(self.weights, feature)    #线性近似方法 v(s)=w*x(s)

    def update(self, delta, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get derivative value  #偏导向量就是特征向量 这里可以把特征向量作为公共的量，就不用value函数和update函数都要算了
        derivative_value = np.asarray([func(state) for func in self.bases])
        self.weights += delta * derivative_value

# gradient Monte Carlo algorithm
# @value_function: an instance of class ValueFunction
# @alpha: step size
# @distribution: array to store the distribution statistics
def gradient_monte_carlo(value_function, alpha, distribution=None):
    state = START_STATE
    trajectory = [state]

    # 不打折 gamma = 1, so return is just the same as the latest reward
    reward = 0.0
    while state not in END_STATES:
        action = get_action()   #prediction问题，选取动作的策略一直是固定的
        next_state, reward = step(state, action)    #reward一直都是0，可以到最后根据状态是0或者1来判断是+1还是-1
        trajectory.append(next_state)   #记录状态序列
        state = next_state

    # Gradient update for each state in this trajectory  monte carlo知道episode结束才会更新价值
    for state in trajectory[:-1]:
        delta = alpha * (reward - value_function.value(state))
        value_function.update(delta, state)
        if distribution is not None:    #对应状态次数加1
            distribution[state] += 1

# semi-gradient n-step TD algorithm
# @valueFunction: an instance of class ValueFunction
# @n: # of steps
# @alpha: step size
def semi_gradient_temporal_difference(value_function, n, alpha):
    # initial starting state
    state = START_STATE

    # arrays to store states and rewards for an episode
    # space isn't a major consideration, so I didn't use the mod trick
    states = [state]
    rewards = [0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        # go to next time step
        time += 1

        if time < T:    #当time>T时，说明已经终止了，但是还不能跳出，还要继续更新T-n后面的状态
            # choose an action randomly
            action = get_action()
            next_state, reward = step(state, action)

            # store new state and new reward
            states.append(next_state)   #n步自举法，像Monte Carlo一样，也要保存状态序列，但不用episode结束后才更新状态
            rewards.append(reward)

            if next_state in END_STATES:
                T = time    #记录终止时间步

        # get the time of the state to update
        update_time = time - n  #当时间步超过n之后开始更新状态
        if update_time >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += rewards[t]   #这个例子reward只有终止状态不是0，没什么好累加的
            # add state value to the return #当后n步超过了终止时间步时，就没有状态价值可加，就只是累加后面几个reward
            if update_time + n <= T:#被更新的时间步的target值=后n-1个reward和+后第n个状态的价值估计值
                returns += value_function.value(states[update_time + n])
            state_to_update = states[update_time]#取出与被更新的时间步对应的状态
            # update the value function
            if not state_to_update in END_STATES:#由于是状态聚合，偏导向量仍然是只有当前状态所属组对应的分量为1其他分量为0的向量
                delta = alpha * (returns - value_function.value(state_to_update))
                value_function.update(delta, state_to_update)
        if update_time == T - 1:    #当被更新状态也到了最后一个状态，才跳出
            break
        state = next_state

# Figure 9.1, gradient Monte Carlo algorithm
def figure_9_1(true_value):
    episodes = int(1e5)
    alpha = 2e-5

    # we have 10 aggregations in this example, each has 100 states
    value_function = ValueFunction(10)
    distribution = np.zeros(N_STATES + 2)
    for ep in tqdm(range(episodes)):
        gradient_monte_carlo(value_function, alpha, distribution)

    distribution /= np.sum(distribution)
    state_values = [value_function.value(i) for i in STATES]    #不要终止状态的

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(STATES, state_values, label='Approximate MC value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(STATES, distribution[1: -1], label='State distribution')
    plt.xlabel('State')
    plt.ylabel('Distribution')
    plt.legend()

    plt.savefig('figure_9_1.png')
    plt.close()

# semi-gradient TD on 1000-state random walk
def figure_9_2_left(true_value):
    episodes = int(1e5)
    alpha = 2e-4
    value_function = ValueFunction(10)
    for ep in tqdm(range(episodes)):
        semi_gradient_temporal_difference(value_function, 1, alpha)

    stateValues = [value_function.value(i) for i in STATES]
    plt.plot(STATES, stateValues, label='Approximate TD value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

# different alphas and steps for semi-gradient TD
def figure_9_2_right(true_value):
    # all possible steps
    steps = np.power(2, np.arange(0, 10))   #n步自举法的n

    # all possible alphas
    alphas = np.arange(0, 1.1, 0.1) #步长

    # each run has 10 episodes  只计算前10次的均方误差
    episodes = 10

    # perform 100 independent runs
    runs = 100

    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(runs)):
        for step_ind, step in zip(range(len(steps)), steps):#比较不同n和不同步长对学习性能的影响
            for alpha_ind, alpha in zip(range(len(alphas)), alphas):
                # we have 20 aggregations in this example
                value_function = ValueFunction(20)
                for ep in range(0, episodes):
                    semi_gradient_temporal_difference(value_function, step, alpha)
                    # calculate the RMS error
                    state_value = np.asarray([value_function.value(i) for i in STATES])#先各个状态的估计值与真实值作差再平方，然后求和，再对总状态数求平均，开方，最后把前10次episode的相加
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(state_value - true_value[1: -1], 2)) / N_STATES)
    # take average
    errors /= episodes * runs   #对run和10次episode求平局
    # truncate the error
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()

def figure_9_2(true_value):
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    figure_9_2_left(true_value)
    plt.subplot(2, 1, 2)
    figure_9_2_right(true_value)

    plt.savefig('figure_9_2.png')
    plt.close()

# Figure 9.5, Fourier basis and polynomials
def figure_9_5(true_value):
    # my machine can only afford 1 run
    runs = 1

    episodes = 5000

    # # of bases
    orders = [5, 10, 20]

    alphas = [1e-4, 5e-5]
    labels = [['polynomial basis'] * 3, ['fourier basis'] * 3]

    # track errors for each episode
    errors = np.zeros((len(alphas), len(orders), episodes))
    for run in range(runs): #run
        for i in range(len(orders)):    #阶数
            value_functions = [BasesValueFunction(orders[i], POLYNOMIAL_BASES), BasesValueFunction(orders[i], FOURIER_BASES)]
            for j in range(len(value_functions)):   #构建价值近似函数的方法
                for episode in tqdm(range(episodes)):   #episode

                    # gradient Monte Carlo algorithm
                    gradient_monte_carlo(value_functions[j], alphas[j])

                    # get state values under current value function 这里又要对每个状态用特征向量×权重向量求一次价值
                    state_values = [value_functions[j].value(state) for state in STATES]

                    # get the root-mean-squared error  按方法×阶数×episode记录误差
                    errors[j, i, episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

    # average over independent runs
    errors /= runs

    for i in range(len(alphas)):#前一个alpha对应多项式，后一个alpha对应傅里叶级数
        for j in range(len(orders)):#误差随episode变化的曲线
            plt.plot(errors[i, j, :], label='%s order = %d' % (labels[i][j], orders[j]))
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('figure_9_5.png')
    plt.close()

# Figure 9.10, it will take quite a while
def figure_9_10(true_value):

    # My machine can only afford one run, thus the curve isn't so smooth
    runs = 1

    # number of episodes
    episodes = 5000

    num_of_tilings = 50 #单个tiling，就是1000个状态分成5个格子 ；多个tiling，就是第一个tiling5个格子，1->200,201->400
                    #第2个tiling也是5个格子，但偏移了4个状态，5->205,206->406
    # each tile will cover 200 states
    tile_width = 200    #每个格子包含的状态数

    # how to put so many tilings
    tiling_offset = 4

    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']

    # track errors for each episode
    errors = np.zeros((len(labels), episodes))
    for run in range(runs):
        # initialize value functions for multiple tilings and single tiling
        value_functions = [TilingsValueFunction(num_of_tilings, tile_width, tiling_offset),
                         ValueFunction(N_STATES // tile_width)]
        for i in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                # I use a changing alpha according to the episode instead of a small fixed alpha
                # With a small fixed alpha, I don't think 5000 episodes is enough for so many
                # parameters in multiple tilings.
                # The asymptotic performance for single tiling stays unchanged under a changing alpha,
                # however the asymptotic performance for multiple tilings improves significantly
                alpha = 1.0 / (episode + 1)

                # gradient Monte Carlo algorithm
                gradient_monte_carlo(value_functions[i], alpha)

                # get state values under current value function
                state_values = [value_functions[i].value(state) for state in STATES]

                # get the root-mean-squared error
                errors[i][episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

    # average over independent runs
    errors /= runs

    for i in range(0, len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('figure_9_10.png')
    plt.close()

if __name__ == '__main__':
    true_value = compute_true_value()

    #figure_9_1(true_value)
    #figure_9_2(true_value)
    figure_9_5(true_value)
    figure_9_10(true_value)