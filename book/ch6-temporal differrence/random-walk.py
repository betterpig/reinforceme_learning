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

# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
VALUES = np.zeros(7)
VALUES[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
VALUES[6] = 1

# set up true state values
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
def temporal_difference(values, alpha=0.1, batch=False):
    state = 3
    trajectory = [state]    #记录每个step的状态
    rewards = [0]
    while True:
        old_state = state   #上一个状态
        if np.random.binomial(1, 0.5) == ACTION_LEFT:#左右概率各半
            state -= 1
        else:
            state += 1
        # Assume all rewards are 0
        reward = 0  #本来reward就是0
        trajectory.append(state)
        # TD update
        if not batch:   #每个step都会更新单个状态
            values[old_state] += alpha * (reward + values[state] - values[old_state])
        if state == 6 or state == 0:
            break
        rewards.append(reward)  #没必要搞reward
    return trajectory, rewards

# @values: current states value, will be updated if @batch is False
# @alpha: step size
# @batch: whether to update @values
def monte_carlo(values, alpha=0.1, batch=False):
    state = 3
    trajectory = [3]

    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break

    if not batch:
        for state_ in trajectory[:-1]:
            # MC update 增量实现，本来应该是乘以总个数的，推广到了alpha
            values[state_] += alpha * (returns - values[state_]) #结束完episode每个出现的状态都更新，
    return trajectory, [returns] * (len(trajectory) - 1)    

# Example 6.2 left
def compute_state_value():
    episodes = [0, 1, 10, 100]
    current_values = np.copy(VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:   #episode等于0 1 10 100时画状态价值图
            plt.plot(current_values, label=str(i) + ' episodes')
        temporal_difference(current_values) #不是所有状态在一次episode中都会出现
    plt.plot(TRUE_VALUE, label='true values')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.legend()

# Example 6.2 right
def rms_error():    #方差
    # Same alpha value can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)   #方差随episode的变化曲线
        if i < len(td_alphas):  #前面的是td，后面的是mc
            method = 'TD'
            linestyle = 'solid'
        else:
            method = 'MC'
            linestyle = 'dashdot'
        for r in tqdm(range(runs)):#每次的episode对应的方差对100次运行求平均
            errors = []
            current_values = np.copy(VALUES)
            for i in range(0, episodes):    #均方根值
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - current_values, 2)) / 5.0))
                if method == 'TD':
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
            total_errors += np.asarray(errors)  #数组和数组相加，对应元素相加
        total_errors /= runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
    plt.xlabel('episodes')
    plt.ylabel('RMS')
    plt.legend()

# Figure 6.2
# @method: 'TD' or 'MC'
def batch_updating(method, episodes, alpha=0.001):
    # perform 100 independent runs
    runs = 100
    total_errors = np.zeros(episodes)
    for r in tqdm(range(0, runs)):
        current_values = np.copy(VALUES)    #价值表是每次run不一样，但同一次run的各个episode是static的
        errors = [] #误差也是每次episode更新一次
        # track shown trajectories and reward/return sequences
        trajectories = []   #每完成一次episode，之前的所有episode都会拿来更新value
        rewards = []    #没用的
        for ep in range(episodes):
            if method == 'TD':  #先获取序列
                trajectory_, rewards_ = temporal_difference(current_values, batch=True)
            else:
                trajectory_, rewards_ = monte_carlo(current_values, batch=True)#只是为了获得序列，传过去的value没作用
            trajectories.append(trajectory_)    #保存序列
            rewards.append(rewards_)
            while True:#用已拥有的序列取更新状态直至收敛
                # keep feeding our algorithm with trajectories seen so far until state value function converges
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):#每次取一条序列
                    for i in range(0, len(trajectory_) - 1):
                        if method == 'TD':  3#TD用到了reward，但这个例子reward始终为0
                            updates[trajectory_[i]] += rewards_[i] + current_values[trajectory_[i + 1]] - current_values[trajectory_[i]]
                        else:#MC压根就不需要reward，这里的reward应该是与序列中每个状态相对应的return，但这个例子return恒为1
                            updates[trajectory_[i]] += rewards_[i] - current_values[trajectory_[i]]#作者为了少定义一个变量就都用rewards
                updates *= alpha#对该序列计算各个状态的增量，也就是TD error，对某个序列可能某个状态增量为0
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                # perform batch updating
                current_values += updates#用算好的各个状态的增量同步更新状态价值
            # calculate rms error
            errors.append(np.sqrt(np.sum(np.power(current_values - TRUE_VALUE, 2)) / 5.0))
        total_errors += np.asarray(errors)  #按episode保存的方差
    total_errors /= runs
    return total_errors

def example_6_2():
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_state_value()

    plt.subplot(2, 1, 2)
    rms_error()
    plt.tight_layout()

    plt.savefig('example_6_2.png')
    plt.close()

def figure_6_2():
    episodes = 100 + 1
    td_erros = batch_updating('TD', episodes)
    mc_erros = batch_updating('MC', episodes)

    plt.plot(td_erros, label='TD')
    plt.plot(mc_erros, label='MC')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.legend()

    plt.savefig('figure_6_2.png')
    plt.close()

if __name__ == '__main__':
    example_6_2()
    figure_6_2()