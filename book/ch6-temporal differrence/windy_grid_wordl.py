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

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_LU=4 #可以走斜的
ACTION_LD=5
ACTION_RU=6
ACTION_RD=7
ACTION_STAND=8  #不走

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,ACTION_LU,ACTION_LD,ACTION_RU,ACTION_RD,ACTION_STAND]

def step(state, action):#走一步
    i, j = state    #i是行 j是列
    if action == ACTION_UP: #向上走只可能超出上边界
        return [max(i - 1 - WIND[j], 0), j] #超出边界则留在边界
    elif action == ACTION_DOWN: #但是向下走既可能超出下边界，也可能超出上边界，因为风都是往上吹的
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action ==ACTION_LU:
        return [max(i - 1 - WIND[j], 0), max(j - 1, 0)]
    elif action ==ACTION_LD:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0),max(j - 1, 0)]
    elif action ==ACTION_RU:
        return [max(i - 1 - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action ==ACTION_RD: 
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action ==ACTION_STAND:
        return [max(i - WIND[j], 0), j]   
    else:
        assert False

# play for an episode
def episode(q_value):
    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1: #二项分布，以episolon的概率是1
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        #对于所有具有最大价值的动作，随机选择其中一个
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action]-q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1   #记录所需要的steps数
    return time

def figure_6_3():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 9))#本来状态就是指坐标的，这里把坐标展开来变成了二维的，再乘以四种动作得到状态动作表
    episode_limit = 500

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))  #记录每次episode所需的steps数
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    #steps = np.add.accumulate(steps)    #累加：计算对应位置的前面所有数的和

    #plt.plot(steps, np.arange(1, len(steps) + 1))   #累加后的steps随episode变化的曲线
    plt.plot(steps) #不要累加，一点也不直观
    #plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
    plt.xlabel('Episodes')
    plt.ylabel('Time steps')
    plt.savefig('figure_6_3.png')
    plt.close()
    
    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):    #在每个格子时，所应该采取的动作，不局限于给定的起点
        optimal_policy.append([])   #新增一空行
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')  #在最后一行append，也即新增的那行
                continue
            bestAction = np.argmax(q_value[i, j, :])#在该状态，有四个相应的动作价值，选最大的
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
            elif bestAction ==ACTION_LU:
                optimal_policy[-1].append('LU')
            elif bestAction ==ACTION_LD:
                optimal_policy[-1].append('LD')
            elif bestAction ==ACTION_RU:
                optimal_policy[-1].append('RU')
            elif bestAction ==ACTION_RD: 
                optimal_policy[-1].append('RD')
            elif bestAction ==ACTION_STAND:
                optimal_policy[-1].append('0')
    
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))

if __name__ == '__main__':
    figure_6_3()