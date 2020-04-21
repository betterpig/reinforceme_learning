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

ACTION_LEFT = 0
ACTION_RIGHT = 1

# behavior policy
def behavior_policy():
    return np.random.binomial(1, 0.5)

# target policy
def target_policy():
    return ACTION_LEFT

# one turn
def play():
    # track the action for importance ratio
    trajectory = []
    while True:
        action = behavior_policy()
        trajectory.append(action)
        if action == ACTION_RIGHT:  #往右直接终止
            return 0, trajectory
        if np.random.binomial(1, 0.1) == 1: #往左有0.1的概率终止
            return 1, trajectory

def figure_5_4():
    runs = 10
    episodes = 100000
    for run in range(runs):
        rewards = []
        for episode in range(0, episodes):
            reward, trajectory = play()
            if trajectory[-1] == ACTION_RIGHT:  #如果最后是往右到了终止，那分子就是0，因为pi策略往右的概率为0
                rho = 0
            else:
                rho = 1.0 / pow(0.5, len(trajectory))   #pi策略一直往左，所以分子为1.b策略左右各半，所以是0.5次幂
            rewards.append(rho * reward)    #这里的rewards就是每次episode的return
        rewards = np.add.accumulate(rewards)
        estimations = np.asarray(rewards) / np.arange(1, episodes + 1)  #普通重要性采样
        plt.plot(estimations)   #estimation本身就对应着episode
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.savefig('figure_5_4.png')
    plt.close()

if __name__ == '__main__':
    figure_5_4()