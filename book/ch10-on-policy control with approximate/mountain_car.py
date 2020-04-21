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
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import floor

#######################################################################
# Following are some utilities for tile coding from Rich.
# To make each file self-contained, I copied them from
# http://incompleteideas.net/tiles/tiles3.py-remove
# with some naming convention changes
#
# Tile coding starts
class IHT:
    "Structure to handle collisions"
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count

def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): 
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): 
        return hash(tuple(coordinates)) % m
    if m is None: return coordinates

def tiles(iht_or_size, num_tilings, states, actions=None, read_only=False):
    """returns num-tilings tile indices corresponding to the states and actions"""
    if actions is None:
        actions = []
    q_states = [floor(state * num_tilings) for state in states]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for qstate in q_states:
            coords.append((qstate + b) // num_tilings)
            b += tilingX2
        coords.extend(actions)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles
# Tile coding ends
#######################################################################

# all possible actions
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1
# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0

# take an @action at @position and @velocity
# @return: new position, new velocity, reward (always -1)
def step(position, velocity, action):
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)    #先更新速度
    new_velocity = min(max(VELOCITY_MIN, new_velocity), VELOCITY_MAX)
    new_position = position + new_velocity  #再更新位置
    new_position = min(max(POSITION_MIN, new_position), POSITION_MAX)
    reward = -1.0   #反正reward一直是-1，直接在上一层赋值就好了，不用返回reward。但按强化学习的过程来说，应该实在step返回下一个状态和reward
    if new_position == POSITION_MIN:
        new_velocity = 0.0
    return new_position, new_velocity, reward

# wrapper class for state action value function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @max_size: the maximum # of indices  #这是一个二维tiling，先对二维的范围tiling，然后再对二维网格进行偏置
    def __init__(self, step_size, num_of_tilings=8, max_size=2048):#tilings×tiles of position×tiles of velocity×action=8×8×8×4
        self.max_size = max_size    #8个tiling，那每个状态对会对应8个网格，这8个网格分别位于这8个tiling
        self.num_of_tilings = num_of_tilings

        # divide step size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)   #权重即value

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def get_active_tiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                            [self.position_scale * position, self.velocity_scale * velocity],
                            [action])
        return active_tiles

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)    #先找到该状态所在的各个网格
        return np.sum(self.weights[active_tiles])   #把所在的各个网格对应的权重相加作为该状态的价值

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.step_size * (target - estimation)  #偏导为1
        for active_tile in active_tiles:    #每个所在网格都更新权重，且delta是平均到8个tiling的，因为step_size除了tiling_num
            self.weights[active_tile] += delta

    # get # of steps to reach the goal under current state value function
    def cost_to_go(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))    #cost即动作价值，reward一直为-1所以cost都是负的，取最大值就是动作价值最大的
        return -np.max(costs)   #而reward每次是-1，所以取相反数就得到还需要多少步终止

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def get_action(position, velocity, value_function):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)    #随机
    values = []
    for action in ACTIONS:
        values.append(value_function.value(position, velocity, action)) #取得当前状态能采取的各个动作的价值
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)]) - 1 #values只存了value，action_实际上是索引0 1 2，所以最后-1，刚好与动作-1 0 1对应

# semi-gradient n-step Sarsa
# @valueFunction: state value function to learn
# @n: # of steps
def semi_gradient_n_step_sarsa(value_function, n=1):
    # start at a random position around the bottom of the valley
    current_position = np.random.uniform(-0.6, -0.4)
    # initial velocity is 0
    current_velocity = 0.0
    # get initial action
    current_action = get_action(current_position, current_velocity, value_function)

    # track previous position, velocity, action and reward
    positions = [current_position]
    velocities = [current_velocity]
    actions = [current_action]
    rewards = [0.0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')    #先设成无穷，那在还没终止时time肯定会小于T
    while True:
        # go to next time step
        time += 1

        if time < T:
            # take current action and go to the new state   #
            new_postion, new_velocity, reward = step(current_position, current_velocity, current_action)
            # choose new action
            new_action = get_action(new_postion, new_velocity, value_function)  #基于新的状态又选择了动作

            # track new state and action
            positions.append(new_postion)   #储存
            velocities.append(new_velocity)
            actions.append(new_action)
            rewards.append(reward)

            if new_postion == POSITION_MAX: #更新T如果终止了
                T = time

        # get the time of the state to update
        update_time = time - n  #被更新的时间步
        if update_time >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(update_time + 1, min(T, update_time + n) + 1):   #累加后n个reward
                returns += rewards[t]
            # add estimated state action value to the return
            if update_time + n <= T:    #加上第n个状态的价值
                returns += value_function.value(positions[update_time + n], #从储存的序列中按索引取出来
                                                velocities[update_time + n],
                                                actions[update_time + n])
            # update the state value function
            if positions[update_time] != POSITION_MAX:
                value_function.learn(positions[update_time], velocities[update_time], actions[update_time], returns)
        if update_time == T - 1:    #如果被更新步到了最后，那么所有的状态都更新过了
            break
        current_position = new_postion
        current_velocity = new_velocity
        current_action = new_action

    return time

# print learned cost to go
def print_cost(value_function, episode, ax):
    grid_size = 40
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    # positionStep = (POSITION_MAX - POSITION_MIN) / grid_size
    # positions = np.arange(POSITION_MIN, POSITION_MAX + positionStep, positionStep)
    # velocityStep = (VELOCITY_MAX - VELOCITY_MIN) / grid_size
    # velocities = np.arange(VELOCITY_MIN, VELOCITY_MAX + velocityStep, velocityStep)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)
    axis_x = []
    axis_y = []
    axis_z = []
    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            axis_z.append(value_function.cost_to_go(position, velocity))

    ax.scatter(axis_x, axis_y, axis_z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.set_title('Episode %d' % (episode + 1))

# Figure 10.1, cost to go in a single run
def figure_10_1():
    episodes = 9000
    plot_episodes = [0, 99, episodes - 1]
    fig = plt.figure(figsize=(40, 10))
    axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]
    num_of_tilings = 8
    alpha = 0.3
    value_function = ValueFunction(alpha, num_of_tilings)
    for ep in tqdm(range(episodes)):
        semi_gradient_n_step_sarsa(value_function)  #这里只是更新值函数，没有记录步数
        if ep in plot_episodes:
            print_cost(value_function, ep, axes[plot_episodes.index(ep)])

    plt.savefig('figure_10_1.png')
    plt.close()

# Figure 10.2, semi-gradient Sarsa with different alphas
def figure_10_2():
    runs = 10
    episodes = 500
    num_of_tilings = 8
    alphas = [0.1, 0.2, 0.5]

    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        value_functions = [ValueFunction(alpha, num_of_tilings) for alpha in alphas]
        for index in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                step = semi_gradient_n_step_sarsa(value_functions[index])   #每次episode的步数
                steps[index, episode] += step   #按函数×episode记录步数

    steps /= runs   #对run求平均

    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='alpha = '+str(alphas[i])+'/'+str(num_of_tilings))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    plt.savefig('figure_10_2.png')
    plt.close()

# Figure 10.3, one-step semi-gradient Sarsa vs multi-step semi-gradient Sarsa
def figure_10_3():
    runs = 10
    episodes = 500
    num_of_tilings = 8
    alphas = [0.5, 0.3]     #不同的n对应不同的alpha
    n_steps = [1, 8]

    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        value_functions = [ValueFunction(alpha, num_of_tilings) for alpha in alphas]
        for index in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                step = semi_gradient_n_step_sarsa(value_functions[index], n_steps[index])
                steps[index, episode] += step

    steps /= runs

    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='n = %.01f' % (n_steps[i]))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    plt.savefig('figure_10_3.png')
    plt.close()

# Figure 10.4, effect of alpha and n on multi-step semi-gradient Sarsa
def figure_10_4():
    alphas = np.arange(0.25, 1.75, 0.25)    #在范围0.25->1.75(不包括1.75)内按0.25的间隔取值
    n_steps = np.power(2, np.arange(0, 5))
    episodes = 50
    runs = 5

    max_steps = 300
    steps = np.zeros((len(n_steps), len(alphas)))   #按n×alpha记录
    for run in range(runs):
        for n_step_index, n_step in enumerate(n_steps):
            for alpha_index, alpha in enumerate(alphas):
                if (n_step == 8 and alpha > 1) or \
                        (n_step == 16 and alpha > 0.75):
                    # In these cases it won't converge, so ignore them
                    steps[n_step_index, alpha_index] += max_steps * episodes
                    continue
                value_function = ValueFunction(alpha)
                for episode in tqdm(range(episodes)):
                    step = semi_gradient_n_step_sarsa(value_function, n_step)   #n是梯度函数的参数，而不是价值函数的参数
                    steps[n_step_index, alpha_index] += step

    # average over independent runs and episodes
    steps /= runs * episodes

    for i in range(0, len(n_steps)):
        plt.plot(alphas, steps[i, :], label='n = '+str(n_steps[i]))
    plt.xlabel('alpha * number of tilings(8)')
    plt.ylabel('Steps per episode')
    plt.ylim([220, max_steps])
    plt.legend()

    plt.savefig('figure_10_4.png')
    plt.close()

if __name__ == '__main__':
    figure_10_1()
    figure_10_2()
    figure_10_3()
    figure_10_4()