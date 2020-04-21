#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# This file is contributed by Tahsincan Köse which implements a synchronous policy evaluation, while the car_rental.py
# implements an asynchronous policy evaluation. This file also utilizes multi-processing for acceleration and contains
# an answer to Exercise 4.5

import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import multiprocessing as mp    #子进程
from functools import partial   #固定函数某个参数，返回新的函数
import time
import itertools    #迭代器

############# PROBLEM SPECIFIC CONSTANTS #######################
MAX_CARS = 20
MAX_MOVE = 5
MOVE_COST = -2
ADDITIONAL_PARK_COST = -4

RENT_REWARD = 10
RENTAL_REQUEST_FIRST_LOC = 3# expectation
RENTAL_REQUEST_SECOND_LOC = 4# expectation
RETURNS_FIRST_LOC = 3# expectation
RETURNS_SECOND_LOC = 2# expectation

poisson_cache = dict()

def poisson(n, lam):#泊松概率 
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = math.exp(-lam) * math.pow(lam, n) / math.factorial(n)
    return poisson_cache[key]


class PolicyIteration:
    def __init__(self, truncate, parallel_processes, delta=1e-2, gamma=0.9, solve_4_5=False):
        self.TRUNCATE = truncate
        self.NR_PARALLEL_PROCESSES = parallel_processes
        self.actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)
        self.inverse_actions = {el: ind[0] for ind, el in np.ndenumerate(self.actions)}
        self.values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        self.policy = np.zeros(self.values.shape, dtype=np.int)
        self.delta = delta
        self.gamma = gamma
        self.solve_extension = solve_4_5

    def solve(self):
        iterations = 0
        total_start_time = time.time()
        while True:
            start_time = time.time()
            self.values = self.policy_evaluation(self.values, self.policy)
            elapsed_time = time.time() - start_time#evaluation time
            print(f'PE => Elapsed time {elapsed_time} seconds')
            
            start_time = time.time()
            policy_change, self.policy = self.policy_improvement(self.actions, self.values, self.policy)
            elapsed_time = time.time() - start_time
            print(f'PI => Elapsed time {elapsed_time} seconds')#improvement time

            if policy_change == 0:#当policy不再变化，结束迭代
                break
            iterations += 1
        total_elapsed_time = time.time() - total_start_time#迭代总时间
        print(f'Optimal policy is reached after {iterations} iterations in {total_elapsed_time} seconds')

    # out-place  计算完所有value，才会把就values覆盖，同步更新
    def policy_evaluation(self, values, policy):

        global MAX_CARS
        while True:
            new_values = np.copy(values)
            k = np.arange(MAX_CARS + 1)
            # cartesian product AXB 第一个对象是A第二个对象是B所组成的所有可能有序对
            all_states = ((i, j) for i, j in itertools.product(k, k))#k是一维向量，两个k做笛卡尔乘积

            results = []
            with mp.Pool(processes=self.NR_PARALLEL_PROCESSES) as p:
                #将policy和values作为参数依次付给expected_return_pe函数，使其前两个参数固定，
                # 生成一个新的函数cook，因此cook只需要一个参数states就可
                cook = partial(self.expected_return_pe, policy, values)
                #map类似于bind，将all_states作为参数传入cook函数，并开始该进程
                #每次传入all_states的一个元素，all_states是二维数组，每次拿出一个二元对是一位数组
                results = p.map(cook, all_states)

            for v, i, j in results:#把results还原成状态作为索引的二维数组形式
                new_values[i, j] = v

            difference = np.abs(new_values - values).sum()#计算新旧value的差值
            print(f'Difference: {difference}')
            values = new_values
            if difference < self.delta:#小于容差则认为收敛
                print(f'Values are converged!')
                return values

    def policy_improvement(self, actions, values, policy):
        new_policy = np.copy(policy)

        expected_action_returns = np.zeros((MAX_CARS + 1, MAX_CARS + 1, np.size(actions)))
        cooks = dict()
        with mp.Pool(processes=8) as p:
            for action in actions:#对每个动作
                k = np.arange(MAX_CARS + 1)
                all_states = ((i, j) for i, j in itertools.product(k, k))#对每个状态
                cooks[action] = partial(self.expected_return_pi, values, action)#values和action固定，state作为cooks的参数
                results = p.map(cooks[action], all_states)#对每个状态动作对计算相应的return
                for v, i, j, a in results:
                    expected_action_returns[i, j, self.inverse_actions[a]] = v#三维数组
        for i in range(expected_action_returns.shape[0]):
            for j in range(expected_action_returns.shape[1]):#第一维和第二维组合起来就表示状态
                new_policy[i, j] = actions[np.argmax(expected_action_returns[i, j])]#该状态下的最优动作
                #按i，j取出returns中的元素后，每个元素就是action的一维向量

        policy_change = (new_policy != policy).sum()#统计有多少个状态的最优动作不同
        print(f'Policy changed in {policy_change} states')
        return policy_change, new_policy

    # O(n^4) computation for all possible requests and returns
    def bellman(self, values, action, state):#给定当前状态和动作
        expected_return = 0
        if self.solve_extension:
            if action > 0:
                # Free shuttle to the second location
                expected_return += MOVE_COST * (action - 1)
            else:
                expected_return += MOVE_COST * abs(action)
        else:
            expected_return += MOVE_COST * abs(action)

        for req1 in range(0, self.TRUNCATE):#借车数量
            for req2 in range(0, self.TRUNCATE):
                # moving cars
                num_of_cars_first_loc = int(min(state[0] - action, MAX_CARS))
                num_of_cars_second_loc = int(min(state[1] + action, MAX_CARS))

                # valid rental requests should be less than actual # of cars
                real_rental_first_loc = min(num_of_cars_first_loc, req1)
                real_rental_second_loc = min(num_of_cars_second_loc, req2)

                # get credits for renting
                reward = (real_rental_first_loc + real_rental_second_loc) * RENT_REWARD

                if self.solve_extension:
                    if num_of_cars_first_loc >= 10:
                        reward += ADDITIONAL_PARK_COST
                    if num_of_cars_second_loc >= 10:
                        reward += ADDITIONAL_PARK_COST

                num_of_cars_first_loc -= real_rental_first_loc#借车之后剩余的车辆
                num_of_cars_second_loc -= real_rental_second_loc

                # probability for current combination of rental requests
                prob = poisson(req1, RENTAL_REQUEST_FIRST_LOC)*poisson(req2, RENTAL_REQUEST_SECOND_LOC)#当前借车数量组合的概率
                for ret1 in range(0, self.TRUNCATE):#还车数量
                    for ret2 in range(0, self.TRUNCATE):
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + ret1, MAX_CARS)#还车后库车数量
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + ret2, MAX_CARS)
                        #当前还车数量组合与借车数量组合同时发生的概率，该组合与状态s，以及动作a，一起决定了下一个状态s'时两地库存数量
                        prob_ = poisson(ret1, RETURNS_FIRST_LOC)*poisson(ret2, RETURNS_SECOND_LOC) * prob
                        # Classic Bellman equation for state-value
                        # prob_ corresponds to p(s'|s,a) 
                        expected_return += prob_ * (reward + self.gamma * values[num_of_cars_first_loc_, num_of_cars_second_loc_])
        return expected_return

    # Parallelization enforced different helper functions
    # Expected return calculator for Policy Evaluation
    def expected_return_pe(self, policy, values, state):

        action = policy[state[0], state[1]]
        expected_return = self.bellman(values, action, state)
        return expected_return, state[0], state[1]#返回三元组

    # Expected return calculator for Policy Improvement
    def expected_return_pi(self, values, action, state):
        #action不能比state大
        if ((action >= 0 and state[0] >= action) or (action < 0 and state[1] >= abs(action))) == False:
            return -float('inf'), state[0], state[1], action
        expected_return = self.bellman(values, action, state)
        return expected_return, state[0], state[1], action

    def plot(self):
        print(self.policy)
        plt.figure()
        plt.xlim(0, MAX_CARS + 1)
        plt.ylim(0, MAX_CARS + 1)
        plt.table(cellText=self.policy, loc=(0, 0), cellLoc='center')
        plt.show()


if __name__ == '__main__':
    TRUNCATE = 9
    solver = PolicyIteration(TRUNCATE, parallel_processes=4, delta=1e-1, gamma=0.9, solve_4_5=True)
    solver.solve()
    solver.plot()