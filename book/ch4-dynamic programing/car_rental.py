#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns   
from scipy.stats import poisson

matplotlib.use('Agg')   #不在运行时显示图片


MAX_CARS = 20   # maximum # of cars in each location
MAX_MOVE_OF_CARS = 5    # maximum # of cars to move during night
RENTAL_REQUEST_FIRST_LOC = 3    # expectation for rental requests in first location
RENTAL_REQUEST_SECOND_LOC = 4   # expectation for rental requests in second location
RETURNS_FIRST_LOC = 3   # expectation # of cars returned in first location
RETURNS_SECOND_LOC = 2  # expectation # of cars returned in second location
DISCOUNT = 0.9
RENTAL_CREDIT = 10  # credit earned by a car
MOVE_CAR_COST = 2   # cost of moving a car
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)    # all possible actions

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
poisson_cache = dict()

def poisson_probability(n, lam):    #计算以lam为均值的泊松分布，成功次数为n的概率
    global poisson_cache
    key = n * 10 + lam  #用key作为泊松概率的索引，key的值有3,13,23,33,43,53,63,73,83,93,103;4,14,24,34，。。。
    if key not in poisson_cache:    #如果字典中不存在该key，就把该key和对应的概率加到字典中
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

def expected_return(state, action, state_value, constant_returned_cars):
    """
    @state: [cars_number in first location, cars_number in second location]
    @action: positive if moving cars from first location to second location,
            negative if moving cars from second location to first location
    @stateValue: state value matrix
    @constant_returned_cars:  if set True, model is simplified such that
    the number of cars returned in daytime becomes constant
    rather than a random value from poisson distribution, which will reduce calculation time
    and leave the optimal policy/value state matrix almost the same
    """
    
    returns = 0.0   # initailize total return
    #不管两个地方要借出多少辆车，运车的花费都只取决于action，和后面转移到哪个状态无关
    returns -= MOVE_CAR_COST * abs(action)  # cost for moving cars 

    # moving cars
    NUM_OF_CARS_FIRST_LOC = min(state[0] - action, MAX_CARS)    #最多只能有20辆车
    NUM_OF_CARS_SECOND_LOC = min(state[1] + action, MAX_CARS)

    # go through all possible rental requests
    for rental_request_first_loc in range(POISSON_UPPER_BOUND): #最多租10辆车
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            # probability for current combination of rental requests
            prob = poisson_probability(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * \
                poisson_probability(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            num_of_cars_first_loc = NUM_OF_CARS_FIRST_LOC
            num_of_cars_second_loc = NUM_OF_CARS_SECOND_LOC

            # valid rental number should be less than actual number of cars 
            valid_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
            valid_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)

            # get reward for renting 是因为处于该种状态（两地各有多少辆车），才有了相应的reward
            #比如，如果某地没有足够车来借，那相应的reward就要减少
            reward = (valid_rental_first_loc + valid_rental_second_loc) * RENTAL_CREDIT
            num_of_cars_first_loc -= valid_rental_first_loc
            num_of_cars_second_loc -= valid_rental_second_loc

            if constant_returned_cars:  #每天还车的数量固定
                # get returned cars, those cars can be used for renting tomorrow
                returned_cars_first_loc = RETURNS_FIRST_LOC
                returned_cars_second_loc = RETURNS_SECOND_LOC
                num_of_cars_first_loc = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                num_of_cars_second_loc = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                #在状态s采取动作a后，转移到状态s'，这里的s'取决于借的车和还的车，实际上应该要按泊松分布
                #生成借车数量和换车数量，然后得到最终两地的库存车数量，作为s'，再去求和算return
                #这里借数量设了上限10，按理上限应该是20，因为车库最多20辆车
                #还有一个问题，泊松分布是取值应该是无限的，这里不管怎样都要进行截取，导致实际上各种可能结果的
                #概率加起来要小于1
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])
            else:#  按泊松分布生成还车数量
                for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                    for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(
                            returned_cars_first_loc, RETURNS_FIRST_LOC) * poisson_probability(returned_cars_second_loc, RETURNS_SECOND_LOC)
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, MAX_CARS)
                        prob_ = prob_return * prob
                        returns += prob_ * (reward + DISCOUNT *
                                            state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])
    return returns

def figure_4_2(constant_returned_cars=True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))  #两个地方不同车辆个数的组合就对应一种state
    policy = np.zeros(value.shape, dtype=np.int)    #状态对应相应的action

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])#状态策略图
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # policy evaluation (in-place)每更新一个状态值，就把之前的覆盖了；然后更新下一个状态值时，就用已经更新的状态值去计算
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max() 
            print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:#当状态值变化小于某一阈值时，认为已经收敛，得到了当前policy的实际状态价值
                break

        # policy improvement 要进行很多次价值迭代得到收敛的value才会进行一次improvement
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):   #对每个状态
                old_action = policy[i, j]
                action_returns = []
                for action in actions:  #求在该状态时采取每个动作的价值，然后取最大的作为该处于该状态时要采取的动作
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)  #因为要取最大，所以不能采取的动作设为负无穷就好了
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:  #如果新动作不等于旧动作，就不收敛
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))

        if policy_stable:   #最后一幅图  状态价值图
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break

        iterations += 1

    plt.savefig('figure_4_2.png')
    plt.close()


if __name__ == '__main__':
    figure_4_2()
