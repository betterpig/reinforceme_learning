#multi_arm_bandit problem
#comparison between different methods to choose action 

import random
import numpy as np
from enum import Enum
import math
import matplotlib.pyplot as plt

action_value=[]
action_count=[]

action_preference=np.zeros((10))
action_policy=np.exp(action_preference)/sum(np.exp(action_preference))
epsilon=0
t=0

def init(arms_number):
    global action_value
    global action_count
    for i in range(arms_number):
        action_value.append(0)
        action_count.append(0)

def bandit(k,money):
    return random.gauss(money[k],1)

def exploration(arms_number):
    k=random.randint(0,arms_number-1)
    return k

def epsilon_greedy(arms_number):
    if random.random()<epsilon or t==0:
        k=exploration(arms_number)
    else:
        k=action_value.index(max(action_value))
    return k

def UCB(arms_number):
    upper_bound=[]
    c=10
    for i in range(arms_number):
        if action_count[i]==0:
            upper_bound.append(action_value[i]+c*math.sqrt(math.log(t+1)/0.01))
        else:
            upper_bound.append(action_value[i]+c*math.sqrt(math.log(t+1)/action_count[i]))
    k=upper_bound.index(max(upper_bound))
    return k

def gradient(arms_number):
    k = np.random.choice([0,1,2,3,4,5,6,7,8,9], p = action_policy.ravel())
    return k

def main():
    global action_value
    global action_count
    global action_policy
    global action_preference
    global epsilon
    global t

    arms_number=10     #摇臂机个数
    #probability=np.array([0.1,0.9,0.3,0.2,0.7])   #摇臂机吐出金币的概率
    money=np.array([0.2,-0.8,1.5,0.35,1.2,-1.7,-0.15,-1.1,0.8,-0.5])     #摇臂机吐出金币的数量
    steps=2000      #迭代步数
    methods={1:exploration, 2:epsilon_greedy ,3:UCB, 4:gradient}
    method_number=4
    action=methods.get(method_number,None)

    total_reward=0
    reward=0
    reward_list=[]
    optimal_action_list=[]
    epsilon=0.1
    alpha=0.1
    init(arms_number)

    for t in range(steps):
        k=action(arms_number)
        reward=bandit(k,money)
        if t==200 or t==500 or t==800 or t==1200 or t==1500 :
            a=1
        
        action_count[k]=action_count[k]+1
        total_reward+=reward
        reward_list.append(total_reward/(t+1))
        optimal_action_list.append(action_count[2]/(t+1))
        action_value[k]=action_value[k]+(reward-action_value[k])/action_count[k]
        
        action_preference[k]=action_preference[k]+alpha*(reward-total_reward/(t+1))*(1-action_policy[k])
        for i in range(arms_number):
            if i != k:
                action_preference[i]=action_preference[i]-alpha*(reward-total_reward/(t+1))*action_policy[k]
        
        exp_sum=np.sum(np.exp(action_preference))
        action_policy=np.exp(action_preference)/exp_sum


    plt.figure(1)
    x=np.arange(1,steps+1,1)
    plt.plot(x,reward_list)
    plt.show()

    plt.figure(2)
    plt.plot(x,optimal_action_list)
    plt.show()

    print(total_reward)

if __name__ == "__main__":
    main()