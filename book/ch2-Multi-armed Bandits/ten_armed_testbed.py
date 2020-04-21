#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange #进度条

matplotlib.use('Agg')

class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm

    def __init__(self,k_arm=10,epsilon=0.,initial=0.,step_size=0.1,sample_averages=False,
                UCB_param=None,gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        #按标准正态分布生成10个数，作为摇臂回报正态分布的均值 。对于gradient，还要每个数加上偏移量 
        self.q_true=np.random.randn(self.k)+self.true_reward

        #最优动作即上述生成的10个均值中最大的对应的索引，这里几乎不可能两个同样的随机数，所以不用担心最大值多个的问题
        self.best_action=np.argmax(self.q_true)

        #estimation of action value，默认初始时为0. 对于最优初始值方法，则为可变参数
        self.q_estimation=np.zeros(self.k)+self.initial

        #chosen times for each action
        self.action_count=np.zeros(self.k)

        self.time=0

    #get an action for this time bandit according to chosen leaning method
    def act(self):
        if np.random.rand()<self.epsilon:#如果小于epsilon则随机，大于则转到函数最后，取estimation最大的
            return np.random.choice(self.indices)
        
        if self.UCB_param is not None:
            UCB_estimation=self.q_estimation+self.UCB_param*np.sqrt(np.log(self.time+1)/(self.action_count+1e-5))#分母加上一个很小的数，避免除以0报错
            q_best=np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation==q_best)[0])#同理应对最大值出现多次的情况

        if self.gradient:
            exp_est=np.exp(self.q_estimation)
            self.action_prob=exp_est/np.sum(exp_est)
            return np.random.choice(self.indices,p=self.action_prob)#按照action_prob的概率分布随机选择臂
        
        q_best=np.max(self.q_estimation)    #为什么要先找到最大，再找最大对应的索引，直接用np.argmax不就好了？
        return np.random.choice(np.where(self.q_estimation==q_best)[0])#找到所有具有最大值的项，再随机选择一个。这样即使在一开始都是0的时候，也能随机选一个
        #max_action=np.argmax(self.q_estimation)  #对于最大值出现多次的情况，argmax返回第一次出现的索引
        # return max_action

    #take an action ,update estimation for this action
    def step(self,action):
        #generate the reward under N(real reward,1)
        reward=np.random.randn()+self.q_true[action]
        self.time+=1
        self.action_count[action]+=1#先加1 再去更新各种值
        self.average_reward+=(reward-self.average_reward)/self.time

        if self.sample_averages:
            #update estimation using sample averages 样本平均 每次的reward只用来更新对应动作的estimation
            self.q_estimation[action]+=(reward-self.q_estimation[action])/self.action_count[action]
        elif self.gradient:
            one_hot=np.zeros(self.k)
            one_hot[action]=1#因此每次action不同，所以每次都要新建一个one hot
            if self.gradient_baseline:
                baseline=self.average_reward
            else:
                baseline=0
            #这里的estimation实际上是action_preference
            self.q_estimation+=self.step_size*(reward-baseline)*(one_hot-self.action_prob)

        else:
            #update estimation with constant step size  weighted average（加权平均）  每次的reward用来更新10个臂的 preference
            self.q_estimation[action]+=self.step_size*(reward-self.q_estimation[action])
        return reward

def simulate(runs,time,bandits):
    rewards=np.zeros((len(bandits),runs,time))#三个维度：方法数(不同算法，不同参数) X 运行次数 X 迭代步
    best_action_counts=np.zeros(rewards.shape)
    for i,bandit in enumerate(bandits):#enumerate 将下标与列表中的元素绑定   for each leaning method 
        for r in trange(runs):#trange将该循环与进度条绑定  each run
            bandit.reset()#初始化bandit problem对象  apply to a bandit problem generate randomly
            for t in range(time):#迭代学习  experience over 1000 time step
                action=bandit.act() #根据不同leaning method 选择动作
                reward=bandit.step(action)  #根据动作生成回报
                rewards[i,r,t]=reward   #按leaning method，run，time索引记录该reward
                if action==bandit.best_action:
                    best_action_counts[i,r,t]=1 #按索引更新最佳动作被采取个数
    mean_best_action_counts=best_action_counts.mean(axis=1) #对每一个时间步，对2000次求平均
    mean_rewards=rewards.mean(axis=1)
    return mean_best_action_counts,mean_rewards

def figure_2_1():   #reward distribution of an example bandit problem
    #随机生成10个数作均值，然后加上对应的用标准正态分布生成的一列数，相当于图像偏移。
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    #plt.savefig('./images/figure_2_1.png')
    plt.close()

def figure_2_2(runs=3, time=1000):   #compares a greedy method with two epsilon-greedy methods
    epsilons=[0,0.1,0.01]
    bandits=[Bandit(epsilon=eps,sample_averages=True) for eps in epsilons]#用epsilon中的每个元素作为参数创建bandit对象，组成bandits列表
    best_action_counts,rewards=simulate(runs,time,bandits)

    plt.figure(figsize=(10,20))

    plt.subplot(2,1,1)  #绘制不同epsilon，2000次运行每一步的平均reward随time的变化曲线
    for eps,rewards in zip(epsilons,rewards):   
        plt.plot(rewards,label='epsilon=%.02f' % (eps))#只有y，没有x。x default to range(len(y))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)    #绘制不同epsilon，2000次运行每一步的平均最优动作被采取率随time的变化曲线
    for eps,counts in zip(epsilons,best_action_counts):
        plt.plot(counts,label='epsilon=%.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    #plt.savefig('./images/figure_2_2.png')
    plt.close()

def figure_2_3(runs=3, time=1000):#The efect of optimistic initial action-value estimates
    bandits = []
    bandits.append(Bandit(epsilon=0, initial=5, step_size=0.1))
    bandits.append(Bandit(epsilon=0.1, initial=0, step_size=0.1))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='epsilon = 0, q = 5')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, q = 0')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    #plt.savefig('./images/figure_2_3.png')
    plt.close()

def figure_2_4(runs=3, time=1000):#comparison between UCB and epsilon-greedy
    bandits = []
    bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, sample_averages=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB c = 2')
    plt.plot(average_rewards[1], label='epsilon greedy epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    #plt.savefig('./images/figure_2_4.png')
    plt.close()

def figure_2_5(runs=3, time=1000):#comparison of the gradient bandit algorithm with and without a reward
#baseline ,and different step_size when the q_true are chosen to be +4 rather than near zero.
    bandits = []
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    #plt.savefig('./images/figure_2_5.png')
    plt.close()

def figure_2_6(runs=3, time=1000):#2000次运行的平均第1000步reward，随不同算法，不同参数，变化的曲线
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),#epsilon是参数，input到Bandit中，构造bandit对象
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))#调用lambda匿名函数，生成一系列不同算法不同参数的bandit对象

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)#parameter作x，rewards作y
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    #plt.savefig('./images/figure_2_6.png')
    plt.close()


if __name__ == '__main__':
    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    figure_2_6()
