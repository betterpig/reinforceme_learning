#######################################################################
# Copyright (C)                                                       #
# 2018 Sergii Bondariev (sergeybondarev@gmail.com)                    #
# 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def true_value(p):
    """ True value of the first state
    Args:
        p (float): probability of the action 'right'.
    Returns:
        True value of the first state.
        The expression is obtained by manually solving the easy linear system
        of Bellman equations using known dynamics. 
    """
    return (2 * p - 4) / (p * (1 - p))# 曲线是已知的

class ShortCorridor:
    """
    Short corridor environment, see Example 13.1
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0

    def step(self, go_right):
        """
        Args:
            go_right (bool): chosen action
        Returns:
            tuple of (reward, episode terminated?)
        """
        if self.state == 0 or self.state == 2:#第1个和第3个状态时，向右就是向右
            if go_right:
                self.state += 1
            else:
                self.state = max(0, self.state - 1)#向左就是向左
        else:   #第2个状态时，向右是向左
            if go_right:
                self.state -= 1
            else:   #向做是向右
                self.state += 1

        if self.state == 3:# 0 1 2是普通状态，3是终止状态
            # terminal state
            return 0, True  #返回本次step的reward和本次episode是否结束
        else:
            return -1, False

def softmax(x):
    t = np.exp(x - np.max(x))
    return t / np.sum(t)

class ReinforceAgent:
    """
    ReinforceAgent that follows algorithm
    'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma):
        # set values such that initial conditions correspond to left-epsilon greedy
        self.theta = np.array([-1.47, 1.47])
        self.alpha = alpha
        self.gamma = gamma
        # first column - left, second - right
        self.x = np.array([[0, 1],  #对于所有状态，只要动作向右就是（1,0），动作向左就是（0,1），这里已经把状态这个终于特征给抹去了
                           [1, 0]])
        self.rewards = []
        self.actions = []

    def get_pi(self):
        h = np.dot(self.theta, self.x)  #action preference linear in features
        t = np.exp(h - np.max(h))   #相当于归一化，对结果没有影响
        pmf = t / np.sum(t) #在状态s和theta(t)时采取各个动作的概率
        # never become deterministic,
        # guarantees episode finish
        imin = np.argmin(pmf)
        epsilon = 0.05

        if pmf[imin] < epsilon: #保证采取动作的概率大于等于epsilon
            pmf[:] = 1 - epsilon
            pmf[imin] = epsilon

        return pmf

    def get_p_right(self):
        return self.get_pi()[1]

    def choose_action(self, reward):
        if reward is not None:
            self.rewards.append(reward) #记录reward和action，等episode结束之后，还要利用序列来计算

        pmf = self.get_pi()#在0->>1上随机取点，数值小于向右的 概率则向右，也就相当于以向右的概率生成动作
        go_right = np.random.uniform() <= pmf[1]    #按照pmf的概率生成动作，这里也可用二项分布，1次试验，成功的概率为pmf1
        self.actions.append(go_right)

        return go_right

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)    #最后一次reward是0

        # learn theta
        G = np.zeros(len(self.rewards)) #每次reward都是-1，列表的总长度的相反数就是return
        G[-1] = self.rewards[-1]    #前面加负号表示从后往前数

        for i in range(2, len(G) + 1):  #从后往前算
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1   #gama的t次方，这里一直是1

        for i in range(len(G)):
            j = 1 if self.actions[i] else 0#向右就是取x里的第2列，向左就是取第1列
            pmf = self.get_pi()#基于当前theta重新计算动作概率分布，也就是说拿出每个状态都会用新的theta
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * G[i] * grad_ln_pi

            self.theta += update    #每拿出一个状态都会更新一次theta
            gamma_pow *= self.gamma

        self.rewards = []   #更新theta完毕，reward和action重新归零
        self.actions = []

class ReinforceBaselineAgent(ReinforceAgent):
    def __init__(self, alpha, gamma, alpha_w):
        super(ReinforceBaselineAgent, self).__init__(alpha, gamma)#调用基类的初始化函数
        self.alpha_w = alpha_w
        self.w = 0

    def episode_end(self, last_reward):
        self.rewards.append(last_reward)

        # learn theta and w
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            self.w += self.alpha_w * gamma_pow * (G[i] - self.w)#更新w，价值函数对w求导数就是1

            j = 1 if self.actions[i] else 0
            pmf = self.get_pi()
            grad_ln_pi = self.x[:, j] - np.dot(self.x, pmf)
            update = self.alpha * gamma_pow * (G[i] - self.w) * grad_ln_pi

            self.theta += update
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []

def trial(num_episodes, agent_generator):
    env = ShortCorridor()   #环境
    agent = agent_generator()   #agent

    rewards = np.zeros(num_episodes)
    for episode_idx in range(num_episodes):
        rewards_sum = 0
        reward = None
        env.reset()

        while True: #直到终止才算一个episode
            go_right = agent.choose_action(reward)  #因为只有左右两个动作，所以用bool值就能表示
            reward, episode_end = env.step(go_right)    #agent来选择动作，environment来给出反馈
            rewards_sum += reward   #一直累加，其实也就是步数的负数

            if episode_end:
                agent.episode_end(reward)   #每次episode结束才会进行theta更新
                break

        rewards[episode_idx] = rewards_sum#记录每个episode的总reward

    return rewards

def example_13_1():
    epsilon = 0.05
    fig, ax = plt.subplots(1, 1)

    # Plot a graph
    p = np.linspace(0.01, 0.99, 100)
    y = true_value(p)
    ax.plot(p, y, color='red')

    # Find a maximum point, can also be done analytically by taking a derivative
    imax = np.argmax(y)
    pmax = p[imax]
    ymax = y[imax]
    ax.plot(pmax, ymax, color='green', marker="*", label="optimal point: f({0:.2f}) = {1:.2f}".format(pmax, ymax))

    # Plot points of two epsilon-greedy policies    直接代数计算就好了
    ax.plot(epsilon, true_value(epsilon), color='magenta', marker="o", label="epsilon-greedy left")
    ax.plot(1 - epsilon, true_value(1 - epsilon), color='blue', marker="o", label="epsilon-greedy right")

    ax.set_ylabel("Value of the first state")
    ax.set_xlabel("Probability of the action 'right'")
    ax.set_title("Short corridor with switched actions")
    ax.set_ylim(ymin=-105.0, ymax=5)
    ax.legend()

    plt.savefig('example_13_1.png')
    plt.close()

def figure_13_1():
    num_trials = 100
    num_episodes = 1000
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=2e-4, gamma=gamma),
                        lambda : ReinforceAgent(alpha=2e-5, gamma=gamma),
                        lambda : ReinforceAgent(alpha=2e-3, gamma=gamma)]#生成三种步长的agent
    labels = ['alpha = 2e-4',
              'alpha = 2e-5',
              'alpha = 2e-3']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))#按agent×run×episode存储reward

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward
    #渐近线
    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)#先取出每个agent的reward表，再对第0轴也就是run求平均，剩下的就是每个episode的reward了
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('figure_13_1.png')
    plt.close()

def figure_13_2():
    num_trials = 100
    num_episodes = 1000
    alpha = 2e-4
    gamma = 1
    agent_generators = [lambda : ReinforceAgent(alpha=alpha, gamma=gamma),
                        lambda : ReinforceBaselineAgent(alpha=alpha*10, gamma=gamma, alpha_w=alpha*100)]
    labels = ['Reinforce without baseline',
              'Reinforce with baseline']

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward

    plt.plot(np.arange(num_episodes) + 1, -11.6 * np.ones(num_episodes), ls='dashed', color='red', label='-11.6')
    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('figure_13_2.png')
    plt.close()

if __name__ == '__main__':
    #example_13_1()
    #figure_13_1()
    figure_13_2()