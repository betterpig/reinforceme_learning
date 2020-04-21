import numpy as np
import matplotlib.pyplot as plt

GOAL=100
STATES=np.arange(GOAL+1)
HEAD_PROB=0.4

values=np.zeros(101)

def figure_4_3():
    values=np.zeros(GOAL+1)
    values[GOAL]=1.0

    sweep_history=[]    #存储每次价值迭代后的价值表

    #value iteration
    while True:
        old_values=values.copy()
        sweep_history.append(old_values)

        for state in STATES[1:GOAL]:
            returns=[]
            actions=np.arange(min(state,GOAL-state)+1)  #能投的钱小于等于手上有的钱
            for action in actions:  #投多少钱，要么赢回要么输掉相同数目的钱
                action_return=HEAD_PROB*values[state+action]+(1-HEAD_PROB)*values[state-action]
                returns.append(action_return)
            values[state]=np.max(returns)#in-place更新，且把策略评估和策略改善合并在一起了
        
        delta=abs(values-old_values).max()  #判断价值是否收敛
        if delta<1e-9:
            sweep_history.append(values)
            break

    #get optimal policy
    policys=np.zeros(GOAL+1)

    for state in STATES[1:GOAL]:
        returns=[]
        actions=np.arange(min(state,GOAL-state)+1)
        for action in actions:
            action_return=HEAD_PROB*values[state+action]+(1-HEAD_PROB)*values[state-action]
            returns.append(action_return)
        #policys[state]=np.argmax(returns)  #根据最终价值表获得每个状态对应的最优动作
        policys[state] = actions[np.argmax(np.round(returns[1:], 5)) + 1]

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweep_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policys)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('figure_4_3.png')
    plt.close()

if __name__ == '__main__':
    figure_4_3()