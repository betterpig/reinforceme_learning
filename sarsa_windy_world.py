# -*- coding: utf-8 -*-
## 强化学习（六）时序差分在线控制算法SARSA ##

import numpy as np
import matplotlib

matplotlib.use('agg')
import  matplotlib.pyplot as plt

WORLD_HEIGHT=7
WORLD_WIDTH=10

WIND=[0,0,0,1,1,1,2,2,1,0]  #wind strength for each column

UP=0    #posible actions
DOWN=1
LEFT=2
RIGHT=3
ACTIONS=[UP,DOWN,LEFT,RIGHT]

epsilon=0.1 #exploration probability
alpha=0.5   #SARSA step size
reward=-1   #reward for each step

start=[3,0]
goal=[3,7]

def step(state,action):
    i,j=state
    if action==UP:
        return [max(i-1-WIND[j],0),j]
    elif action==DOWN:
        return [max(min(i+1-WIND[j],WORLD_HEIGHT-1),0),j]
    elif aciton==LEFT:
        return [max(i-WIND[j],0),max(j-1,0)]
    elif action==RIGHT:
        return [max(i-WIND[j],0),min(j+1,WORLD_WIDTH-1)]
    else :
        assert  False
def episode(q_value):
    time=0
    state=start
    if np.random.binomial(1,epsilon)==1:
        action=np.random.choice(ACTIONS)
    else:
        values=q_value[state[0],state[1],:]
        action=np.random.choice([action_ for action_,value in enumerate(values) if value==np.max(values)])
        while state !=goal:
            next_state=step(state,action)
            if np.random.binomial(1,epsilon)==1:
                next_action=np.random.choice(ACTIONS)
            else:
                values = q_value[state[0], state[1], :]
                next_action = np.random.choice([action_ for action_, value in enumerate(values) if value == np.max(values)])
            q_value[state[0],state[1],action]+=\alpha*(reward+q_value[next_state[0],next_state[1],next_action]-q_value[state[0],state[1],action])
            state=next_state
            action=next_action
            time+=1
        return time

def sarsa():
    q_value=np.zeros((WORLD_HEIGHT,WORLD_WIDTH,4))
    episode_limit=500

    steps=[]
    ep=0

    while ep<episode_limit:
        steps.append(episode(q_value))
        ep+=1
    steps=np.add.accumulate(steps)

    plt.plot(steps,np.range(1,len(steps)+1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.savefig('./sarsa.png')
    plt.close()

    optimal_policy=[]
    for i in range(0,WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0,WORLD_WIDTH):
            if [i,j]==goal:
                optimal_policy[-1].append('G')
                continue
            best_action = np.argmax(q_value[i, j, :])
            if best_action == UP:
                optimal_policy[-1].append('U')
            elif best_action == DOWN:
                optimal_policy[-1].append('D')
            elif best_action == LEFT:
                optimal_policy[-1].append('L')
            if best_action == RIGHT:
                optimal_policy[-1].append('F')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))

if __name__ == '__main__':
    sarsa()


