#######################################################################
# Copyright (C)                                                       #
# 2016 - 2019 Pinard Liu(liujianping-ok@163.com)                      #
# https://www.cnblogs.com/pinard                                      #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
##https://www.cnblogs.com/pinard/p/9714655.html ##
## 强化学习（八）价值函数的近似表示与Deep Q-Learning ##

import gym
import torch as t
from torch import nn
from torch import  optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch


class Q_net(nn.Module):
    def __init__(self,in_features, hidden_features, out_features):
        nn.Module.__init__(self)
        self.layer1 = nn.Linear(in_features, hidden_features) # 此处的Linear是前面自定义的全连接层
        self.layer2 = nn.Linear(hidden_features, out_features)
    def forward(self,x):
        x = self.layer1(x)
        x = t.relu(x)
        return self.layer2(x)


class DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

    self.q_net=Q_net(self.state_dim,20,self.action_dim)
    self.optimizer = optim.SGD(params=self.q_net.parameters(), lr=0.05)
    self.criterion = nn.MSELoss()


  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network()

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = t.from_numpy(np.array([data[0] for data in minibatch])).view(BATCH_SIZE,-1).float()
    action_batch =t.from_numpy( np.array([data[1] for data in minibatch])).view(BATCH_SIZE,-1).float()
    reward_batch = t.from_numpy(np.array([data[2] for data in minibatch])).view(BATCH_SIZE,-1).float()
    next_state_batch = t.from_numpy(np.array([data[3] for data in minibatch])).view(BATCH_SIZE,-1).float()

    # Step 2: calculate target
    y_target = t.FloatTensor()
    Q_value_batch = self.q_net(next_state_batch)
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_target=t.cat((y_target,reward_batch[i]),0)
      else :
        y_target=t.cat((y_target,reward_batch[i] + GAMMA * t.max(Q_value_batch[i])),0)

    #step 3:calculate current
    y_2=self.q_net(state_batch)
    y_currrent=y_2*action_batch
    y_currrent=y_currrent.sum(1)

    #反向传播更新参数
    
    self.optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()   
    
    loss = self.criterion( y_currrent,y_target)
    loss.backward()

    #print(self.q_net.layer2.weight.data,self.q_net.layer2.weight.grad)

    self.optimizer.step()

    #print(self.q_net.layer2.weight.data)




  def egreedy_action(self,state):
    state=t.from_numpy(state)
    state=state.float().view(1,4)
    Q_value = self.q_net(state)
    Q_value=Q_value.detach().numpy()
    if self.epsilon>0.1:
      self.epsilon *= 0.999
    if random.random() <= self.epsilon:
        return random.randint(0,self.action_dim - 1)
    else:
        return np.argmax(Q_value)

  def action(self,state):
    state=t.from_numpy(state).float().view(1,4)
    Q_value=self.q_net(state)
    action=t.argmax(Q_value)
    return action.item()

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE =3000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME)
  agent = DQN(env)

  total_steps=0
  steps_ave=[]
  steps_per=[]
  for episode in range(EPISODE):
    # initialize task
    state = env.reset()
    # Train
    steps=0
    for step in range(STEP):
      env.render()    # 刷新环境
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done,_ = env.step(action)

      #x, x_dot, theta, theta_dot = next_state
      #r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
      #r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
      #reward = r1 + r2 +1  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
      # Define reward for agent
      reward = -1 if done else reward
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
      if done:
        break
      steps=steps+1
    total_steps=total_steps+steps
    steps_per.append(steps)
    steps_ave.append(total_steps/(episode+1))
    print('episode:',episode,'  steps ：',steps)
    # Test every 100 episodes
    if episode % 100 == 0:
      total_reward = 0
      for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
          env.render()
          action = agent.action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
  
  plt.plot(steps_ave)
  plt.plot(steps_per)
  plt.show()
  plt.savefig('./steps.jpg')

if __name__ == '__main__':
  main()