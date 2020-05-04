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
import copy
import os

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
UPDATE_FREQUENCY=10

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_memory = []
        ISWeights = []
        b_idx = []
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights.append(np.power(prob/min_prob, -self.beta))
            b_idx.append(idx)
            b_memory.append(data)
        return b_idx,b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors.detach().numpy(), self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Q_net(nn.Module):
    def __init__(self,in_features, hidden_features, out_features):
        nn.Module.__init__(self)
        self.layer1 = nn.Linear(in_features, hidden_features) # 输入层到隐藏层
        self.layer2 = nn.Linear(hidden_features, out_features)#隐藏层到输出层
    def forward(self,x):#前向传播
        x = self.layer1(x)
        x = t.relu(x)
        return self.layer2(x)

class Nature_DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.memory = Memory(capacity=REPLAY_SIZE)  #经验池
    self.replay_total = 0
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

    self.current_net=Q_net(self.state_dim,20,self.action_dim)#定义网络及优化器和损失函数
    self.optimizer = optim.SGD(params=self.current_net.parameters(), lr=0.01)
    self.target_net=Q_net(self.state_dim,20,self.action_dim)

  def perceive(self,state,action,reward,next_state,done):
    loss=0.0
    one_hot_action = np.zeros(self.action_dim)  #将动作变为[0,1] [1,0]形式
    one_hot_action[action] = 1
    self.memory.store((state,one_hot_action,reward,next_state,done))#将S A R S A存入经验池
    self.replay_total+=1

    if self.replay_total > BATCH_SIZE:#经验池已有样本超过size就开始采样训练
      loss=self.train_Q_network()

    return loss

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    tree_idx, minibatch, ISWeights = self.memory.sample(BATCH_SIZE)
    ISWeights=t.from_numpy(np.array(ISWeights)).view(BATCH_SIZE).float()
    state_batch = t.from_numpy(np.array([data[0] for data in minibatch])).view(BATCH_SIZE,-1).float()
    action_batch =t.from_numpy( np.array([data[1] for data in minibatch])).view(BATCH_SIZE,-1).float()
    reward_batch = t.from_numpy(np.array([data[2] for data in minibatch])).view(BATCH_SIZE,-1).float()
    next_state_batch = t.from_numpy(np.array([data[3] for data in minibatch])).view(BATCH_SIZE,-1).float()

    #step 2:calculate current #计算实际值
    y_2=self.current_net(state_batch)
    max_action=y_2.argmax(dim=1)
    y_currrent=y_2*action_batch
    y_currrent=y_currrent.sum(1)
    
    # Step 3: calculate target 计算目标值
    y_target = t.FloatTensor()
    Q_value_batch = self.target_net(next_state_batch)  #利用网络计算下一个状态的价值
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_target=t.cat((y_target,reward_batch[i]),0)
      else :  #下一状态价值*discount再加上reward就得到目标值
        action=max_action[i]
        y_target=t.cat((y_target,reward_batch[i] + GAMMA * Q_value_batch[i][action]),0)

    #反向传播更新参数
    self.optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()  
    y_err=y_target.detach()- y_currrent
    loss = ISWeights * y_err**2
    loss=loss.sum(0)
    loss.backward()#反向传播得到梯度

    #查看step前后 参数的data和grad
    #print(self.current_net.layer2.weight.data,self.current_net.layer2.weight.grad)
    self.optimizer.step()
    #print(self.current_net.layer2.weight.data)
    self.memory.batch_update(tree_idx, t.abs(y_err))
    return loss

  def update_target_net(self):
    self.target_net=copy.deepcopy(self.current_net)

  def egreedy_action(self,state):
    state=t.from_numpy(state) #先把np转化成Tensor
    state=state.float().view(1,4)
    Q_value = self.current_net(state) #计算该状态的每个动作的价值
    Q_value=Q_value.detach().numpy()

    if self.epsilon>0.2:
        self.epsilon *= 0.9999#epsilon随着迭代不断减小，使其更加接近target policy
    elif self.epsilon>0.01:
        self.epsilon *= 0.999
    if random.random() <= self.epsilon:
        return random.randint(0,self.action_dim - 1)
    else:
        return np.argmax(Q_value)

  def action(self,state):
    state=t.from_numpy(state).float().view(1,4)
    Q_value=self.target_net(state) 
    action=t.argmax(Q_value)  #只取价值最大的动作，没有随机的可能
    return action.item()

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE =1000 # Episode limitation
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME)  #生成环境
  agent = Nature_DQN(env)
   
  steps_per=[]
  losses=[]
  count=0
  test_num=0
  is_converge=False
  for episode in range(EPISODE):
    if test_num>10:
      break
    
    state = env.reset()# initialize task
    steps=0
    episode_loss=0
    while not is_converge:
      env.render()    # 刷新环境
      action = agent.egreedy_action(state) # e-greedy action for train
      if count>20 :
        is_converge=True
        agent.update_target_net()
        break
      
      next_state,reward,done,_ = env.step(action)

      #x, x_dot, theta, theta_dot = next_state
      #r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
      #r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
      #reward = r1 + r2 +1  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
      # Define reward for agent
      reward = -1 if done else reward
      loss=agent.perceive(state,action,reward,next_state,done)
      state = next_state
      
      steps=steps+1
      episode_loss=episode_loss+loss
      
      if done:
        break

    if not is_converge:
      if steps==200:
        count=count+1
      else:
        count=0
      losses.append(episode_loss/(steps+1))  
      steps_per.append(steps)
      print('episode:',episode,'  steps ：',steps)

    if (episode+1) % UPDATE_FREQUENCY == 0:
      agent.update_target_net()

    # Test every 100 episodes
    if (episode+1) % 30 == 0:  #测试10次的平均reward，采用target policy
      if is_converge:
        test_num=test_num+1

      total_reward = 0
      for i in range(TEST):
        state = env.reset()
        while True:
          env.render()
          action = agent.action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
  
  fig=plt.figure()
  ax1=fig.add_subplot(111)
  ax1.plot(steps_per,'b')
  ax1.set_ylabel('steps in each episode')
  ax1.set_title("leaning curves")

  ax2 = ax1.twinx()
  ax2.plot(losses,'r')
  ax2.set_ylabel('average loss of steps in each episode')
  #plt.show()

  script_path = os.path.realpath(__file__)
  script_dir = os.path.dirname(script_path)
  path=script_dir+'\\learning_curve.png'
  plt.savefig(path,dpi=1200)
  t.save(agent.target_net,script_dir+'\\net_model.pkl')
if __name__ == '__main__':
  main()