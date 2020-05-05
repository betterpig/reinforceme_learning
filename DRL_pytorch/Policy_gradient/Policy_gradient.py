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
import numpy as np
import matplotlib.pyplot as plt
import os

GAMMA = 0.9 # discount factor
t.set_default_tensor_type(t.DoubleTensor)

class Policy_net(t.nn.Module):
    def __init__(self,in_features, hidden_features, out_features):
        t.nn.Module.__init__(self)
        self.input_hidden = t.nn.Linear(in_features, hidden_features) # 输入层到隐藏层
        self.hidden_preference = t.nn.Linear(hidden_features, out_features)#隐藏层到输出层
    def forward(self,x):#前向传播
        x = self.input_hidden(x)
        x = t.relu(x)
        preference=self.hidden_preference(x)
        m=t.nn.Softmax()
        action_probabilities=m(preference)
        return action_probabilities

class Policy_gradient():
  # DQN Agent
  def __init__(self, env):
    self.time_step = 0  # init some parameters
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n
    self.states=[]
    self.actions=[]
    self.rewards=[]

    self.net=Policy_net(self.state_dim,20,self.action_dim)
    self.optimizer = t.optim.Adam(params=self.net.parameters(), lr=0.01)

  def learn(self):
    returns=np.zeros_like(self.rewards)
    return_t=0
    for step in reversed(range(0,len(self.rewards)-1)):
        return_t=GAMMA*return_t+self.rewards[step]
        returns[step]=return_t
    returns=(returns-np.mean(returns))/np.std(returns)

    returns=t.tensor(returns)
    states=t.tensor(self.states)
    actions=t.zeros((len(self.actions),2))
    for i in range(len(self.actions)):
        actions[i][self.actions[i]]=1

    action_probabilities=self.net(states)
    neg_log_prob=t.sum(-t.log(action_probabilities)*actions,1)
    loss=t.sum(neg_log_prob*returns)

    self.optimizer.zero_grad()
    loss.backward()
    #查看step前后 参数的data和grad
    #print(self.net.hidden_preference.weight.data,self.net.hidden_preference.weight.grad)
    self.optimizer.step()
    #print(self.net.hidden_preference.weight.data)

    self.rewards.clear()
    self.states.clear()
    self.actions.clear()
    return loss

  def choose_action(self,state):
    state=t.tensor(state).view(1,-1)
    action_probabilities=self.net(state)
    action=t.distributions.Categorical(action_probabilities).sample()
    return action.item()

  def store_transition(self,state,action,reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE =1000 # Episode limitation
#STEP = 300 # Step limitation in an episode 300步没啥用，坚持到了200步done为true自然就结束episode了
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME)  #生成环境
  agent = Policy_gradient(env)

  steps_per=[]
  losses=[]
  count=0
  test_num=0
  is_converge=False
  
  for episode in range(EPISODE):
    if test_num>10:
      break
    
    state = env.reset() # initialize task
    steps=0
    episode_loss=0
    while not is_converge:
        env.render()    # 刷新环境
        action = agent.choose_action(state) # e-greedy action for train

        next_state,reward,done,_ = env.step(action)
        agent.store_transition(state,action,reward)    

        '''x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2 +1  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率'''
        
        state = next_state
        
        steps=steps+1   
        if done:
            episode_loss=agent.learn()
            break

    if not is_converge:
        if steps==200:
            count=count+1
            if count>25 :
                is_converge=True
        else:
            count=0
        losses.append(episode_loss/(steps+1))  
        steps_per.append(steps)
        print('episode:',episode,'  steps ：',steps)
    # Test every 100 episodes
    if (episode+1) % 30 == 0:  #测试10次的平均reward，采用target policy
      if is_converge:
        test_num=test_num+1

      total_reward = 0
      for i in range(TEST):
        state = env.reset()
        while True:
          env.render()
          action = agent.choose_action(state) # direct action for test
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
  t.save(agent.net,script_dir+'\\net_model.pkl')

if __name__ == '__main__':
  main()