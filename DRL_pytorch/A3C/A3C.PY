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
import threading
import copy

GAME = 'CartPole-v0'
N_WORKERS = 3
MAX_EPISODE = 3000
UPDATE_FREQUENCY = 100
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_ACTOR = 0.001    # learning rate for actor
LR_CRITIC = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
STEP = 3000 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

episode = 0
episode_steps=[]

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

t.set_default_tensor_type(t.DoubleTensor)

class Q_net(t.nn.Module):
    def __init__(self,in_features, hidden_features, out_features):
        t.nn.Module.__init__(self)
        self.layer1 = t.nn.Linear(in_features, hidden_features) # 输入层到隐藏层
        self.layer2 = t.nn.Linear(hidden_features, out_features)#隐藏层到输出层
    def forward(self,x):#前向传播
        x = self.layer1(x)
        x = t.relu(x)
        return self.layer2(x)

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

class Actor_critic():
  # DQN Agent
    def __init__(self, env):
        self.time_step = 0  # init some parameters
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.actor_net=Policy_net(self.state_dim,20,self.action_dim)
        self.actor_optim = t.optim.Adam(params=self.actor_net.parameters(), lr=0.01)

        self.critic_net=Q_net(self.state_dim,20,1)
        self.critic_optim=t.optim.SGD(params=self.critic_net.parameters(), lr=0.01)

    def train_actor(self,state,action,td_error):
        one_hot_action = t.zeros(self.action_dim)
        one_hot_action[action] = 1
        action_probabilities=self.actor_net(t.tensor(state))
        neg_log_prob=t.sum(-t.log(action_probabilities)*one_hot_action)
        neg_expect=-neg_log_prob.view(1)*td_error.item()

        self.actor_optim.zero_grad()
        neg_expect.backward()
        #查看step前后 参数的data和grad
        #print(self.actor_net.hidden_preference.weight.data,self.actor_net.hidden_preference.weight.grad)
        self.actor_optim.step()
        #print(self.actor_net.hidden_preference.weight.data)

        return neg_expect
  
    def train_critic(self,state,reward,next_state):
        current_value = self.critic_net(t.tensor(state))
        target_value = reward + GAMMA * self.critic_net(t.tensor(next_state))
        td_error=target_value-current_value

        loss=td_error**2
        self.critic_optim.zero_grad()
        loss.backward()
        #查看step前后 参数的data和grad
        #print(self.critic_net.layer2.weight.data,self.critic_net.layer2.weight.grad)
        self.critic_optim.step()
        #print(self.critic_net.layer2.weight.data)
        return td_error

    def choose_action(self,state):
        action_probabilities=self.actor_net(t.tensor(state))
        action=t.distributions.Categorical(action_probabilities).sample()
        return action.item()

    def update(self,states,actions,advantages):
        td_errors=self.train_critic(state,reward,next_state) 
        self.train_actor(state,action,td_errors)

class Worker():
    def __init__(self,name,global_AC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = Actor_critic(name, globalAC)

    def work(self):
        global episode,episode_steps
        states=[]
        actions=[]
        rewards=[]
        total_steps=0
        while episode_steps<MAX_EPISODE:
            state=self.env.reset()
            steps=0
            while True:
                action = agent.choose_action(state)
                next_state,reward,done,_ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                steps=steps+1
                total_steps=total_steps+1
                if total_steps % UPDATE_FREQUENCY==0:
                    advantages=np.zeros(len(self.rewards))
                    if done:
                        last_advantage=0
                    else:
                        last_advantage=self.AC.critic_net(t.tensor(state[-1]))
                    advantages.append(last_advantage)
                    for step in reversed(range(0,len(self.rewards)-1)):
                        return_t=GAMMA*return_t+self.rewards[step]
                        returns[step]=return_t

                    self.AC.update(states,actions,advantages)
                    self.AC.pull()
                    states.clear()
                    actions.clear()
                    rewards.clear()

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE =2000 # Episode limitation
#STEP = 300 # Step limitation in an episode 300步没啥用，坚持到了200步done为true自然就结束episode了
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)  #生成环境
    agent = Actor_critic(env)

    steps_per=[]
    count=0
    test_num=0
    is_converge=False
    
    for episode in range(EPISODE):
        if test_num>10:
            break
    
        state = env.reset() # initialize task
        steps=0
        while not is_converge:
            env.render()    # 刷新环境
            action = agent.choose_action(state) 
            next_state,reward,done,_ = env.step(action)
            agent.learn(state,action,reward,next_state)
        
        '''x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2 +1  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率'''
        
            state = next_state
            steps=steps+1   
            if done:
                break

        if not is_converge:
            if steps==200:
                count=count+1
                if count>25 :
                    is_converge=True
            else:
                count=0
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

    #plt.show()

    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    path=script_dir+'\\learning_curve.png'
    plt.savefig(path,dpi=1200)
    t.save(agent.actor_net,script_dir+'\\actor_net_model.pkl')
    t.save(agent.critic_net,script_dir+'\\critic_net_model.pkl')

if __name__ == '__main__':
  main()