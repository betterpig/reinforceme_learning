import gym
import torch as t
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time

GAMMA = 0.99 # discount factor for target Q
REPLAY_SIZE = 50000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
TAU=0.005
t.set_default_tensor_type(t.DoubleTensor)

class Q_net(t.nn.Module):
    def __init__(self):
        t.nn.Module.__init__(self)
        self.state_hidden1 = t.nn.Linear(3, 64) # 输入层到隐藏层
        self.action_hidden1 = t.nn.Linear(1, 64)
        self.hidden1_hidden2= t.nn.Linear(128,32)
        self.hidden2_output = t.nn.Linear(32,1)#隐藏层到输出层
    def forward(self,x,a):#前向传播
        x = t.relu(self.state_hidden1(x))
        a = t.relu(self.action_hidden1(a))
        cat=t.cat((x,a),dim=1)
        q = t.relu(self.hidden1_hidden2(cat))
        q=self.hidden2_output(q)
        return q

class Policy_net(t.nn.Module):
    def __init__(self):
        t.nn.Module.__init__(self)
        self.state_hidden1 = t.nn.Linear(3, 128)# 输入层到隐藏层
        self.hidden1_hidden2= t.nn.Linear(128,32) 
        self.hidden2_output = t.nn.Linear(32,1)#隐藏层到输出层
    def forward(self,x):#前向传播
        x=t.relu(self.state_hidden1(x))
        x = t.relu(self.hidden1_hidden2(x))
        mu = t.tanh(self.hidden2_output(x))*2
        return mu

class DDPG():
  # DQN Agent
    def __init__(self):
        self.replay_buffer = deque()
        self.current_actor=Policy_net()
        self.target_actor=Policy_net()
        self.target_actor.load_state_dict(self.current_actor.state_dict())
        self.actor_optim = t.optim.Adam(params=self.current_actor.parameters(), lr=0.0005)

        self.current_critic=Q_net()
        self.target_critic=Q_net()
        self.target_critic.load_state_dict(self.current_critic.state_dict())
        self.critic_optim=t.optim.Adam(params=self.current_critic.parameters(), lr=0.001)

    def store_transition(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))#将S A R S A存入经验池
        if len(self.replay_buffer) > REPLAY_SIZE: #超过经验池的容量则开始舍弃最老的
            self.replay_buffer.popleft()

    def learn(self):
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)  #采样
        state_batch = t.from_numpy(np.array([data[0] for data in minibatch])).view(BATCH_SIZE,-1)
        action_batch = t.from_numpy(np.array([data[1] for data in minibatch])).view(BATCH_SIZE,-1)
        reward_batch = t.from_numpy(np.array([data[2] for data in minibatch])).view(BATCH_SIZE,-1)
        next_state_batch = t.from_numpy(np.array([data[3] for data in minibatch])).view(BATCH_SIZE,-1)
        
        actions=self.current_actor(state_batch)
        current_values = self.current_critic(state_batch,actions)
        actor_loss=-t.mean(current_values)
        self.current_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        #print(self.current_actor.hidden2_output.weight.data,self.current_actor.hidden2_output.weight.grad)
        self.actor_optim.step()
        #print(self.current_actor.hidden2_output.weight.data)

        next_action_batch=self.target_actor(next_state_batch)
        current_values = self.current_critic(state_batch,action_batch)
        target_values = reward_batch + GAMMA * self.target_critic(next_state_batch,next_action_batch)
        #td_errors=target_values.detach()-current_values
        critic_loss=F.smooth_l1_loss(target_values.detach(),current_values)
        #critic_loss=t.sum(td_errors**2)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        #print(self.current_critic.hidden2_output.weight.data,self.current_critic.hidden2_output.weight.grad)
        self.critic_optim.step()
        #print(self.current_critic.hidden2_output.weight.data)      
    
    def update_target_net(self):
        for target_param,current_param in zip(self.target_critic.parameters(),self.current_critic.parameters()):
            target_param=TAU*current_param+(1-TAU)*target_param
        
        for target_param,current_param in zip(self.target_actor.parameters(),self.current_actor.parameters()):
            target_param=TAU*current_param+(1-TAU)*target_param


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

# ---------------------------------------------------------
# Hyper Parameters
EPISODE =5000 # Episode limitation
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
    ENV_NAME = 'Pendulum-v0'
    env = gym.make(ENV_NAME)
    agent = DDPG()

    episode_rewards=[]
    count=0
    test_num=0
    is_converge=False
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
    rewards=0
    for episode in range(EPISODE):
        if test_num>10:
            break
        #start=time.time()
        state = env.reset() # initialize task
        for step in range(300):
            #env.render()    # 刷新环境
            action = agent.current_actor(t.from_numpy(state))
            action=action.item()+ou_noise()[0]
            next_state,reward,done,_ = env.step([action])
            
            agent.store_transition(state,action,reward/100,next_state,done)
            
            state = next_state
            rewards=rewards+reward 
            if done:
                break
        
        if len(agent.replay_buffer) > 2000:
            for i in range(10):
                agent.learn()
                agent.update_target_net()

        if episode%20==0:
            print('episode:',episode,' rewards:',rewards/20)
            rewards=0
        #end=time.time()
        #a=(end-start)*1000
        #print(a)
        # Test every 100 episodes
        if (episode+1) % 200 == 0 and episode>1000:  #测试10次的平均reward，采用target policy
            if is_converge:
                test_num=test_num+1

            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                steps=0
                while steps<200:
                    env.render()
                    steps=steps+1
                    action = agent.current_actor(t.from_numpy(state)) # direct action for test
                    state,reward,done,_ = env.step([action.item()])
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
        
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    #ax1.plot(steps_per,'b')
    ax1.set_ylabel('steps in each episode')
    ax1.set_title("leaning curves")

    #plt.show()

    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    path=script_dir+'\\learning_curve.png'
    plt.savefig(path,dpi=1200)
    #t.save(agent.actor_net,script_dir+'\\actor_net_model.pkl')
    #t.save(agent.critic_net,script_dir+'\\critic_net_model.pkl')

if __name__ == '__main__':
    main()