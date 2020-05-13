import gym
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
import copy
os.environ["OMP_NUM_THREADS"] = "1"

MAX_EPISODE = 3000
UPDATE_FREQUENCY = 5
GAMMA = 0.9
ENTROPY_BETA = 0.001
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test 

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
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.actor_net=Policy_net(self.state_dim,20,self.action_dim)
        self.actor_optim = t.optim.Adam(params=self.actor_net.parameters(), lr=0.01)
        self.critic_net=Q_net(self.state_dim,20,1)
        self.critic_optim=t.optim.SGD(params=self.critic_net.parameters(), lr=0.01)

    def choose_action(self,state):
        action_probabilities=self.actor_net(t.tensor(state))
        action=t.distributions.Categorical(action_probabilities).sample()
        return action.item()

def push(worker_AC):
    global global_AC
    for global_param,local_param in zip(global_AC.critic_net.parameters(),worker_AC.critic_net.parameters()):
        global_param.grad=local_param.grad
    for global_param,local_param in zip(global_AC.actor_net.parameters(),worker_AC.actor_net.parameters()):
        global_param.grad=local_param.grad
    #查看step前后 参数的data和grad
    #print(self.critic_net.layer2.weight.data,self.critic_net.layer2.weight.grad)
    global_AC.critic_optim.step()
    global_AC.actor_optim.step()
    #print(self.critic_net.layer2.weight.data)

def pull(worker_AC):
    global global_AC
    for global_param,local_param in zip(global_AC.critic_net.parameters(),worker_AC.critic_net.parameters()):
        local_param.data=global_param.data
    for global_param,local_param in zip(global_AC.actor_net.parameters(),worker_AC.actor_net.parameters()):
        local_param.data=global_param.data
    
    

class Worker():
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.AC = Actor_critic(env)
        pull(self.AC)
        global global_AC

    def compute_gradients(self,done,states,actions,rewards):
        target_values=[]
        if done:
            last_target_value=0
        else:
            last_target_value=self.AC.critic_net(t.tensor(states[-1]))
        target_values.append(last_target_value)
        for step in reversed(range(0,len(rewards)-1)):
            target_value_t=GAMMA*target_value_t+rewards[step]
            target_values[step]=target_value_t

        current_values=self.AC.critic_net(states)
        td_errors=target_values-current_values
        critic_loss=t.sum(td_errors**2)

        action_probabilities=self.AC.actor_net(t.tensor(states))
        neg_log_prob=-t.log(action_probabilities[actions])
        neg_expect=t.sum(-neg_log_prob*td_errors,dim=1)

        self.AC.actor_optim.zero_grad()
        self.AC.critic_optim.zero_grad()
        critic_loss.backward()
        neg_expect.backward()

    def work(self):
        global episodes
        states=[]
        actions=[]
        rewards=[]
        total_steps=0
        while episodes<MAX_EPISODE :
            if test_num>10:
                break
            episodes=episodes+1
            state=self.env.reset()
            steps=0
            while not is_converge:
                action = self.AC.choose_action(state)
                next_state,reward,done,_ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                steps=steps+1
                total_steps=total_steps+1
            print('episode:',episodes,' steps:',steps)
            if (episodes+1) % UPDATE_FREQUENCY==0:
                self.compute_gradients(done,states,actions,rewards)
                push(self.AC)
                pull(self.AC)
                states.clear()
                actions.clear()
                rewards.clear()
            episode_steps.append(steps)
            if steps==200:
                count=count+1
                if count>25 :
                    is_converge=True
            else:
                count=0
            
            if (episodes+1) % 50 == 0:  #测试10次的平均reward，采用target policy
                if is_converge:
                    test_num=test_num+1

                total_reward = 0
                for i in range(TEST):
                    state = env.reset()
                    while True:
                        self.env.render()
                        action = global_AC.choose_action(state) # direct action for test
                        state,reward,done,_ = env.step(action)
                        total_reward += reward
                        if done:
                            break
                ave_reward = total_reward/TEST
                print ('episode: ',episodes,'Evaluation Average Reward:',ave_reward)

# ---------------------------------------------------------
    
episode_steps=[]
episodes=0
count=0
test_num=0
is_converge=False
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    global_AC=Actor_critic(env)
    
    
    workers = [Worker() for i in range(3)]
    for i in range(3):
        t=threading.Thread(target=workers[i].work)
        t.start()
        t.join()

    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.plot(episode_steps,'b')
    ax1.set_ylabel('steps in each episode')
    ax1.set_title("leaning curves")

    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    path=script_dir+'\\learning_curve.png'
    plt.savefig(path,dpi=1200)
    t.save(global_AC.actor_net,script_dir+'\\actor_net_model.pkl')
    t.save(global_AC.critic_net,script_dir+'\\critic_net_model.pkl')