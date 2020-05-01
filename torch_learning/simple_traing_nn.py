import torch as t
import numpy as np
from torch import nn
from torch import  optim
import matplotlib.pyplot as plt
from IPython import display

point_num=30
x=t.arange(0,point_num,1,dtype=t.float).view(point_num,1)
x_norm=(x-x.mean(0))/x.std()
y=x_norm**2+t.randn(x.size())/10
#y=x**2+t.randn(point_num,1,dtype=t.float)
class Net(nn.Module):
    def __init__(self,input_num,hidden_num,output_num):
        super(Net,self).__init__()
        self.input_hidden=nn.Linear(input_num,hidden_num)
        #self.hidden1_hidden2=nn.Linear(hidden1_num,hidden2_num)
        self.hidden_output=nn.Linear(hidden_num,output_num)

    def forward(self,x):
        z2=self.input_hidden(x)
        a2=t.relu(z2)
        z3=self.hidden_output(a2)
        return z3

net=Net(1,2,1)
opt=t.optim.SGD(net.parameters(),lr=0.1)
lossfunc=t.nn.MSELoss()

plt.ion()  # 动态学习过程展示  

losses=[]
for i in range(0,300):
    y_predict=net(x_norm)
    loss=lossfunc(y_predict,y)
    losses.append(loss)
    if i%20==0:
        plt.cla()
        plt.scatter(x_norm,y)
        plt.scatter(x_norm,y_predict.detach().numpy())
        plt.text(0.2, 0, 'L=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'}) 
        plt.show()
        plt.pause(1)
        #plt.close()
    opt.zero_grad()
    loss.backward()
    
    y_err=y_predict-y
    y_err_mean=y_err.sum(0)
    print(net.hidden_output.weight.data, net.hidden_output.weight.grad)
    #for name, param in net.named_parameters():
        #print(name, param.data,param.grad)

    opt.step()
    #for name, param in net.named_parameters():
        #print(name, param.data)
    #print(net.hidden_output.bias.item())
    print(net.hidden_output.weight.data)
    

