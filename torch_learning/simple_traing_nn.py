import torch as t
import numpy as np
from torch import nn
from torch import  optim
import matplotlib.pyplot as plt
from IPython import display

point_num=30
x=t.arange(0,point_num,1,dtype=t.float).view(point_num,1)
x_norm=(x-x.mean(0))/x.std()
#如果x的范围很大，要先归一化，否则不能收敛，原因如下
#比如x从1到100，那y=x^2就是1到10000.如果一开始w和b都是0，那y_predict就是0，但不同
#的x，y_err是从1到10000，而w=w-lr*loss*x，若lr也很大的话，更新量相对于w来说就很大
#这样的话w就变化很大，只能一直震荡
#所以要求lr loss 和x三者都要在合适范围，最方便的就是先让x归一化在-1到1之间，那y_err->loss
#值也不会太大，然后单步跟中查看backward之后的grad，看与w和b的值相比会不会太大或太小，
#超过十分之一就太大，很难收敛，小于千分之一就太小，学习太慢。lossfunc的选取也有影响，
#比如MSELoss()取不取平均，就会使得loss值变大变小，也会影响能否收敛和收敛快慢

#所以以后一定要检查step前后参数的data和grad，以及loss和y_err等量取值正不正常，这是
#能否收敛的关键
y=x_norm**2+t.randn(x.size())/10
#y=x**2+t.randn(point_num,1,dtype=t.float)
class Net(nn.Module):
    def __init__(self,input_num,hidden_num,output_num):
        super(Net,self).__init__()
        self.input_hidden=nn.Linear(input_num,hidden_num)#输入层到隐藏层
        self.hidden_output=nn.Linear(hidden_num,output_num)#隐藏层到输出层

    def forward(self,x):
        z2=self.input_hidden(x)
        a2=t.relu(z2)#激活单元
        z3=self.hidden_output(a2)
        return z3

net=Net(1,5,1)#定义网络对象
opt=t.optim.SGD(net.parameters(),lr=0.1)#虽然名字叫SGD，但并不是随机梯度下降，是
#批量梯度下降，就是会把所有样本的y_err*x再求和，然后作为梯度
lossfunc=t.nn.MSELoss()#如果没指定mean，就只是求平方和

plt.ion()  # 动态学习过程展示  

losses=[]
for i in range(0,300):
    y_predict=net(x_norm)#计算预测量
    loss=lossfunc(y_predict,y)#计算loss
    losses.append(loss)
    if i%20==0:
        plt.cla()#先清除同一个figure上次遗留的内容
        plt.scatter(x_norm,y)
        plt.scatter(x_norm,y_predict.detach().numpy())
        plt.text(0.2, 0, 'L=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'}) 
        plt.show()#显示
        plt.pause(1)#暂停
    opt.zero_grad()#先清除上一次backward得到的梯度
    loss.backward()#反向传播得到梯度
    
    y_err=y_predict-y
    y_err_mean=y_err.sum(0)#查看step前后的参数的值变化，正常情况下应该是data=data-grad*lr
    print(net.hidden_output.weight.data, net.hidden_output.weight.grad)
    #for name, param in net.named_parameters():
        #print(name, param.data,param.grad)

    opt.step()#每个参数减去相应的梯度
    print(net.hidden_output.weight.data)
    

