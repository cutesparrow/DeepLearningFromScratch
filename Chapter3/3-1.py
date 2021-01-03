import numpy as np

def step_function(x):
    y = x>0
    return y.astype(np.int)
x = np.arange(-5,5,0.1)
import matplotlib.pylab as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1)
plt.show()
#sigmoid 函数的平滑性对神经网络的学习具有重要意义
#激活函数不能使 用线性函数
#无法发挥多层网络带来的优势。因此，为了发挥叠加层所 带来的优势，激活函数必须使用非线性函数。
def relu(x):
    return np.maximum(0,x)
#更常用的激活函数

