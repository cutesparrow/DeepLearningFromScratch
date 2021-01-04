#gradient
import numpy as np
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        temp_val = x[idx]
        x[idx] = temp_val + h
        fxh1 = f(x)
        x[idx] = temp_val -h
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = temp_val
    return grad
#梯度指示的方向是各点处的函数值减小最多的方向
#实际上，在复杂的函数中，梯度指示的方向基本上都不是函数值最小处。
#因为这里使用的数据是随机选择的mini batch数据，所以又称为 随机梯度下降法(stochastic gradient descent)。
#深度学习的很多框架中，随机梯度下降法一般由一个名为 SGD 的函数来实现。 SGD 来源于随机梯度下降法的英文名称的首字母。

