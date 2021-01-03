#回归问题用恒等函数，分类问题用 softmax 函数。
import numpy as np
a = np.array([0.3,2.9,4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
y = exp_a/sum_exp_a
print(y)
def softmax(a):
    exp_a = np.exp(a) 
    sum_exp_a = np.sum(exp_a) 
    y = exp_a / sum_exp_a
    return y    
#计 算 机 处 理“ 数 ”时 ，数 值 必 须 在 4 字 节 或 8 字 节 的 有 限 数 据 宽 度 内 。 这意味着数存在有效位数，也就是说，可以表示的数值范围是有 限的。因此，会出现超大值无法表示的问题。这个问题称为溢出， 在进行计算机的运算时必须(常常)注意。
def softmax2(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
#softmax 函数的输出是 0.0 到 1.0 之间的实数。并且，softmax 函数的输出值的总和是 1,上面的例子可以解释成 y[0] 的概率是 0.018(1.8 %)，y[1] 的概率 是 0.245(24.5 %)，y[2] 的概率是 0.737(73.7 %)。从概率的结果来看，可以 说“因为第 2 个元素的概率最高，所以答案是第 2 个类别”
#推理阶段一般会省 略输出层的 softmax 函数。在输出层使用 softmax 函数是因为它和 神经网络的学习有关系

    
