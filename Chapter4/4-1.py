#计算损失函数时必须将所有的训练数据作为对象。也就是说，如果训练数据 有 100 个的话，我们就要把这 100 个损失函数的总和作为学习的指标。
#神经网络的学习也是从训练数据中选出一批数据(称为mini-batch,小 批量)，然后对每个 mini-batch 进行学习。比如，从 60000 个训练数据中随机 选择 100 笔，再用这 100 笔数据进行学习。这种学习方式称为 mini-batch 学习。
import numpy as np
import tensorflow as tf
import sys,os
sys.path.append(os.pardir)
from book_dir.dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-7))/batch_size
#在进行神经网络的学习时，不能将识别精度作为指标。因为如果以识别精度为指标，则参数的导数在绝大多数地方都会变为 0。
#识别精度对微小的参数变化基本上没有什么反应，即便有反应，它的值也是不连续地、突然地变化。
#出于相同的原因，如果使用阶跃函数作为激活函数，神经网络的学习将无法进行。
#如果使用了阶跃函数，那么即便将损失函数作为指标，参数的微 小变化也会被阶跃函数抹杀，导致损失函数的值不会产生任何变化。

