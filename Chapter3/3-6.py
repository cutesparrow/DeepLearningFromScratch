import tensorflow as tf
import numpy as np
import pickle
def softmax(a):
   c = np.max(a)
   exp_a = np.exp(a-c)
   sum_exp_a = np.sum(exp_a)
   y = exp_a/sum_exp_a
   return y
def getData():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test.reshape((x_test.shape[0],-1)),y_test

def initNetwork():
    with open("./sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network
def sigmoid(x):
    return 1/(1+np.exp(-x))
def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a1 = np.dot(x, W1) + b1 
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3 
    y = softmax(a3)
    return y
x,t = getData()
network = initNetwork()
accuracy_cnt = 0
for i in range(len(x)):
    y =  predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt+=1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
#使用批处理，可以实现高速 且高效的运算