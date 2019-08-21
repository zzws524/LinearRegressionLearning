import matplotlib.pyplot as plt
import d2l
from mxnet import autograd,nd,gluon
from mxnet.gluon import nn
import random

def synthetic_data(w,b,num_examples):
    """generate y=Xw+b+noise"""
    X=nd.random.normal(scale=1,shape=(num_examples,len(w)))
    y=nd.dot(X,w)+b
    y+=nd.random.normal(scale=0.01,shape=y.shape)
    return X,y

def load_array(data_arrays,batch_size,is_train=True):
    """construct a Gluon data loader"""
    dataset=gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset,batch_size,shuffle=is_train)


def linreg(X,w,b):
    return nd.dot(X,w)+b

def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,learning_rate,batch_size):
    for param in params:
        param[:]=param-learning_rate*param.grad/batch_size


if (__name__=='__main__'):
    true_w=nd.array([2,-3.4])
    true_b=4.2
    features,labels=synthetic_data(true_w,true_b,1000)

    batch_size=10
    data_iter=load_array((features,labels),batch_size)

    net=nn.Sequential()
    net.add(nn.Dense(1))




