import matplotlib.pyplot as plt
import d2l
from mxnet import autograd,nd,gluon,init
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
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


if (__name__=='__main__'):
    true_w=nd.array([2,-3.4])
    true_b=4.2
    features,labels=synthetic_data(true_w,true_b,1000)

    batch_size=10
    data_iter=load_array((features,labels),batch_size)

    net=nn.Sequential()
    net.add(nn.Dense(1))

    net.initialize(init.Normal(sigma=0.01))
    loss=gloss.L2Loss() #The squared loss is known as the L2 norm loss
    trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

    num_epochs=3
    for epoch in range(1,num_epochs+1):
        for X,y in data_iter:
            with autograd.record():
                l=loss(net(X),y)
            l.backward()
            trainer.step(batch_size)
        l=loss(net(features),labels)
        print('epoch %d,loss: %f'%(epoch,l.mean().asnumpy()))

    print('After regression, w is ',net[0].weight.data())
    print('After regression, b is ',net[0].bias.data())






