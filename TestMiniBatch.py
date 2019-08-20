import matplotlib.pyplot as plt
import d2l
from mxnet import autograd,nd
import random

def synthetic_data(w,b,num_examples):
    """generate y=Xw+b+noise"""
    X=nd.random.normal(scale=1,shape=(num_examples,len(w)))
    y=nd.dot(X,w)+b
    y+=nd.random.normal(scale=0.01,shape=y.shape)
    return X,y

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)
        # The “take” function will then return the corresponding element
        # based on the indices

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

    ##debug: plot synthetic data
    #fig,axs=plt.subplots(1,len(true_w),figsize=[3.5,2.5],sharey=True)
    #fig.suptitle('synthetic data')
    #for i in range(len(true_w)):
    #    axs[i].scatter(features[:,i].asnumpy(),labels.asnumpy(),1)
    #plt.show()

    #random init
    w=nd.random.normal(scale=0.01,shape=(2,1))
    b=nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()

    learning_rate=0.03
    num_epochs=3 #number of iterations

    net=linreg
    loss=squared_loss
    batch_size=10

    for epoch in range(num_epochs):
        for X,y in data_iter(batch_size,features,labels):
            with autograd.record():
                l=loss(net(X,w,b),y)
            l.backward() #compute gradient on l with respect to [w,b]
            sgd([w,b],learning_rate,batch_size) #update parameters using their gradient
        train_l=loss(net(features,w,b),labels)
        print('epoch %d,loss %f'%(epoch+1,train_l.mean().asnumpy()))

    print('After regression, w is ',w)
    print('After regression, b is ',b)



