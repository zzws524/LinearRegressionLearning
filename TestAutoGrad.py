import numpy as np
from mxnet import ndarray as nd
from mxnet import autograd


# Size of the points dataset.
m = 20

# Points x-coordinate and dummy value (x0, x1).
x = nd.ones((m, 1))

# Points y-coordinate
y = nd.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)

# The Learning Rate alpha.
alpha = 0.01

def error_function(X_actual,Y_actual):
    diff=nd.dot(X_actual,w)+b-Y_actual
    return (1./(2*m))*nd.dot(diff.reshape(1,m),diff)


def SGD(params, alpha):
    for param in params:
        param[:] = param - alpha * param.grad  # param[:] 代表对被遍历的 params 做原地修改 inplace update


w = nd.random_normal(shape=(1, 1))
b = nd.zeros((1,))
params = [w, b]

# 给系数列表的每个元素分配存放梯度的内存
for param in params:
    param.attach_grad()


total_loss=0
#while not np.all((np.absolute(params[0].asnumpy())))<1e-5:
for i in range(0,200):
    with autograd.record():
        total_loss=error_function(x,y)
    total_loss.backward()
    SGD(params, alpha)

print(total_loss)
print(params)
