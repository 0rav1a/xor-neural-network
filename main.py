#coding:utf-8
import random
import numpy as np
from Network import Network

XOR = np.array([[[0,0],[0]],
                 [[0,1],[1]],
                 [[1,0],[1]],
                 [[1,1],[0]]])

CONT = np.array([[[0,0,0],[0,0,1]],
                 [[0,0,1],[0,1,0]],
                 [[0,1,0],[0,1,1]],
                 [[0,1,1],[1,0,0]],
                 [[1,0,0],[1,0,1]],
                 [[1,0,1],[1,1,0]],
                 [[1,1,0],[1,1,1]],
                 [[1,1,1],[0,0,0]]])

test = []
for _ in range(10000):
    n1 = random.uniform(0, 1)
    n2 = random.uniform(0, 1)
    test.append(([n1,n2],[n1*n2]))

net = Network([2,2,1])
net.backprop(test)
net.setInputs([0.5124, 0.1925])
net.calcOutputs()
print net.getOutputs()

'''
net.setInputs([0, 0])
net.calcOutputs()
print net.getOutputs()
net.setInputs([0, 1])
net.calcOutputs()
print net.getOutputs()
net.setInputs([1, 0])
net.calcOutputs()
print net.getOutputs()
net.setInputs([1, 1])
net.calcOutputs()
print net.getOutputs()
'''

'''
net.setInputs([0.428291, 0.859212])
net.calcOutputs()
print net.getOutputs()

net.setInputs([1, 0.628231])
net.calcOutputs()
print net.getOutputs([1, 0.628231])
'''
