#coding:utf-8
import random
from Network import Network

XOR = (([0,0],[0]),
       ([0,1],[1]),
       ([1,0],[1]),
       ([1,1],[0]))
         
CONT = (([0,0,0],[0,0,1]),
        ([0,0,1],[0,1,0]),
        ([0,1,0],[0,1,1]),
        ([0,1,1],[1,0,0]),
        ([1,0,0],[1,0,1]),
        ([1,0,1],[1,1,0]),
        ([1,1,0],[1,1,1]),
        ([1,1,1],[0,0,0]))

test = []
for _ in range(1000):
    n1 = random.uniform(0, 1)
    n2 = random.uniform(0, 1)
    test.append(([n1,n2],[n1*n2]))
    
net = Network([3,3,3])
net.backprop(CONT)

'''
net.setInputs([0.428291, 0.859212])
net.calcOutputs()
print net.getOutputs()

net.setInputs([1, 0.628231])
net.calcOutputs()
print net.getOutputs([1, 0.628231])
'''