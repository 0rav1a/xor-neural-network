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
    n1 = random.uniform(0, 0.5)
    n2 = random.uniform(0, 0.5)
    test.append(([n1,n2],[n1+n2]))
    
net = Network(2, 1)
<<<<<<< HEAD
net.backprop(test)
=======
net.backprop(test)
>>>>>>> 5127112ee5dcf8132bc320e0819ef7a0f4bd7ed4
