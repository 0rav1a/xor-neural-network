#coding:utf-8
from Network import Network

#TEST = Lista de pares [entrada, salida esperada] para entrenar la red
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

test = XOR
net = Network(len(test[0][0]), len(test[0][1]))
net.backprop(test)