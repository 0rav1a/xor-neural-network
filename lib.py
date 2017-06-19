#coding:utf-8
from math import e

def sigmoid(x):
    return 1/(1+e**-x)
    
def gauss(x, mu = 0):
    return e**((-(x-mu)**2)*4)