#coding:utf-8
from Network import Network
from lib import gauss

N_INPUTS = 2
N_OUTPUTS = 1
ERROR = 0.0001 #Margen de error de la red a partir del cual se considerará entrenada

TESTS = (([0,0],[0]), #Pares [entrada, salida esperada] para entrenar la red
         ([0,1],[1]),
         ([1,0],[1]),
         ([1,1],[0]))
'''        
TESTS = (([0,0,0],[0,0,1]), #Pares [entrada, salida esperada] para entrenar la red
         ([0,0,1],[0,1,0]),
         ([0,1,0],[0,1,1]),
         ([0,1,1],[1,0,0]),
         ([1,0,0],[1,0,1]),
         ([1,0,1],[1,1,0]),
         ([1,1,0],[1,1,1]),
         ([1,1,1],[0,0,0]))
'''
      
def train(net):
    '''Entrena la red con todos los casos posibles hasta que el error sea el deseado
    '''
    while calcError(net) > ERROR:
    #for _ in range(20000):
        calcError(net)
        for test in TESTS:
            inputs = test[0]
            target = test[1]
            net.train(inputs, target)
    
    print net.getOutput([0,1,1])
    
def calcError(net):
    '''Calcula el error total de la red realizando las pruebas establecidas
    '''
    error = 0.0
    for input, target in TESTS:
        output = net.getOutput(input)
        for i in range(N_OUTPUTS):
            error += 1-gauss(output[i], target[i])
        
        print output,
    
    error /= len(TESTS)*N_OUTPUTS

    print "\nError:\t" + str(error)
    return error

net = Network(N_INPUTS, N_OUTPUTS)
train(net)