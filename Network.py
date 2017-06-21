#coding:utf-8
import random
from Neuron import Neuron
from lib import gauss, sigmoid

LEARNING_RATE = 1 #Para gradient descent
ERROR = 0.01 #Margen de error de la red a partir del cual se considerará entrenada

class Network:
    def __init__(self, nNeurons = [1, 1, 1], layers = None):
        '''
        nNeurons: Número de neuronas que habrá en cada capa. La longitud de esta lista es el número de capas de la red.
        layers: Conjunto ya creado de capas (se usa al clonar una red)
        '''
        self.layers = layers
        self.nNeurons = nNeurons
        
        if (not layers):
            self.layers = [[] for _ in nNeurons]
            
            for _ in range(nNeurons[0]): #Capa input
                self.layers[0].append(Neuron([], self.layers[1]))
    
            for l in range(len(nNeurons))[1:-1]:
                for _ in range(nNeurons[l]): #Capas hidden
                    self.layers[l].append(Neuron(self.layers[l-1], self.layers[l+1]))
                    
            for _ in range(nNeurons[-1]): #Capa output
                self.layers[-1].append(Neuron(self.layers[-2], []))
    
    def setInputs(self, inputs):
        for i in range(self.nNeurons[0]):
            self.layers[0][i].setBias(inputs[i])
                
    def getOutputs(self, inputs):
        self.setInputs(inputs)
        return [neuron.getOutput() for neuron in self.layers[-1]]
            
    def backprop(self, tests, lr = LEARNING_RATE):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        tests: Listas de pares [entrada, salida esperada] para entrenar la red
        '''
        while self.calcError(tests) > ERROR:
            for test in tests: #Para cada iteración se realizan todos los tests
                inputs = test[0]
                targets = {self.layers[-1][o]: test[1][o] for o in range(len(self.layers[-1]))}
                self.setInputs(inputs)
                
                for layer in self.layers[1:]:
                    for neuron in layer:
                        neuron.update(targets, lr)
                        
    def calcError(self, tests):
        '''Calcula el error de la red probando los tests
        '''
        self.error = 0.0
        for inputs, targets in tests:
            outputs = self.getOutputs(inputs)
            for o in range(self.nNeurons[-1]):
                self.error += abs(outputs[o] - targets[o]) #1-gauss(outputs[o], targets[o])
            
            #print outputs,
        
        self.error /= len(tests)*self.nNeurons[-1]

        print "\nError:\t" + str(self.error)
        return self.error