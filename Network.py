#coding:utf-8
import random
from Neuron import Neuron
from lib import gauss, sigmoid

LEARNING_RATE = 3 #Para gradient descent
ERROR = 0.05 #Margen de error de la red a partir del cual se considerará entrenada

class Network:
    def __init__(self, nNeurons = [1, 1, 1], layers = None):
        '''
        nNeurons: Número de neuronas que habrá en cada capa. La longitud de esta lista es el número de capas de la red.
        layers: Conjunto ya creado de capas (se usa al clonar una red)
        '''
        self.layers = layers
        self.nNeurons = nNeurons
        self.error = 1
        
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
            
    def setTargets(self, targets):
        for o in range(self.nNeurons[-1]):
            self.layers[-1][o].setTarget(targets[o])
            
    def calcOutputs(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.calcOutput()
                
        self.outputs = [neuron.getOutput() for neuron in self.layers[-1]]
        
    def getOutputs(self):
        return self.outputs
            
    def backprop(self, tests, lr = LEARNING_RATE):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        tests: Listas de pares [entrada, salida esperada] para entrenar la red
        '''
        while self.error > ERROR:
            self.error = 0
            for inputs, targets in tests: #Para cada iteración se realizan todos los tests
                self.setInputs(inputs)
                self.setTargets(targets)
                self.calcOutputs() #Se necesita la salida de cada neurona
                
                for layer in self.layers[::-1]: #Las capas se recorren al revés para calcular los errores (backpropagation)
                    for neuron in layer:
                        neuron.calcNeuronError()
                        neuron.calcBiasError()
                        neuron.calcWeightsErrors()
                        
                for layer in self.layers:
                    for neuron in layer:
                        neuron.update(lr)
                        
                for o in range(self.nNeurons[-1]):
                    self.error += abs(self.outputs[o] - targets[o])
            
            self.error /= len(tests)*self.nNeurons[-1]
            print "\nError:\t" + str(self.error)