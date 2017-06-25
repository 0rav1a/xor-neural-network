#coding:utf-8
import random
from Neuron import Neuron

LEARNING_RATE = 1 #Para gradient descent
MIN_ERROR = 0.001 #Margen de error de la red a partir del cual se considerará entrenada

class Network:
    def __init__(self, nNeurons):
        '''
        nNeurons: Número de neuronas que habrá en cada capa. La longitud de esta lista es el número total de capas de la red
        '''
        self.nNeurons = nNeurons
        
        self.layers = [[] for _ in nNeurons]
        bias = Neuron([], [])
        bias.setOutput(1)
        
        self.layers[0][:] = [Neuron([], self.layers[1]) for _ in range(nNeurons[0])] #Capa input
        
        for l in range(len(nNeurons))[1:-1]:
            self.layers[l][:] = [Neuron(self.layers[l-1]+[bias], self.layers[l+1]) for _ in range(nNeurons[l])] #Capas hidden
            
        self.layers[-1][:] = [Neuron(self.layers[-2]+[bias], []) for _ in range(nNeurons[-1])] #Capa output
    
    def setInputs(self, inputs):
        for i in range(self.nNeurons[0]):
            self.layers[0][i].setOutput(inputs[i])
            
    def setTargets(self, targets):
        for o in range(self.nNeurons[-1]):
            self.layers[-1][o].setTarget(targets[o])
            
    def calcOutputs(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.calcOutput()
            
    def backprop(self, tests, lr = LEARNING_RATE):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        tests: Listas de pares [entrada, salida esperada] para entrenar la red
        '''
        error = 1
        while error > MIN_ERROR:
            error = 0
            random.shuffle(tests)
            for inputs, targets in tests: #Para cada iteración se realizan todos los tests
                self.setInputs(inputs)
                self.setTargets(targets)
                self.calcOutputs() #Se necesita la salida de cada neurona
                
                for layer in self.layers[::-1]: #Las capas se recorren al revés para calcular los errores (backpropagation)
                    for neuron in layer:
                        neuron.calcError()
                        neuron.calcWeightsErrors()
                        
                for layer in self.layers:
                    for neuron in layer:
                        neuron.updateWeights(lr)
                        
                for o in range(self.nNeurons[-1]):
                    error += abs(self.layers[-1][o].getOutput() - targets[o])
            
            error /= len(tests)*self.nNeurons[-1]
            print "\nError:\t" + str(error)
            print "LR:\t" + str(lr)
            
    def getOutputs(self): return [neuron.getOutput() for neuron in self.layers[-1]]