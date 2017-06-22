#coding:utf-8
import random
from lib import sigmoid

class Neuron:
    '''
    Neurona con varias entradas y una salida
    Cada entrada tiene asignada un peso, y uno extra para el bias
    Si no hay entradas a la neurona, su salida será el bias
    '''
    def __init__(self, prevLayer, nextLayer):
        '''
        prevLayer: Lista de neuronas de la capa anterior (lista vacía si la neurona pertenece a la capa input)
        nextLayer: Lista de neuronas de la capa siguiente (lista vacía si la neurona pertenece a la capa output)
        '''
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer
        self.bias = random.uniform(-1,1)
        
        self.weights = {} #Diccionario de pesos, accesibles por la neurona de la capa anterior a la que están conectados
        for neuron in prevLayer:
            self.weights[neuron] = random.uniform(-1,1)
            
    def calcOutput(self):
        if not self.prevLayer: self.output = self.bias
        else:
            self.output = 0
            for neuron in self.prevLayer:
                self.output += neuron.getOutput() * self.weights[neuron]
            
            self.output += self.bias
            self.output = sigmoid(self.output)
        
    def calcNeuronError(self):
        '''Calcula la derivada del error total de la red respecto de la salida de esta neurona
        '''
        if not self.nextLayer: self.error = self.output - self.target
        else:
            self.error = 0
            for neuron in self.nextLayer:
                self.error += neuron.getBiasError() * neuron.getWeights()[self]
            
    def calcBiasError(self):
        '''Calcula la derivada del error total de la red respecto del bias
        '''
        self.biasError = self.error * self.output * (1-self.output)
        
    def calcWeightsErrors(self):
        '''Calcula la derivada del error total de la red respecto de los pesos de la neurona
        '''
        self.weightsErrors = {neuron: self.biasError * neuron.getOutput() for neuron in self.prevLayer}
        
    def update(self, lr):
        '''Actualiza los pesos y bias de la neurona según los errores ya calculados
        '''
        self.bias -= self.biasError * lr
        for neuron in self.prevLayer:
            self.weights[neuron] -= self.weightsErrors[neuron] * lr
        
    def setBias(self, b): self.bias = b
    def setTarget(self, t): self.target = t
    def getWeights(self): return dict(self.weights)
    def getOutput(self): return self.output
    def getNeuronError(self): return self.error
    def getBiasError(self): return self.biasError
    def getWeightsErrors(self): return self.weightsErrors