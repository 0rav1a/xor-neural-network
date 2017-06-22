#coding:utf-8
import random
from lib import sigmoid

class Neuron:
    '''
    Neurona con varias entradas, un peso asignado a cada una, y una salida
    Una de las entradas será bias (1 constante)
    Si la neurona está en la capa input no tendrá entradas
    '''
    def __init__(self, prevLayer, nextLayer):
        '''
        prevLayer: Lista de neuronas de la capa anterior (lista vacía si la neurona pertenece a la capa input)
        nextLayer: Lista de neuronas de la capa siguiente (lista vacía si la neurona pertenece a la capa output)
        '''
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer
        
        self.weights = {} #Diccionario de pesos, accesibles por la neurona de la capa anterior a la que están conectados
        for neuron in prevLayer:
            self.weights[neuron] = random.uniform(-1,1)
            
    def calcOutput(self):
        if self.prevLayer:
            self.output = 0
            for neuron in self.prevLayer:
                self.output += neuron.getOutput() * self.weights[neuron]
            
            self.output = sigmoid(self.output)
        
    def calcError(self):
        '''Calcula la derivada del error total de la red respecto de la salida de esta neurona
        '''
        if not self.nextLayer: self.error = self.output - self.target
        else:
            self.error = 0
            for neuron in self.nextLayer:
                self.error += neuron.getError() * neuron.getOutput() * (1-neuron.getOutput()) * neuron.getWeights()[self]
        
    def calcWeightsErrors(self):
        '''Calcula la derivada del error total de la red respecto de los pesos de la neurona
        '''
        self.weightsErrors = {neuron: self.error * self.output * (1-self.output) * neuron.getOutput() for neuron in self.prevLayer}
        
    def updateWeights(self, lr):
        '''Actualiza los pesos de la neurona según los errores ya calculados
        '''
        for neuron in self.prevLayer:
            self.weights[neuron] -= self.weightsErrors[neuron] * lr
        
    def setOutput(self, o): self.output = o
    def setTarget(self, t): self.target = t
    def getWeights(self): return dict(self.weights)
    def getOutput(self): return self.output
    def getError(self): return self.error