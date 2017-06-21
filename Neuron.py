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
            
    def getOutput(self):
        if not self.prevLayer: return self.bias
        
        output = 0
        for neuron in self.prevLayer:
            output += neuron.getOutput() * self.weights[neuron]
        
        output += self.bias
        output = sigmoid(output)
        return output
        
    def getNeuronError(self, targets):
        '''
        Calcula la derivada del error total de la red respecto de la salida de esta neurona
        targets: Diccionario de salidas esperadas de la red, accesibles por la neurona de salida
        '''
        if not self.nextLayer: return self.getOutput() - targets[self]
        
        error = 0
        for neuron in self.nextLayer:
            error += neuron.getBiasError(targets) * neuron.getWeights()[self]
        return error
            
    def getBiasError(self, targets):
        '''
        Calcula la derivada del error total de la red respecto del bias
        targets: Diccionario de salidas esperadas de la red, accesibles por la neurona de salida
        '''
        return self.getNeuronError(targets) * self.getOutput() * (1-self.getOutput())
        
    def getWeightError(self, targets, inputNeuron):
        '''
        Calcula la derivada del error total de la red respecto de un peso de la neurona
        targets: Diccionario de salidas esperadas de la red, accesibles por la neurona de salida
        inputNeuron: Neurona a la que está conectada el peso del que se quiere saber el error
        '''
        return self.getBiasError(targets) * inputNeuron.getOutput()
        
    def update(self, targets, lr):
        '''Actualiza (gently) los pesos y bias de la neurona para que la salida de la red se asemeje a targets
        '''
        self.bias -= self.getBiasError(targets) * lr
        for neuron in self.prevLayer:
            self.weights[neuron] -= self.getWeightError(targets, neuron) * lr
        
    def setBias(self, b): self.bias = b
    def addBiasError(self, b): self.bias += b
    def getBias(self): return self.bias
    def setWeight(self, i, w): self.weights[i] = w
    def addWeight(self, i, w): self.weights[i] += w
    def getWeights(self): return dict(self.weights)