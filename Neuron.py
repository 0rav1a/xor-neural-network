#coding:utf-8
import random
from lib import sigmoid

class Neuron:
    '''
    Neurona con varias entradas y una salida
    Cada entrada tiene asignada un peso, y uno extra para el bias
    Si no hay entradas a la neurona, su salida será el bias
    '''
    def __init__(self, inputNeurons = [], weights = []):
        '''
        inputNeurons: Lista de neuronas de entrada
        weights: Lista de pesos de cada neurona de entrada (misma longitud que inputNeurons)
        '''
        self.inputNeurons = inputNeurons
        self.weights = weights
        if (not weights):
            for neuron in inputNeurons:
                #self.weights.append(random.uniform(-1,1))
                self.weights[neuron] = random.uniform(-1,1)
            
        self.bias = random.uniform(-1,1)
        
    def getOutput(self):
        if not self.inputNeurons: return self.bias
        
        out = 0
        for i in range(len(self.inputNeurons)):
            out += self.inputNeurons[i].getOutput() * self.weights[i]
        
        out += self.bias
        out = sigmoid(out)
        return out
        
    def getNeuronError(self, nextLayer = [], target = 0):
        '''Calcula la derivada del error total de la red respecto de la salida de la neurona
        '''
        if target: #Si se proporciona una salida esperada, la neurona es de salida
            return self.getOutput - target
        
        if nextLayer: #Si se proporciona la siguiente capa, la neurona es hidden
            error = 0
            for neuron in nextLayer:
                error += neuron.getBiasError()
            return error
        
        
    def setBias(self, b): self.bias = b
    def addBias(self, b): self.bias += b
    def getBias(self): return self.bias
    def setWeight(self, i, w): self.weights[i] = w
    def addWeight(self, i, w): self.weights[i] += w
    def getWeights(self): return list(self.weights)