#coding:utf-8
import random
from lib import sigmoid

class Neuron:
    '''
    Neurona con varias entradas y una salida
    Cada entrada tiene asignada un peso, y uno extra para bias
    Si no hay entradas a la neurona, su salida será "bias"
    '''
    def __init__(self, inputNeurons = [], weights = []):
        '''
        inputNeurons: Lista de neuronas de entrada
        weights: Lista de pesos de cada neurona de entrada (misma longitud que inputNeurons)
        '''
        self.inputNeurons = inputNeurons
        self.weights = weights
        if (not weights):
            for _ in inputNeurons:
                self.weights.append(random.uniform(-1,1))
            
        self.bias = random.uniform(-1,1)
        
    def getOutput(self):
        '''Genera la salida de la neurona. Para ello se llama a getOutput de cada neurona de entrada
        '''
        if not self.inputNeurons: return self.bias
        
        out = 0
        for i in range(len(self.inputNeurons)):
            out += self.inputNeurons[i].getOutput() * self.weights[i]
        
        out += self.bias
        out = sigmoid(out)
        return out
        
    def setBias(self, b): self.bias = b
    def addBias(self, b): self.bias += b
    def getBias(self): return self.bias
    def setWeight(self, i, w): self.weights[i] = w
    def addWeight(self, i, w): self.weights[i] += w
    def getWeights(self): return list(self.weights)