#coding:utf-8
import random
from Neuron import Neuron
from lib import gauss, sigmoid

LEARNING_RATE = 1 #Para gradient descent
ERROR = 0.015 #Margen de error de la red a partir del cual se considerará entrenada

class Network:
    def __init__(self, nNeurons = [1, 1, 1], layers = None):
        '''
<<<<<<< HEAD
        nNeurons: Número de neuronas que habrá en cada capa. La longitud de esta lista es el número de capas de la red.
=======
        nInputs: Número de entradas
        nOutputs: Número de salidas
>>>>>>> 5127112ee5dcf8132bc320e0819ef7a0f4bd7ed4
        layers: Conjunto ya creado de capas (se usa al clonar una red)
        '''
        self.layers = layers
        self.nNeurons = nNeurons
        
        if (not layers):
            self.layers = []
            
            self.layers.append([]) #Capa input
            for _ in range(nNeurons[0]):
                self.layers[0].append(Neuron())
    
<<<<<<< HEAD
            for l in range(len(nNeurons))[1:]:
                self.layers.append([]) #Resto de capas
                for _ in range(nNeurons[l]):
                    self.layers[l].append(Neuron(self.layers[l-1], []))
    
    def setInputs(self, inputs):
        for i in range(self.nNeurons[0]):
            self.layers[0][i].setBias(inputs[i])
                
    def getOutput(self, inputs):
        self.setInputs(inputs)
        return [outputNeuron.getOutput() for outputNeuron in self.layers[-1]]
            
    def backprop(self, tests, lr = LEARNING_RATE):
        '''
        Dada una lista de tests, modifica (gently) la red para que la salida se asemeje a la esperada
        tests: Listas de pares [entrada, salida esperada] para entrenar la red
        '''
        while self.calcError(tests) > ERROR:
            for test in tests: #Para cada iteración se realizan todos los tests
                inputs = test[0]
                target = test[1]
        
                #Se calcula la salida de cada neurona
                outputs = []
                self.setInputs(inputs)
                for layer in self.layers:
                    outputs.append([neuron.getOutput() for neuron in layer])
                    
                #Se calcula la derivada del error respecto de la salida de cada neurona:
                
                
                
                #Se calcula la derivada del error respecto de cada peso y bias:
                biasOutput = []
                for o in range(self.nOutputs):
                    biasOutput.append((outputs[o]-target[o])*outputs[o]*(1-outputs[o]))
                
                weightsOutput = []
                for h in range(self.nHidden):
                    weightsOutput.append([])
                    for o in range(self.nOutputs):
                        weightsOutput[h].append(biasOutput[o]*hidden[h])
                
                biasHidden = []
                for h in range(self.nHidden):
                    biasHidden.append(0)
                    for o in range(self.nOutputs):
                        biasHidden[-1] += self.layers["output"][o].getWeights()[h]*weightsOutput[h][o]*(1-hidden[h])
                
                weightsHidden = []
                for i in range(self.nInputs):
                    weightsHidden.append([])
                    for h in range(self.nHidden):
                        weightsHidden[i].append(biasHidden[h]*inputs[i])
                
                #Se aplican los incrementos a los pesos y bias:        
                for o in range(self.nOutputs):
                    self.layers["output"][o].addBias(-biasOutput[o]*lr)
                    for h in range(self.nHidden):
                        self.layers["output"][o].addWeight(h, -weightsOutput[h][o]*lr)
                        
                for h in range(self.nHidden):
                    self.layers["hidden"][h].addBias(-biasHidden[h]*lr)
                    for i in range(self.nInputs):
                        self.layers["hidden"][h].addWeight(i, -weightsHidden[i][h]*lr)
                        
    def calcError(self, tests):
        '''Calcula el error de la red probando los tests
        '''
        self.error = 0.0
        for input, target in tests:
            output = self.getOutput(input)
            for i in range(self.nOutputs):
                self.error += abs(output[i] - target[i]) #1-gauss(output[i], target[i])
            
=======
            for _ in range(nInputs): #Capa hidden
                inputLayer = self.layers["input"]
                self.layers["hidden"].append(Neuron(inputLayer, []))
            
            for _ in range(nOutputs): #Capa output
                hiddenLayer = self.layers["hidden"]
                self.layers["output"].append(Neuron(hiddenLayer, []))
    
    def setInputs(self, inputs):
        for i in range(self.nInputs):
            self.layers["input"][i].setBias(inputs[i])
                
    def getOutput(self, inputs):
        '''Dada una lista de inputs, devuelve la lista de outputs de la red
        '''
        self.setInputs(inputs)
        return [outputNeuron.getOutput() for outputNeuron in self.layers["output"]]
            
    def backprop(self, tests):
        '''Dada una lista de inputs, modifica (gently) la red para que la salida se asemeje a la esperada
        '''
        lr = LEARNING_RATE
        while self.calcError(tests) > ERROR:
            for test in tests:
                inputs = test[0]
                target = test[1]
        
                #Se calcula la salida de cada neurona
                self.setInputs(inputs)
                outputs = [outputNeuron.getOutput() for outputNeuron in self.layers["output"]]
                hidden = [hiddenNeuron.getOutput() for hiddenNeuron in self.layers["hidden"]]
                
                #Se calculan los incrementos para cada peso y bias: (Gradient Descent)
                biasOutput = []
                for o in range(self.nOutputs):
                    biasOutput.append((outputs[o]-target[o])*outputs[o]*(1-outputs[o]))
                
                weightsOutput = []
                for h in range(self.nHidden):
                    weightsOutput.append([])
                    for o in range(self.nOutputs):
                        weightsOutput[h].append(biasOutput[o]*hidden[h])
                
                biasHidden = []
                for h in range(self.nHidden):
                    biasHidden.append(0)
                    for o in range(self.nOutputs):
                        biasHidden[-1] += self.layers["output"][o].getWeights()[h]*weightsOutput[h][o]*(1-hidden[h])
                
                weightsHidden = []
                for i in range(self.nInputs):
                    weightsHidden.append([])
                    for h in range(self.nHidden):
                        weightsHidden[i].append(biasHidden[h]*inputs[i])
                
                #Se aplican los incrementos a los pesos y bias:        
                for o in range(self.nOutputs):
                    self.layers["output"][o].addBias(-biasOutput[o]*lr)
                    for h in range(self.nHidden):
                        self.layers["output"][o].addWeight(h, -weightsOutput[h][o]*lr)
                        
                for h in range(self.nHidden):
                    self.layers["hidden"][h].addBias(-biasHidden[h]*lr)
                    for i in range(self.nInputs):
                        self.layers["hidden"][h].addWeight(i, -weightsHidden[i][h]*lr)
                        
    def calcError(self, tests):
        '''Calcula el error de la red probando los tests
        '''
        self.error = 0.0
        for input, target in tests:
            output = self.getOutput(input)
            for i in range(self.nOutputs):
                self.error += abs(output[i] - target[i]) #1-gauss(output[i], target[i])
            
>>>>>>> 5127112ee5dcf8132bc320e0819ef7a0f4bd7ed4
            print output,
        
        self.error /= len(tests)*self.nOutputs

        print "\nError:\t" + str(self.error)
        return self.error