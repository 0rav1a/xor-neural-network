#coding:utf-8
import random
from Neuron import Neuron
from lib import gauss

LEARNING_RATE = 1 #Para gradient descent
ERROR = 0.0001 #Margen de error de la red a partir del cual se considerará entrenada

class Network:
    '''Red neuronal con una capa oculta (misma longitud que la capa input)
    '''
    def __init__(self, nInputs = 1, nOutputs = 1, layers = None):
        '''
        nInputs: Número de entradas
        nOutputs: Número de salidas
        layers: Conjunto ya creado de capas (se usa al clonar una red)
        '''
        self.layers = layers
        self.nInputs = nInputs
        self.nHidden = nInputs
        self.nOutputs = nOutputs
        
        if (not layers):
            self.layers = {"input": [], "hidden": [], "output": []}
            
            for _ in range(nInputs): #Capa input
                self.layers["input"].append(Neuron())
    
            for _ in range(nInputs): #Capa hidden
                inputLayer = self.layers["input"]
                self.layers["hidden"].append(Neuron(inputLayer, []))
            
            for _ in range(nOutputs): #Capa output
                hiddenLayer = self.layers["hidden"]
                self.layers["output"].append(Neuron(hiddenLayer, []))
            
    def backprop(self, tests):
        '''Dada una lista de inputs, modifica (gently) la red para que la salida se asemeje a target
        '''
        while self.calcError(tests) > ERROR:
        #for _ in range(20000):
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
                    self.layers["output"][o].addBias(-biasOutput[o]*LEARNING_RATE)
                    for h in range(self.nHidden):
                        self.layers["output"][o].addWeight(h, -weightsOutput[h][o]*LEARNING_RATE)
                        
                for h in range(self.nHidden):
                    self.layers["hidden"][h].addBias(-biasHidden[h]*LEARNING_RATE)
                    for i in range(self.nInputs):
                        self.layers["hidden"][h].addWeight(i, -weightsHidden[i][h]*LEARNING_RATE)
                        
    def calcError(self, tests):
        '''Calcula el error de la red probando los tests
        '''
        error = 0.0
        for input, target in tests:
            output = self.getOutput(input)
            for i in range(self.nOutputs):
                error += 1-gauss(output[i], target[i])
            
            print output,
        
        error /= len(tests)*self.nOutputs

        print "\nError:\t" + str(error)
        return error
    
    def clone(self):
        '''Devuelve una nueva red idéntica a self
        '''
        newLayers = {layer: [] for layer in self.layers}
            
        for inputNeuron in self.layers["input"]:
            inputNeurons = []
            weights = inputNeuron.getWeights()
            newLayers["input"].append(Neuron(inputNeurons, weights))

        for hiddenNeuron in self.layers["hidden"]:
            inputNeurons = newLayers["input"]
            weights = hiddenNeuron.getWeights()
            newLayers["hidden"].append(Neuron(inputNeurons, weights))
        
        for outputNeuron in self.layers["output"]:
            inputNeurons = newLayers["hidden"]
            weights = outputNeuron.getWeights()
            newLayers["output"].append(Neuron(inputNeurons, weights))
                
        return Network(self.nInputs, self.nOutputs, newLayers)

    def show(self):
        '''Imprime los valores de pesos y bias de la red (para debugging)
        '''
        for layer in self.layers:
            for neuron in self.layers[layer]:
                print neuron.getWeights()
                print neuron.getBias()
                
    def getOutput(self, inputs):
        '''
        Dada una lista de inputs, devuelve la lista de outputs de la red
        La longitud de inputs debe ser la misma que la de inputLayer (nInputs)
        '''
        self.setInputs(inputs)
        return [outputNeuron.getOutput() for outputNeuron in self.layers["output"]]

    def setInputs(self, inputs):
        for i in range(self.nInputs):
            self.layers["input"][i].setBias(inputs[i])





"""
import matplotlib.pyplot as plt
def draw(self, inputs, target, l, i, w):
    '''
    Dibuja un gráfico en el que se muestra el error de la red neuronal en función de uno de los pesos
    inputs: Lista de inputs de la red neuronal
    target: Lista de outputs esperados, en base a los cuales se calculará el error total
    l: Capa en la que está el peso que se quiere optimizar ["hidden", "output"]
    i: Índice de la neurona dentro de la capa
    w: Índice del peso dentro de la neurona
    '''
    
    xs = [x/1000.0 for x in range(1000)]        #Se define el dominio de la función (pesos que se probarán)
    ys = []
    for x in xs:                                #Se recorren todos los valores del dominio
        self.layers[l][i].setWeight(w, x)       #Se establece el peso correspondiente en la red neuronal
        outputs = self.getOutput(inputs)        #Se calcula la salida con el peso establecido
        y = 0.0
        for o in range(len(outputs)):           #Se calcula el error en base a la salida obtenida y la esperada
            y += (target[o]-outputs[o])**2
            
        ys.append(y/2)
    
    print xs[ys.index(min(ys))]
    #plt.plot(xs, ys)
    #plt.show()
    self.layers[l][i].setWeight(w, random.uniform(0,1))
"""