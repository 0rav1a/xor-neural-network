#coding:utf-8
import random
import numpy as np

LEARNING_RATE = 1 #Para gradient descent
MIN_ERROR = 0.1 #Margen de error de la red a partir del cual se considerará entrenada

class NPNetwork:
    '''
    Ahora en lugar de hacer los cálculos manualmente se utiliza la librería numpy. Se supone que así es más eficiente
    '''
    def __init__(self, nNeurons):
        '''
        nNeurons: Número de neuronas que habrá en cada capa. La longitud de esta lista es el número total de capas de la red
        '''
        self.weights = [np.random.randn(srcNeurons, dstNeurons) for srcNeurons, dstNeurons in zip(nNeurons[:-1], nNeurons[1:])] #Pesos de salida de una neurona [capa][neurona][peso]
        self.biases = [np.random.randn(neurons, 1) for neurons in nNeurons[1:]] #Bias de cada capa [capa][peso]
        self.layers = [np.zeros(n) for n in nNeurons] #Salidas de cada neurona [capa][neurona]

    def calcOutputs(self):
        '''Calcula la salida de cada neurona de la red (las de entrada no)
        '''
        self.outputs[1:] = [sigmoid(np.dot(layer, weights) + biases)
            for layer, weights, biases in zip(self.layers[:-1], self.weights, self.biases)]
        
    def backprop(self, tests, lr = LEARNING_RATE):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        tests: Listas de pares [entrada, salida esperada] para entrenar la red
        '''
        epochs = 0
        cost = 1.0 #Valor de la función de coste para los pesos y bias actuales

        while cost > MIN_ERROR:
            weightsErrors = [np.zeros(layer.shape) for layer in self.weights] #Derivada de la función de coste de la red respecto de un peso [capa][neurona][peso]
            biasesErrors = [np.zeros(layer.shape) for layer in self.biases] #Derivada de la función de coste de la red respecto de un bias [capa][neurona]
            
            for inputs, targets in tests: #En cada iteración se realizan todos los tests
                outputsErrors = [np.zeros(len(layer)) for layer in self.outputs] #Derivada del error total de la red respecto de una salida [capa][neurona]
                self.setInputs(inputs)
                self.calcOutputs() #Se necesita la salida de cada neurona

                #Se calculan los errores de neuronas y pesos
                outputsErrors[-1] = self.outputs[-1] - targets
                for l in range(len(self.weights))[::-1]: #Se empieza por la última capa hasta la primera (backpropagation)
                    weightsErrors[l] += np.outer(self.outputs[l], outputsErrors[l+1] * self.outputs[l+1] * (1-self.outputs[l+1]))
                    outputsErrors[l] = np.dot(self.weights[l][:-1], outputsErrors[l+1] * self.outputs[l+1] * (1-self.outputs[l+1]))

            #Se actualizan los pesos según los errores calculados
            for l in range(len(self.weights)):
                self.weights[l] -= weightsErrors[l] * lr

            #En función de los nuevos pesos se calcula la nueva salida y la función de coste de la red
            cost = 0.0
            for inputs, targets in tests:
                self.setInputs(inputs)
                self.calcOutputs()
                cost += np.sum((self.outputs[-1] - targets)**2 / 2)

            cost /= len(tests)
            epochs += 1
            print "\nCost:\t" + str(cost)
            print "LR:\t" + str(lr)
            print "Iter:\t" + str(epochs)

    def setInputs(self, inputs): self.outputs[0] = inputs
    def getOutputs(self): return self.outputs[-1]
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
