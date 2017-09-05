#coding:utf-8
import random
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.005 #Para gradient descent
MIN_ERROR = 0.1 #Margen de error de la red a partir del cual se considerará entrenada

class NPNetwork:
    '''
    Ahora en lugar de hacer los cálculos manualmente se utiliza la librería numpy. Se supone que así es más eficiente
    '''
    def __init__(self, nNeurons):
        '''
        nNeurons: Número de neuronas que habrá en cada capa. La longitud de esta lista es el número total de capas de la red
        '''
        self.weights = [2*np.random.random((nNeurons[l]+1, nNeurons[l+1]))-1 for l in range(len(nNeurons)-1)] #Pesos de salida de una neurona [capa][neurona][peso]. Incluye los pesos de salida de la neurona bias
        self.outputs = [np.zeros(n) for n in nNeurons] #Salidas calculadas de cada neurona [capa][neurona]

    def calcOutputs(self):
        '''Calcula la salida de cada neurona de la red
        '''
        for l in range(len(self.weights)):
            self.outputs[l+1] = sigmoid(np.dot(np.append(self.outputs[l], 1), self.weights[l])) #Se añade un 1 al final de cada capa (bias)

    def backprop(self, tests, lr = LEARNING_RATE):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        tests: Listas de pares [entrada, salida esperada] para entrenar la red
        '''
        weightsErrors = [np.zeros((nNeurons[l]+1, nNeurons[l+1])) for l in range(len(nNeurons)-1)] #Derivada del error total de la red respecto de un peso [capa][neurona][peso]
        outputsErrors = [np.zeros(n) for n in nNeurons] #Derivada del error total de la red respecto de una salida [capa][neurona]
        #weightsIncr   = [np.zeros((nNeurons[l]+1, nNeurons[l+1])) for l in range(len(nNeurons)-1)] #Incremento de cada peso de la red (una vez calculado el error respecto de varios tests)
        cont = 0
        currError = 1.0 #Error actual de la red (después de realizar todos los tests)
        prevErrors = []

        while currError > MIN_ERROR:
            currError = 0.0
            for inputs, targets in tests: #En cada iteración se realizan todos los tests
                self.setInputs(inputs)
                self.calcOutputs() #Se necesita la salida de cada neurona

                #Se calculan los errores de neuronas y pesos
                outputsErrors[-1] = self.outputs[-1] - targets
                for l in range(len(self.weights))[::-1]: #Se empieza por la última capa hasta la primera (backpropagation)
                    weightsErrors[l] = np.outer(np.append(self.outputs[l], 1), outputsErrors[l+1] * self.outputs[l+1] * (1-self.outputs[l+1]))
                    outputsErrors[l] = np.dot(self.weights[l][:-1], outputsErrors[l+1] * self.outputs[l+1] * (1-self.outputs[l+1]))

                #Se actualizan los pesos según los errores calculados
                for l in range(len(self.weights)):
                    self.weights[l] -= weightsErrors[l] * lr

                #Se calcula la salida de la red con los nuevos pesos y se modifica el error
                self.calcOutputs()
                if np.argmax(self.outputs[-1]) != np.argmax(targets): error += 1

            currError /= len(tests)
            prevErrors.append(currError)
            cont += 1
            plt.plot(errors)
            plt.pause(0.0000000001)
            print "\nError:\t" + str(currError)
            print "LR:\t" + str(lr)
            print "Cont:\t" + str(cont)

    def setInputs(self, inputs): self.outputs[0] = inputs
    def getOutputs(self): return self.outputs[-1]
def sigmoid(x): return 1/(1+np.exp(-x))
