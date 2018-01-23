#coding:utf-8
import numpy as np
import random

LEARNING_RATE = .5 #Para gradient descent
MIN_ERROR = .05 #Margen de error de la red a partir del cual se considerará entrenada
BATCH_SIZE = 10 #Tamaño del bloque de datos que usará la red para modificar los pesos

class Network:
    def __init__(self, nNeurons):
        '''
        nNeurons: Lista de capas de la red. La longitud será el número de capas, y el valor el número de neuronas en cada capa
        '''
        self.weights = [np.random.random((srcNeurons, dstNeurons))*2-1 for srcNeurons, dstNeurons in zip(nNeurons[:-1], nNeurons[1:])] #Pesos de salida de una neurona [capa][neurona][peso]
        self.biases = [np.random.random((1, neurons))*2-1 for neurons in nNeurons] #Bias de salida de cada capa [capa][peso] (los de la primera no se utilizan)
        self.layers = [np.zeros((1, neurons)) for neurons in nNeurons] #Salidas de cada neurona [capa][neurona]
        
    def calcOutputs(self):
        '''Calcula la salida de cada neurona de la red (las de entrada no)
        '''
        for l in range(len(self.layers))[1:]:
            self.layers[l] = sigmoid(np.dot(self.layers[l-1], self.weights[l-1]) + self.biases[l])
        
    def backprop(self, train, tests, lr = LEARNING_RATE):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        train: Listas de pares [entrada, salida esperada] para entrenar la red
        tests: Listas de pares [entrada, salida esperada] para comprobar el funcionamiento la red
        '''
        epochs = 0
        cost = 1.0 #Valor de la función de coste para los pesos y bias actuales

        while cost > MIN_ERROR:
            for batch in [train[i:i+BATCH_SIZE] for i in range(len(train))[::BATCH_SIZE]]:
                weightsErrors = [np.zeros(layer.shape) for layer in self.weights] #Derivada de la función de coste de la red respecto de un peso [capa][neurona][peso]
                biasesErrors = [np.zeros(layer.shape) for layer in self.biases] #Derivada de la función de coste de la red respecto de un bias [capa][neurona]
                
                for inputs, targets in batch:
                    errors = [np.zeros(layer.shape) for layer in self.layers] #Error de la salida de una neurona en este test (antes de aplicar la función de activación)
                    self.setInputs(inputs)
                    self.calcOutputs() #Se calcula la salida de cada neurona

                    #Se calculan los errores de salidas, pesos y bias
                    errors[-1] = self.layers[-1] * (1 - self.layers[-1]) * (self.layers[-1] - targets)
                    for l in range(len(self.layers))[-2::-1]: #Se empieza por la penúltima capa hasta la primera (backpropagation)
                        errors[l] = np.dot(errors[l+1], self.weights[l].transpose()) * self.layers[l] * (1-self.layers[l])
                        biasesErrors[l+1] += errors[l+1] #El error de un bias coincide con el error de su neurona antes de aplicar la función de activación
                        weightsErrors[l] += np.dot(self.layers[l].transpose(), errors[l+1])

                #Se actualizan los pesos y bias según los errores calculados
                for l in range(len(self.weights)):
                    self.weights[l] -= weightsErrors[l] * lr
                    self.biases[l+1] -= biasesErrors[l+1] * lr

            #Se evalúa la función de coste de la red con los nuevos pesos
            cost = 0.0
            success = 0
            for inputs, targets in tests:
                self.setInputs(inputs)
                self.calcOutputs()
                cost += np.sum((self.layers[-1] - targets)**2 / 2)
                if np.argmax(self.layers[-1]) == np.argmax(targets): success += 1

            cost /= len(tests)
            epochs += 1
            lr = float(open("LR", "rb").read())
            print "\nCost:\t", cost
            print "Aciertos:", success, "/", len(tests)
            print "LR:\t", lr
            print "Iter:\t", epochs

    def setInputs(self, inputs): self.layers[0] = np.array([inputs])
    def getOutputs(self): return self.layers[-1]
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def relu(x): return np.maximum(x, 0)
def drelu(x): return np.around(x)
