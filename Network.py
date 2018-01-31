#coding:utf-8
import numpy as np
import random
import time

LEARNING_RATE = 0 #Para gradient descent (si vale 0 se toma el valor del fichero LR)
MIN_ERROR = .005 #Margen de error de la red a partir del cual se considerará entrenada
BATCH_SIZE = 10 #Tamaño del bloque de datos que usará la red para modificar los pesos

class Network:
    def __init__(self, nNeurons):
        '''
        nNeurons: Lista de capas de la red. La longitud será el número de capas, y el valor el número de neuronas en cada capa
        '''
        self.weights = [np.random.randn(srcNeurons, dstNeurons) for srcNeurons, dstNeurons in zip(nNeurons[:-1], nNeurons[1:])] #Pesos de salida de una neurona [capa][neurona][peso]
        self.biases = [abs(np.random.randn(1, neurons)) for neurons in nNeurons] #Bias de cada capa [capa][peso] (los de la primera no se utilizan)
        self.outputs = [np.zeros((1, neurons)) for neurons in nNeurons] #Salidas de cada neurona sin activar [capa][neurona]
        self.activations = [np.zeros((1, neurons)) for neurons in nNeurons] #Salidas de cada neurona activada [capa][neurona]
        
    def calcOutputs(self):
        '''Calcula la salida de cada neurona de la red (las de entrada no)
        '''
        for l in range(len(self.outputs))[1:]:
            self.outputs[l] = np.dot(self.activations[l-1], self.weights[l-1]) + self.biases[l]
            self.activations[l] = sigmoid(self.outputs[l])
        
    def backprop(self, train, tests):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        train: Listas de pares [entrada, salida esperada] para entrenar la red
        tests: Listas de pares [entrada, salida esperada] para comprobar el funcionamiento la red
        '''
        epochs = 0
        cost = 1.0 #Valor de la función de coste para los pesos y bias actuales

        while cost > MIN_ERROR:
            lr = LEARNING_RATE if LEARNING_RATE else float(open("LR", "r").read())
            for batch in [train[i:i+BATCH_SIZE] for i in range(0, len(train), BATCH_SIZE)]:
                weightsErrors = [np.zeros_like(layer) for layer in self.weights] #Incremento de los pesos tras la iteración actual
                biasesErrors = [np.zeros_like(layer) for layer in self.biases] #Incremento de los bias tras la iteración actual
                
                for inputs, targets in batch:
                    outputsErrors = [np.zeros_like(layer) for layer in self.outputs] #Error de la salida de una neurona en este test (antes de aplicar la función de activación)
                    self.setInputs(inputs)
                    self.calcOutputs() #Se calcula la salida de cada neurona

                    #Se calculan los errores de salidas, pesos y bias para este test
                    outputsErrors[-1] = dsigmoid(self.outputs[-1]) * dquadratic(self.activations[-1], targets)
                    for l in range(len(self.outputs))[-2::-1]: #Se empieza por la penúltima capa hasta la primera (backpropagation)
                        outputsErrors[l] = np.dot(outputsErrors[l+1], self.weights[l].transpose()) * dsigmoid(self.outputs[l])
                        biasesErrors[l+1] += outputsErrors[l+1] #El error de un bias coincide con el error de su neurona antes de aplicar la función de activación
                        weightsErrors[l] += np.dot(self.activations[l].transpose(), outputsErrors[l+1])

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
                cost += quadratic(self.activations[-1], targets)
                #cost += np.sum((self.activations[-1] - targets)**2 / 2)
                if np.argmax(self.activations[-1]) == np.argmax(targets): success += 1
            
            cost /= len(tests)
            epochs += 1
            print "\nCost:\t%f\nAciertos: %d/%d\nLR:\t%f\nIter:\t%d" % (cost, success, len(tests), lr, epochs)

    def setInputs(self, inputs):
        self.activations[0] = np.array([inputs])
        self.outputs[0] = np.array([inputs])
    def getOutputs(self): return self.activations[-1]
    
#FUNCIONES DE ACTIVACIÓN
def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def dsigmoid(z):
    y = sigmoid(z)
    return y*(1-y)
    
def relu(z): return np.array([[i if i > 0 else 0 for i in z[0]]])
def drelu(z): return np.array([[1 if i > 0 else 0 for i in z[0]]])

def softmax(z): return np.exp(z)/np.sum(np.exp(z)) #Usar con loglikelihood
def dsoftmax(z): return np.ones_like(z)

#FUNCIONES DE COSTE
def quadratic(a, y): return np.sum(0.5*(a-y)**2)
def dquadratic(a, y): return a-y

def crossentropy(a, y): return np.sum(-y*np.log(a) - (1-y)*np.log(1-a))
def dcrossentropy(a, y): return (a-y)/(a-a**2)

def loglikelihood(a, y): return -np.log(a[0][np.argmax(y)]) #Usar con softmax
def dloglikelihood(a, y): return -1/a[0][np.argmax(y)]
