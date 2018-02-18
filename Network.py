#coding:utf-8
import numpy as np
import random
import time

LR = 0.1 #Para gradient descent (si vale 0 se toma el valor del fichero LR)
MIN_ERROR = .005 #Margen de error de la red a partir del cual se considerará entrenada
BATCH_SIZE = 10 #Tamaño del bloque de datos que usará la red para modificar los pesos

class Network:
    def __init__(self, shape):
        '''
        shape: Lista de capas de la red. La longitud será el número de capas, y el valor el número de neuronas en cada capa
        '''
        self.weights = [np.random.normal(scale=(1/np.sqrt(shape[l])), size=(shape[l], shape[l+1])) for l in range(len(shape))[:-1]] #Pesos de salida de una neurona [capa][neurona_src][neurona_dst]
        self.biases = [abs(np.random.normal(size=(1, n))) for n in shape] #Bias de cada capa [capa][peso][0] (los de la capa 0 no se utilizan)
        
        self.outputs = [np.zeros((BATCH_SIZE, n)) for n in shape]       #Salidas de cada neurona sin activar para un conjunto de tests [capa][test][neurona]
        self.activations = [np.zeros((BATCH_SIZE, n)) for n in shape]   #Salidas de cada neurona activada para un conjunto de tests [capa][test][neurona]
        self.errors = [np.zeros((BATCH_SIZE, n)) for n in shape]        #Error de la salida (sin activar) de cada neurona para un conjunto de tests [capa][test][neurona]
        
    def calcOutputs(self, inputs):
        '''Calcula la salida de cada neurona de la red
        '''
        self.activations[0] = inputs
        for l in range(len(self.outputs))[1:]: #Feedforward: Desde la primera hasta la última capa
            self.outputs[l] = np.dot(self.activations[l-1], self.weights[l-1]) + self.biases[l]
            self.activations[l] = relu(self.outputs[l])
        self.activations[-1] = sigmoid(self.outputs[-1])
            
    def calcErrors(self, targets):
        '''Calcula el error de la salida (sin activar) de cada neurona de la red
        '''
        self.errors[-1] = dsigmoid(self.outputs[-1]) * dquadratic(self.activations[-1], targets) / BATCH_SIZE
        for l in range(len(self.outputs))[-2::-1]: #Backpropagation: Desde la penúltima hasta la primera capa
            self.errors[l] = np.dot(self.errors[l+1], self.weights[l].T) * drelu(self.outputs[l])
        
    def train(self, trainInputs, trainOutputs, testInputs, testOutputs):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        trainInputs: Lista de entradas para entrenar la red [neurona][test]
        trainOutputs: Lista de salidas correspondientes [neurona][test] (misma longitud que trainInputs)
        testInputs: Lista de entradas para comprobar el funcionamiento la red [neurona][test]
        testOutputs: Lista de salidas correspondientes [neurona][test] (misma longitud que testInputs)
        '''
        epochs = 0
        cost = 1.0 #Valor de la función de coste para los pesos y bias actuales
        n_batches = len(trainInputs[0])/BATCH_SIZE
        
        trainInputs = trainInputs.T
        trainOutputs = trainOutputs.T
        testInputs = testInputs.T
        testOutputs = testOutputs.T

        while cost > MIN_ERROR:
            t = time.time()
            for inputs, targets in zip(np.split(trainInputs, n_batches, axis=0), np.split(trainOutputs, n_batches, axis=0)):
                self.calcOutputs(inputs) #Se calcula la salida de cada neurona
                self.calcErrors(targets) #Se calculan los errores de cada neurona
                
                #Se actualizan los pesos y bias en función de los errores calculados
                for l in range(len(self.weights)):
                    self.weights[l] -= np.dot(self.activations[l].T, self.errors[l+1]) * LR
                    self.biases[l+1] -= np.sum(self.errors[l+1], axis=0, keepdims=True) * LR
                
            epochs += 1
            print "Iter:\t%d\nTiempo:\t%f" % (epochs, time.time()-t)
            cost, accuracy = self.test(testInputs, testOutputs)
            print "Cost:\t%f\nAcc:\t%.2f%%\n\n" % (cost, accuracy*100)

    def test(self, testInputs, testOutputs):
        self.calcOutputs(testInputs)
        cost = quadratic(self.activations[-1], testOutputs) / len(testInputs)
        accuracy = np.sum(np.equal(np.argmax(self.activations[-1],axis=1), np.argmax(testOutputs, axis=1))) / float(len(testInputs))
        return cost, accuracy
        
    def show(self):
        print "Capa 1: ", self.activations[1]
        print "Capa 2: ", self.activations[2]
        print "Errores 1: ", self.errors[1]
        print "Errores 2: ", self.errors[2]
        print "Weights 0: ", self.weights[0]
        print "Weights 1: ", self.weights[1]
        
    def getOutputs(self): return self.activations[-1]
    
#FUNCIONES DE ACTIVACIÓN
#z: Salidas sin activar de una capa [test][neurona]
#return: Vector de salidas activadas
#return (derivative): Vector de errores de cada neurona sin activar
def sigmoid(z): return 1.0/(1.0+np.exp(-z)) #Valores entre 0 y 1
def dsigmoid(z):
    tmp = sigmoid(z)
    return tmp*(1-tmp)
    
def relu(z): return np.maximum(0, z) #Valores entre 0 e infinito
def drelu(z): return np.sign(z)/2 + 0.5

def tanh(z): return np.tanh(z) #Valores entre -1 y 1
def dtanh(z): return np.sech(z)**2

def softmax(z): return np.exp(z)/np.sum(np.exp(z)) #Probabilidades entre 0 y 1. La suma es 1
def dsoftmax(z):
    tmp = softmax(z)
    return tmp*(1-tmp)

#FUNCIONES DE COSTE
#a: Activaciones de la última capa [test][neurona]
#y: Salidas esperadas de la última capa [test][neurona]
#return: Escalar que representa el coste de toda la red para el conjunto de tests dado
#return (derivative): Vector de errores de cada salida de la red
def quadratic(a, y): return np.sum(0.5*(a-y)**2)
def dquadratic(a, y): return a-y

def loglikelihood(a, y): #Usar con softmax
    cost = 0.0
    for t in range(len(a)):
        cost += -np.log(a[t][np.argmax(y[t])])
    return cost
def dloglikelihood(a, y): return (a-y)/(a*(1-a))
#def dloglikelihood(a, y): return -1/a[np.argmax(y)]
