#coding:utf-8
import numpy as np
import random
import time

LR = 0.5 #Para gradient descent (si vale 0 se toma el valor del fichero LR)
MIN_ERROR = .005 #Margen de error de la red a partir del cual se considerará entrenada
BATCH_SIZE = 10 #Tamaño del bloque de datos que usará la red para modificar los pesos

class Network:
    def __init__(self, shape):
        '''
        shape: Lista de capas de la red. La longitud será el número de capas, y el valor el número de neuronas en cada capa
        '''
        self.shape = shape
        self.weights = [np.random.standard_normal((dstNeurons, srcNeurons)) for srcNeurons, dstNeurons in zip(shape[:-1], shape[1:])] #Pesos de salida de una neurona [capa][neurona_dst][neurona_src]
        self.biases = [np.random.standard_normal((neurons, 1)) for neurons in shape] #Bias de cada capa [capa][peso][0] (los de la capa 0 no se utilizan)
        
        self.outputs = [np.zeros((neurons, BATCH_SIZE)) for neurons in shape] #Salidas de cada neurona sin activar para un conjunto de tests [capa][neurona][test]
        self.activations = [np.zeros((neurons, BATCH_SIZE)) for neurons in shape] #Salidas de cada neurona activada para un conjunto de tests [capa][neurona][test]
        self.errors = [np.zeros((neurons, BATCH_SIZE)) for neurons in shape] #Error de la salida sin activar de cada neurona para un conjunto de tests [capa][neurona][test]
        
    def calcOutputs(self):
        '''Calcula la salida de cada neurona de la red (las de entrada no)
        '''
        for l in range(len(self.outputs))[1:]: #Feedforward: Desde la primera hasta la última capa
            self.outputs[l] = np.dot(self.weights[l-1], self.activations[l-1]) + self.biases[l]
            self.activations[l] = sigmoid(self.outputs[l])
            
    def calcErrors(self, targets):
        '''Calcula el error de la salida sin activar de cada neurona de la red
        '''
        self.errors[-1] = dsigmoid(self.outputs[-1]) * dquadratic(self.activations[-1], targets)
        for l in range(len(self.outputs))[-2::-1]: #Backpropagation: Desde la penúltima hasta la primera capa
            self.errors[l] = np.dot(self.weights[l].T, self.errors[l+1]) * dsigmoid(self.outputs[l])
        
        
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

        while cost > MIN_ERROR:
            t = time.time()
            for inputs, targets in zip(np.split(trainInputs, n_batches, axis=1), np.split(trainOutputs, n_batches, axis=1)):
                self.setInputs(inputs)
                self.calcOutputs() #Se calcula la salida de cada neurona
                self.calcErrors(targets) #Se calculan los errores de cada neurona
                
                #Se actualizan los pesos y bias en función de los errores calculados
                for l in range(len(self.weights)):
                    self.weights[l] -= np.dot(self.errors[l+1], self.activations[l].T) * LR
                    self.biases[l+1] -= np.sum(self.errors[l+1], axis=1, keepdims=True) * LR
                
            epochs += 1
            print "Iter:\t%d\nTiempo:\t%f" % (epochs, time.time()-t)
            cost, accuracy = self.test(testInputs, testOutputs)
            print "Cost:\t%f\nAcc:\t%.2f%%\n\n" % (cost, accuracy*100)

    def test(self, testInputs, testOutputs):
        self.setInputs(testInputs)
        self.calcOutputs()
        cost = quadratic(self.activations[-1], testOutputs) / len(testInputs[0])
        accuracy = np.sum(np.equal(np.argmax(self.activations[-1],axis=0), np.argmax(testOutputs, axis=0))) / float(len(testInputs[0]))
        return cost, accuracy
        
    def setInputs(self, inputs): self.activations[0] = inputs
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

def loglikelihood(a, y): return -np.log(a[np.argmax(y)]) #Usar con softmax
def dloglikelihood(a, y): return -1/a[np.argmax(y)]
