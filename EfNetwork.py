#coding:utf-8
import random
from math import e

LEARNING_RATE = 10 #Para gradient descent
MIN_ERROR = 0.001 #Margen de error de la red a partir del cual se considerará entrenada

class EfNetwork:
    '''
    Red neuronal que no utiliza la clase neurona. Se supone que es más eficiente
    En su lugar se utilizan listas para almacenar pesos, salidas y errores de cada neurona
    '''
    def __init__(self, nNeurons):
        '''
        nNeurons: Número de neuronas que habrá en cada capa. La longitud de esta lista es el número total de capas de la red
        '''
        self.weights = []       #Pesos de entrada a una neurona [capa][neurona][peso]
        self.outputs = []       #Salidas calculadas de cada neurona [capa][neurona]
        self.targets = []       #Salidas esperadas de la red [neurona]
        self.errors = []        #Derivada del error total de la red respecto de la salida de una neurona [capa][neurona]
        self.weightsErrors = [] #Derivada del error total de la red respecto de un peso [capa][neurona][peso]
        
        self.weights.append([[] for _ in range(nNeurons[0])]) #Pesos capa input
        self.outputs.append([0 for _ in range(nNeurons[0])]) #Salidas capa input (entradas de la red)
        self.errors.append([0 for _ in range(nNeurons[0])]) #Errores capa input
        self.weightsErrors.append([[] for _ in range(nNeurons[0])]) #Errores de pesos capa input
        
        for l in range(len(nNeurons))[1:]: #Resto de capas
            self.weights.append([])
            self.outputs.append([])
            self.errors.append([])
            self.weightsErrors.append([])
            for _ in range(nNeurons[l]):
                self.weights[l].append([random.uniform(-1,1) for _ in range(nNeurons[l-1]+1)])
                self.outputs[l].append(0)
                self.errors[l].append(0)
                self.weightsErrors[l].append([0 for _ in range(nNeurons[l-1]+1)])
    
    def setInputs(self, inputs):
        self.outputs[0] = list(inputs)
            
    def setTargets(self, targets):
        self.targets = list(targets)
            
    def calcOutputs(self):
        '''Calcula la salida de cada neurona de la red
        '''
        for l in range(len(self.weights))[1:]:
            for n in range(len(self.weights[l])):
                self.outputs[l][n] = sigmoid(sum([self.outputs[l-1][i]*self.weights[l][n][i] for i in range(len(self.outputs[l-1]))]) + self.weights[l][n][-1])
                
    def backprop(self, tests, lr = LEARNING_RATE):
        '''
        Dada una lista de tests, se modifican los pesos y bias de todas las neuronas para que la salida se asemeje a la esperada
        tests: Listas de pares [entrada, salida esperada] para entrenar la red
        '''
        error = 1.0
        while error > MIN_ERROR:
            prev_error = error
            error = 0.0
            for inputs, targets in tests: #En cada iteración se realizan todos los tests
                self.setInputs(inputs)
                self.setTargets(targets)
                self.calcOutputs() #Se necesita la salida de cada neurona
                
                #Se calculan los errores de neuronas y pesos
                for n in range(len(self.weights[-1])): #Primero la última capa
                    self.errors[-1][n] = self.outputs[-1][n] - self.targets[n]
                    for w in range(len(self.weights[-1][n])-1):
                        self.weightsErrors[-1][n][w] = self.errors[-1][n] * self.outputs[-1][n] * (1-self.outputs[-1][n]) * self.outputs[-2][w]
                    self.weightsErrors[-1][n][-1] = self.errors[-1][n] * self.outputs[-1][n] * (1-self.outputs[-1][n]) #Bias
                        
                for l in range(len(self.weights))[-2:0:-1]: #Luego el resto de capas hacia atrás (backpropagation). La capa input no es necesaria
                    for n in range(len(self.weights[l])):
                        self.errors[l][n] = sum([self.errors[l+1][x] * self.outputs[l+1][x] * (1-self.outputs[l+1][x]) * self.weights[l+1][x][n] for x in range(len(self.weights[l+1]))])
                        for w in range(len(self.weights[l][n])-1):
                            self.weightsErrors[l][n][w] = self.errors[l][n] * self.outputs[l][n] * (1-self.outputs[l][n]) * self.outputs[l-1][w]
                        self.weightsErrors[l][n][-1] = self.errors[l][n] * self.outputs[l][n] * (1-self.outputs[l][n]) #Bias
                
                #Se actualizan los pesos según los errores calculados
                for l in range(len(self.weights)):
                    for n in range(len(self.weights[l])):
                        for w in range(len(self.weights[l][n])):
                            self.weights[l][n][w] -= self.weightsErrors[l][n][w] * LEARNING_RATE 
                            
                #Se calcula la salida de la red con los nuevos pesos para saber el error 
                self.calcOutputs()       
                for n in range(len(self.weights[-1])):
                    error += abs(self.outputs[-1][n] - self.targets[n])
                    
            error /= len(tests)*len(self.weights[-1])
            print "\nError:\t" + str(error)
            print "LR:\t" + str(lr)
            
    def getOutputs(self): return self.outputs[-1]
def sigmoid(x): return 1/(1+e**-x)