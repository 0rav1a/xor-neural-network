#coding:utf-8
import numpy as np

LEARNING_RATE = .001 #Para gradient descent
MIN_ERROR = .1 #Margen de error de la red a partir del cual se considerará entrenada

class NPNetwork:
    '''
    Ahora en lugar de hacer los cálculos manualmente se utiliza la librería numpy. Se supone que así es más eficiente
    '''
    def __init__(self, nNeurons):
        '''
        nNeurons: Número de neuronas que habrá en cada capa. La longitud de esta lista es el número total de capas de la red
        '''
        self.weights = [np.random.randn(srcNeurons, dstNeurons) for srcNeurons, dstNeurons in zip(nNeurons[:-1], nNeurons[1:])] #Pesos de salida de una neurona [capa][neurona][peso]
        self.biases = [np.random.randn(1, neurons) for neurons in nNeurons[1:]] #Bias de cada capa [capa][peso]
        self.layers = [np.zeros((1, n)) for n in nNeurons] #Salidas de cada neurona [capa][neurona]
        
    def calcOutputs(self):
        '''Calcula la salida de cada neurona de la red (las de entrada no)
        '''
        for l in range(len(self.layers))[1:]:
            self.layers[l] = sigmoid(np.dot(self.layers[l-1], self.weights[l-1]) + self.biases[l-1])
        
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
            biasesErrorsTest = [np.zeros(layer.shape) for layer in self.biases] #Derivada de la función de coste de un test respecto de un bias [capa][neurona]
            
            for inputs, targets in tests: #En cada iteración se realizan todos los tests
                self.setInputs(inputs)
                self.calcOutputs() #Se necesita la salida de cada neurona

                #Se calculan los errores de biases
                biasesErrorsTest[-1] = (self.layers[-1] - targets) * self.layers[-1] * (1 - self.layers[-1])
                biasesErrors[-1] += biasesErrorsTest[-1]
                for l in range(len(self.biases))[-2::-1]: #Se empieza por la penúltima capa hasta la segunda (backpropagation)
                    biasesErrorsTest[l] = np.dot(self.weights[l+1], biasesErrorsTest[l+1].transpose()).transpose() * self.layers[l+1] * (1-self.layers[l+1])
                    #biasesErrorsTest[l] = np.dot(self.weights[l+1], biasesErrorsTest[l+1]) * self.layers[l+1] * (1-self.layers[l+1])
                    biasesErrors[l] += biasesErrorsTest[l]
                    
                #Se calculan los errores de pesos
                for l in range(len(self.weights))[::-1]: #Se empieza por la penúltima capa hasta la primera (backpropagation)
                    weightsErrors[l] += np.dot(self.layers[l].transpose(), biasesErrorsTest[l])
                    #weightsErrors[l] += np.outer(self.layers[l], biasesErrorsTest[l])

            #Se actualizan los pesos y biases según los errores calculados
            for l in range(len(self.weights)):
                self.weights[l] -= weightsErrors[l] * lr
                self.biases[l] -= biasesErrors[l] * lr

            #En función de los nuevos pesos se calcula la nueva salida y la función de coste de la red
            cost = 0.0
            error = 0.0
            for inputs, targets in tests:
                self.setInputs(inputs)
                self.calcOutputs()
                cost += np.sum((self.layers[-1] - targets)**2 / 2)
                if np.argmax(self.layers[-1]) != np.argmax(targets): error += 1

            cost /= len(tests)
            error /= len(tests)
            epochs += 1
            print "\nCost:\t", cost
            print "Error:\t", error
            print "LR:\t", lr
            print "Iter:\t", epochs

    def setInputs(self, inputs): self.layers[0] = np.array([inputs])
    def getOutputs(self): return self.layers[-1]
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
