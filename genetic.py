#coding:utf-8
import time
from Network import Network

SIZE = 100 #Tamaño de cada generación
UNFIT = 0.5 #Porcentaje de individuos de cada generación que muere en el proceso de selección
ERROR = 0.0001 #Error mínimo admisible de la red
MUTATION = 0.3 #Probabilidad de mutación
#TEST = Lista de pares [entrada, salida esperada] para entrenar la red
XOR = (([0,0],[0]),
       ([0,1],[1]),
       ([1,0],[1]),
       ([1,1],[0]))
         
CONT = (([0,0,0],[0,0,1]),
        ([0,0,1],[0,1,0]),
        ([0,1,0],[0,1,1]),
        ([0,1,1],[1,0,0]),
        ([1,0,0],[1,0,1]),
        ([1,0,1],[1,1,0]),
        ([1,1,0],[1,1,1]),
        ([1,1,1],[0,0,0]))

def getFittest(generation):
    '''Devuelve el individuo más apto de una generación
    '''
    fittest = generation[0]
    for individual in generation[1:]:
        if individual.getError() < fittest.getError(): fittest = individual
        
    return fittest
    
def getLessFit(generation):
    '''Devuelve el individuo menos apto de una generación
    '''
    lessfit = generation[0]
    for individual in generation[1:]:
        if individual.getError() > lessfit.getError(): lessfit = individual
        
    return lessfit
    
def evaluation(generation, tests):
    '''Evalúa el error de cada individuo de la población
    '''
    for individual in generation:
        individual.calcError(tests)
    
def selection(generation):
    '''Un porcentaje de los individuos menos aptos muere
    '''
    for _ in range(int(SIZE*UNFIT)):
        generation.remove(getLessFit(generation))
    
def getNewGeneration(generation = None):
    '''Devuelve una nueva generación producto de evolucionar la dada. Si no se da ninguna, se genera con individuos aleatorios
    '''
    newGeneration = []
    
    if (not generation):
        for i in range(SIZE):
            newGeneration.append(Network())
    
    else:
        selection(generation)

        for individual in generation:
            newGeneration.append(individual)
            newGeneration.append(individual.clone())
            newGeneration[-1].mutate(MUTATION)
        
    return newGeneration
    
###MAIN
net = Network(len(XOR[0][0]), len(XOR[0][1]))

generation = getNewGeneration()
evaluation(generation, XOR)
while (getFittest(generation).getError() > ERROR): #Mientras la generación no esté completamente evolucionada, se generará una nueva
    generation = getNewGeneration(generation)
    evaluation(generation, XOR)