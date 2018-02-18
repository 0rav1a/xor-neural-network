#coding:utf-8
from PIL import Image
from Network import Network
import numpy as np

TRAINS = 60000
TESTS = 10000
SIZE = 784

imgTrain = open("train-images-idx3-ubyte", "rb")
imgTrain.read(16)
lblTrain = open("train-labels-idx1-ubyte", "rb")
lblTrain.read(8)
imgTest = open("t10k-images-idx3-ubyte", "rb")
imgTest.read(16)
lblTest = open("t10k-labels-idx1-ubyte", "rb")
lblTest.read(8)

print "Leyendo imágenes de entrenamiento"
trainInputs = np.array(np.split(np.fromstring(imgTrain.read(TRAINS*SIZE), dtype="uint8")/255.0, TRAINS)).T
print "Leyendo labels de entrenamiento"
trainOutputs = np.zeros((TRAINS, 10)).T
for i,y in enumerate(lblTrain.read(TRAINS)): trainOutputs[ord(y)][i] = 1
print "Leyendo imágenes de test"
testInputs = np.array(np.split(np.fromstring(imgTest.read(TESTS*SIZE), dtype="uint8")/255.0, TESTS)).T
print "Leyendo labels de test"
testOutputs = np.zeros((TESTS, 10)).T
for i,y in enumerate(lblTest.read(TESTS)): testOutputs[ord(y)][i] = 1

print "Entrenando"
net = Network([784,30,10])
net.train(trainInputs, trainOutputs, testInputs, testOutputs)
print "Entrenada"


'''
train = []
test = []

print "Leyendo imagenes de entrenamiento"
for _ in range(60000):
    img = []
    for _ in range(784):
        img.append(ord(imgTrain.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(lblTrain.read(1))] = 1

    train.append([img,lbl])
train = np.array(train)

print "Leyendo imagenes de test"
for _ in range(10000):
    img = []
    for _ in range(784):
        img.append(ord(imgTest.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(lblTest.read(1))] = 1

    test.append([img,lbl])
test = np.array(test)


errores = [0,0,0,0,0,0,0,0,0,0]
for inputs, targets in test:
    net.calcOutputs(inputs)
    target = np.argmax(targets)
    output = np.argmax(net.getOutputs())
    if target != output:
        img = Image.new("P", (28,28))
        img.putdata([i*255 for i in inputs])
        img.save('errores/' + str(np.around(net.getOutputs(), 1)) + ".png")
        errores[output] += 1
'''
