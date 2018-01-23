#coding:utf-8
from PIL import Image
from Network import Network
import numpy as np

imgTrain = open("train-images-idx3-ubyte", "rb")
imgTrain.read(16)
lblTrain = open("train-labels-idx1-ubyte", "rb")
lblTrain.read(8)
imgTest = open("t10k-images-idx3-ubyte", "rb")
imgTest.read(16)
lblTest = open("t10k-labels-idx1-ubyte", "rb")
lblTest.read(8)

train = []
test = []

print "Leyendo imagenes de entrenamiento"
for _ in range(1000):
    img = []
    for _ in range(784):
        img.append(ord(imgTrain.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(lblTrain.read(1))] = 1

    train.append([img,lbl])
train = np.array(train)

print "Leyendo imagenes de test"
for _ in range(1000):
    img = []
    for _ in range(784):
        img.append(ord(imgTest.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(lblTest.read(1))] = 1

    test.append([img,lbl])
test = np.array(test)

print "Entrenando"
net = Network([784,30,10])
net.backprop(train, test)
print "Entrenada"

errores = [0,0,0,0,0,0,0,0,0,0]
for inputs, targets in test:
    net.setInputs(inputs)
    net.calcOutputs()
    target = np.argmax(targets)
    output = np.argmax(net.getOutputs())
    if target != output:
        img = Image.new("P", (28,28))
        img.putdata([i*255 for i in inputs])
        img.save('errores/' + str(net.getOutputs()) + ".png")
        errores[output] += 1
