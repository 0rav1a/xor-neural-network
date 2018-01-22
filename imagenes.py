#coding:utf-8
from PIL import Image
from Network import Network
import numpy as np

imagenes = open("train-images.idx3-ubyte", "rb")
imagenes.read(16)
labels = open("train-labels.idx1-ubyte", "rb")
labels.read(8)
train = []
test = []

for _ in range(1000):
    img = []
    for _ in range(784):
        img.append(ord(imagenes.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(labels.read(1))] = 1

    train.append([img,lbl])
train = np.array(train)

for _ in range(1000):
    img = []
    for _ in range(784):
        img.append(ord(imagenes.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(labels.read(1))] = 1

    test.append([img,lbl])
test = np.array(test)

print "Entrenando"
net = Network([784,30,10])
net.backprop(train, test)
print "Entrenada"

test = []
error = 0.0
for _ in range(59000):
    img = []
    for _ in range(784):
        img.append(ord(imagenes.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(labels.read(1))] = 1

    test.append([img,lbl])
    
for inputs, targets in test:
    net.setInputs(inputs)
    net.calcOutputs()
    if np.argmax(net.getOutputs()) != np.argmax(targets): error += 1

print error/59000
    
'''
for inputs, targets in test:
    net.setInputs(inputs)
    net.calcOutputs()
    if np.argmax(net.getOutputs()) != np.argmax(targets):
        raw_input()
        img = Image.new("P", (28,28))
        for c in range(28):
            for f in range(28):
                img.putpixel((f,c), inputs[28*c+f])
        print net.getOutputs()
        img.show()
'''
