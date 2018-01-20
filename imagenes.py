#coding:utf-8
from PIL import Image
from NPNetwork import NPNetwork
import numpy as np

imagenes = open("train-images.idx3-ubyte", "rb")
imagenes.read(16)
labels = open("train-labels.idx1-ubyte", "rb")
labels.read(8)
test = []

for _ in range(1000):
    img = []
    for _ in range(784):
        img.append(ord(imagenes.read(1))/255.0)
    lbl = [0,0,0,0,0,0,0,0,0,0]
    lbl[ord(labels.read(1))] = 1

    test.append([img,lbl])

test = np.array(test)

net = NPNetwork([784,30,10])
net.backprop(test)

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























raw_input("FIN")

for _ in range(20):
    img = Image.new("P", (28,28))
    for c in range(28):
        for f in range(28):
            img.putpixel((f,c), ord(imagenes.read(1)))
    img.show()
    raw_input(ord(labels.read(1)))


image = Image.open("4.png")
img = image.load()

imgList = []
for f in range(image.size[0]):
    for c in range(image.size[1]):
        if img[f,c][0] == 0: imgList.append(1)
        elif img[f,c][0] == 255: imgList.append(0)

net = NPNetwork([8100,8100,1])
net.backprop([(imgList,[1])])
