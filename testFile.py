from cnn import CNN
import numpy as np
from PIL import Image

im = Image.open("TestImages/TI_0001.jpg")
print(im.format, im.size, im.mode)
im.show()

imArr = np.array(im)

kernal = np.array([ [ 1.0, 1.0, 1.0, 1.0, 1.0],
                    [ 0.5, 0.5, 0.5, 0.5, 0.5],
                    [ 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.5,-0.5,-0.5,-0.5,-0.5],
                    [-1.0,-1.0,-1.0,-1.0,-1.0]])
imShape = CNN.convShape(imArr, kernal)

imArrFil = np.empty((imShape[0],imShape[1],3))
for i in range(3):
    imArrFil[...,i] = CNN.normConv(imArr[...,i], kernal)

imFil = Image.fromarray(CNN.toInt(np.abs(imArrFil),imArr.dtype))
imFil.show()
imFil.save("TestImages/VImage.jpg")


kernal = kernal.T
imShape = CNN.convShape(imArr, kernal)

imArrFil = np.empty((imShape[0],imShape[1],3))
for i in range(3):
    imArrFil[...,i] = CNN.normConv(imArr[...,i], kernal)

imFil = Image.fromarray(CNN.toInt(np.abs(imArrFil),imArr.dtype))
imFil.show()
imFil.save("TestImages/HImage.jpg")
