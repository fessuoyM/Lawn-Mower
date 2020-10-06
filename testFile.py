from cnn import CNN
import numpy as np
from PIL import Image

# a = np.array([1.0,2.0,3.0,4.0,5.0])
# b=np.array([1,2,1,2,3,4,3,4,5])
# c=CNN.conv1d(a,b)
# print(a, b, c)
# d=CNN.conv1d(a,b,0)
# e=CNN.conv1d(a,b,2)
# f=CNN.conv1d(a,b,2,1)
# print(d,e,f)

# z=np.array([[ 1.0, 2.0, 3.0],
#             [ 0.0, 0.0, 0.0],
#             [-1.0,-2.0,-3.0]])
# y=np.array([[1,2,3,4,5],
#             [5,4,3,2,1],
#             [1,5,2,4,3],
#             [3,2,4,1,5],
#             [1,1,1,1,1]])
#
# x=CNN.conv2d(z,y)
# print(z)
# print(y)
# print(x)
# w=CNN.conv2d(z,y,0)
# v=CNN.conv2d(z,y,1)
# u=CNN.conv2d(z,y,1,2,0)
# print(w)
# print(v)
# print(u)
# t=CNN.conv2d(z,y,1,2,1)
# print(t)

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
