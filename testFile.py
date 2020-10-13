from numpyConv import NumpyConv as cnv
import numpy as np
from PIL import Image
import time



# a = np.array([1.0,2.0,3.0,4.0,5.0])
# b=np.array([1,2,1,2,3,4,3,4,5])
# c=cnv.conv(a,b)
# print(a, b, c)
# z=np.array([[[1,2,3,4,5],np.flip([1,2,3,4,5])],[[1,2,3,4,5],np.flip([1,2,3,4,5])]])
# print(z.shape)
# c=cnv.convManyH(z,b)
# print(c)
# d=cnv.conv(a,b,0)
# e=cnv.conv(a,b,2)
# f=cnv.conv(a,b,2,1)
# print(d,e,f)

# z=np.array([[[ 1, 2, 3],
#             [ 4, 3, 4],
#             [-1,-2,-3]],[[ 1, 0,-1],
#                         [ 2, 0,-2],
#                         [ 3, 0,-3]]])
# y=np.array([[[1,2,3,4,5],
#             [5,4,3,2,1],
#             [1,5,2,4,3],
#             [3,2,4,1,5],
#             [1,1,1,1,1]],[[1,2,3,4,5],
#                         [5,4,3,2,1],
#                         [1,5,2,4,3],
#                         [3,2,4,1,5],
#                         [0,0,1,0,0]]])
#
# x=cnv.conv(z,y,nDim=2)
# print(z)
# print(y)
# print(x)
# # w=cnv.conv(z,y,0)
# # v=cnv.conv(z,y,1)
# # u=cnv.conv(z,y,np.array([1,2]),0)
# # print(w)
# # print(v)
# # print(u)
# # t=cnv.conv(z,y,np.array([1,2]),1)
# # print(t)

# z=np.array([[[ 1, 1, 1],
#             [ 0, 0, 0],
#             [-1,-1,-1]],[[ 1, 1, 1],
#                         [ 0, 0, 0],
#                         [-1,-1,-1]],[[ 1, 1, 1],
#                                     [ 0, 0, 0],
#                                     [-1,-1,-1]],[[ 1, 1, 1],
#                                                 [ 0, 0, 0],
#                                                 [-1,-1,-1]],[[ 1, 0,-1],
#                                                             [ 1, 0,-1],
#                                                             [ 1, 0,-1]]])
# y=np.array([[[1,2,3,4,5],
#             [5,4,3,2,1],
#             [1,2,3,4,5],
#             [5,4,3,2,1],
#             [1,5,2,4,3]],[[1,5,1,5,1],
#                         [2,4,2,4,5],
#                         [3,3,3,3,2],
#                         [4,2,4,2,4],
#                         [5,1,5,1,3]],[[1,2,3,4,5],
#                                     [5,4,3,2,1],
#                                     [1,2,3,4,5],
#                                     [5,4,3,2,1],
#                                     [1,5,2,4,3]],[[1,5,1,5,1],
#                                                 [2,4,2,4,5],
#                                                 [3,3,3,3,2],
#                                                 [4,2,4,2,4],
#                                                 [5,1,5,1,3]],[[1,2,3,4,5],
#                                                             [5,4,3,2,1],
#                                                             [1,2,3,4,5],
#                                                             [5,4,3,2,1],
#                                                             [1,5,2,4,3]]])
#
# x=cnv.conv(z,y)
# print(z)
# print(y)
# print(x)
# w=cnv.conv(z,y,0)
# v=cnv.conv(z,y,1)
# u=cnv.conv(z,y,np.array([1,2,2]),0)
# print(w)
# print(v)
# print(u)
# t=cnv.conv(z,y,np.array([1,2,2]),1)
# print(t)

im = Image.open("TestImages/TI_0001.jpg")
print(im.format, im.size, im.mode)
# im.show()

imArr = np.moveaxis(np.array(im),2,0)
print(imArr.shape)

kernals=np.array([[ [ 2, 2, 2, 2, 2],
                    [ 1, 1, 1, 1, 1],
                    [ 0, 0, 0, 0, 0],
                    [-1,-1,-1,-1,-1],
                    [-2,-2,-2,-2,-2]],[ [ 2, 1, 0,-1,-2],
                                        [ 2, 1, 0,-1,-2],
                                        [ 2, 1, 0,-1,-2],
                                        [ 2, 1, 0,-1,-2],
                                        [ 2, 1, 0,-1,-2]]])


# numR = 50
# imArrFil = cnv.conv(kernals,imArr, 0)
# tic = time.perf_counter()
# for i in range(numR):
#     imArrFil = cnv.conv(kernals,imArr, 0, nDim=2)
# toc = time.perf_counter()
# print(f"Conv: {(toc - tic)/(numR*2):0.4f} seconds")

imArrFil = cnv.conv(kernals,imArr, 0, nDim=2)
imFil = Image.fromarray(np.moveaxis(cnv.toInt(np.abs(imArrFil[0,...]),imArr.dtype),0,2))
imFil.show()
imFil.save("TestImages/HImage.jpg")
imFil = Image.fromarray(np.moveaxis(cnv.toInt(np.abs(imArrFil[1,...]),imArr.dtype),0,2))
imFil.show()
imFil.save("TestImages/VImage.jpg")

greyImArr =np.array(im.convert('L'))
greyImArrFil = cnv.conv(kernals,greyImArr, 0)
GIA=np.mean(greyImArrFil,0)
imFil = Image.fromarray(cnv.toInt(np.abs(GIA),imArr.dtype),'L')
imFil.show()
imFil.save("TestImages/VHImage.jpg")
