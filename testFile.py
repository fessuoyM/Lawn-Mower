from cnn import CNN
import numpy as np

a = np.arange(5)+1
b = np.array([1,2,1,2,3,4,3,4,5])

print(a)
print(b)
print(CNN.conv(a,b))
print(CNN.convTrunc(a,b,2))

a = np.array([[1,2,3,4],[4,3,2,1],[2,1,3,4],[4,1,3,2]])
b = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

print(a)
print(b)
print(CNN.conv(a,b))
print(CNN.convTrunc(a,b))
print(CNN.normConv(a,b))
