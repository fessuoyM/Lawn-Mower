import numpy as np

class CNN:
    X=0;
    def __init__(self):
        print("A new CNN class was created!")

    @classmethod
    def conv1d(cls, h, y, pad=None, padVal=0):
        #One Dimensional Convolution
        #Pad is a padding to apply to either side of the signal of value padVal
        #Note that when an argument is passed in for pad, a "truncated" convolution will occur in which only fully overlapped regions are convolved. This feature is more usefull in the conv2d, but available here as well.

        #Initallize paramaters and set up return variable ans
        if h.shape[0]>y.shape[0]: temp = h; h=y; h=temp
        if pad is None:
            pad = h.shape[0]-1
        ans = np.empty((y.shape[0]-h.shape[0]+2*pad+1), h.dtype)

        #Calculating convolution in 3 steps, overlapping, overlapped, and unoverlapping
        if padVal == 0:
            h=np.flip(h)
            for i in range(ans.shape[0]):
                if i+h.shape[0]<=pad:
                    ans[i] = 0
                elif i<pad:
                    ans[i] = np.multiply(h[-(i-pad+h.shape[0]):],y[0:i-pad+h.shape[0]]).sum()
                elif i-pad+h.shape[0]<=y.shape[0]:
                    ans[i] = np.multiply(h,y[i-pad:i-pad+h.shape[0]]).sum()
                elif i<y.shape[0]+pad:
                    ans[i] = np.multiply(h[:y.shape[0]-(i-pad)],y[i-pad:]).sum()#h[:y.shape[0]-1-(i-pad)+1]
                else:
                    ans[i]=0
            return ans
        else:
            temp = y;
            y = padVal*np.ones(temp.shape[0]+2*pad)
            y[pad:y.shape[0]-pad] = temp
            return cls.conv1d(h,y,0)

    @classmethod
    def conv2d(cls, h, y, pad=None, pady=None, padVal=0):
        #Two Dimensional Convolution
        #Pad is a padding to apply to all sides of the signal of value padVal
        #Note that when an argument is passed in for pad, a "truncated" convolution will occur, in which only fully overlapped regions are convolved. This can be useful in giving edge pixel more weight in the convolutions

        #Initallize paramaters and set up return variable ans
        if np.argmin((h.shape[0], y.shape[0]))!=np.argmin((h.shape[1], y.shape[1])):
            raise IndexError("One array must be smaller in size in all dimensions")
        if h.shape[0]>y.shape[0]: temp = h; h=y; h=temp
        if pad is None:
            pady = h.shape[0]-1
            pad = h.shape[1]-1
        if pady is None:
            pady = pad
        ans = np.empty((y.shape[0]-h.shape[0]+2*pady+1,y.shape[1]-h.shape[1]+2*pad+1), h.dtype)

        #Calculating convolution in 3 steps, overlapping, overlapped, and unoverlapping
        if padVal == 0:
            h=np.flipud(h)
            for j in range(ans.shape[0]):
                if j+h.shape[0]<=pady:
                    ans[j,...] = 0
                elif j-pady<0:
                    ans[j,...] = cls.conv1d(h[-1,...],y[j-pady+h.shape[0]-1,...],pad)
                    for i in range(1,j-pady+h.shape[0]):
                        ans[j,...] += cls.conv1d(h[-1-i,...],y[j-pady+h.shape[0]-1-i,...],pad)
                elif j-pady+h.shape[0]<=y.shape[0]:
                    ans[j,...] = cls.conv1d(h[0,...],y[j-pady,...],pad)
                    for i in range(1,h.shape[0]):
                        ans[j,...] += cls.conv1d(h[i,...],y[j-pady+i,...],pad)
                elif j-pady<y.shape[0]:
                    ans[j,...] = cls.conv1d(h[0,...],y[j-pady,...],pad)
                    for i in range(1,y.shape[0]-(j-pady)):
                        ans[j,...] += cls.conv1d(h[i,...],y[j-pady+i,...],pad)
                else:
                    ans[j,...]=0
        else:
            temp = y
            y=padVal*np.ones((y.shape[0]+2*pady,y.shape[1]+2*pad))
            y[pady:y.shape[0]-pady,pad:y.shape[1]-pad] = temp
            return cls.conv2d(h,y,0)

        return ans

    @classmethod
    def normConv(cls, h, y, pad=None, pady=None, padVal=0):
        #Ensure h is the smaller array, and in both dimensions
        if h.ndim==1 and y.ndim==1:
            if h.size>y.size:
                temp = h; h=y; y=temp
        elif h.ndim==2 and y.ndim==2:
            if h.shape[0]>y.shape[0] and h.shape[1]>y.shape[1]:
                temp = h; h=y; y=temp

        #Normallize the kernal to 0-1 range and call ConvTrunc method
        h=h.astype(np.double)
        if h.min()>=0:
            h = np.interp(h, (h.min(), h.max()), (0, 1))
            h = h/h.sum()
        else:
            p=h>0
            h[p] = np.interp(h[p], (h[p].min(), h[p].max()), (0, 1))
            h[p] = h[p]/h[p].sum()
            p = p==False
            h[p] = np.interp(h[p], (h[p].min(), h[p].max()), (-1, 0))
            h[p] = h[p]/np.abs(h[p].sum())
        if h.ndim==1 and y.ndim==1:
            return cls.conv1d(h, y, pad, pady, padVal)
        else:
            return cls.conv2d(h, y, pad, pady, padVal)

    @classmethod
    def convShape(cls, h, y, pad=None, pady=None):
        #Function that returns the shape of arrays returned given the above parameters
        if y.ndim==1:
            if pad is None:
                pad = np.min((h.shape[0], y.shape[0]))-1
            return (y.shape[0]-h.shape[0]+2*pad+1)
        elif y.ndim==2:
            if pad is None:
                pad = np.min((h.shape[1], y.shape[1]))-1
                pady = np.min((h.shape[0], y.shape[0]))-1
            if pady is None:
                pady = pad
            if np.argmin((h.shape[0], y.shape[0]))==np.argmin((h.shape[1], y.shape[1])):
                if h.shape[0]>y.shape[0]:
                    temp = y; y=h; h=temp
                return (y.shape[0]-h.shape[0]+2*pady+1,y.shape[1]-h.shape[1]+2*pad+1)
            else:
                raise IndexError("One array must be smaller in size in all dimensions")
        else:
            raise IndexError("This class can only handle up to 2 dimensions")

    @classmethod
    def toInt(cls, a, type):
        infob = np.iinfo(type)
        return np.interp(a,(a.min(), a.max()), (infob.min, infob.max)).astype(type)
