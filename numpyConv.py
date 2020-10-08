import numpy as np

class NumpyConv:
    X=0;
    def __init__(self):
        print("A new CNN class was created!")

    @classmethod
    def conv(cls, h, y, pad=None, padVal=0):
        if h.ndim>y.ndim: temp=h; h=y; y=temp
        if y.ndim==h.ndim:
            ans = np.zeros(cls.convShape(h,y,pad,padVal),h.dtype)
            if h.ndim==1:
                cls.conv1d(ans,h,y,pad,padVal)
            else:
                for i in range(h.ndim):
                    if h.shape[i]>y.shape[i]:
                        raise IndexError("One array must be smaller in size in all dimensions")
                cls.conv_ndim(ans, h,y,pad,padVal)
        elif y.ndim==h.ndim+1:
            ans = np.zeros(np.append(cls.convShape(h,y[...,0],pad,padVal),y.shape[-1]),h.dtype)
            if h.ndim==1:
                for i in range(y.shape[-1]):
                    cls.conv1d(ans[...,i], h,y[...,i],pad,padVal)
            elif h.ndim>1:
                for i in range(y.shape[-1]):
                    cls.conv_ndim(ans[...,i], h,y[...,i],pad,padVal)
            else:
                raise Warning("You must pass in an array with at least 1 dimension")
                return 0
        else:
            raise IndexError("The largest acceptable difference in dimensions is 1")
        return ans

    @classmethod
    def conv1d(cls, ans, h, y, pad=None, padVal=0):
        #One Dimensional Convolution
        #Pad is a padding to apply to either side of the signal of value padVal
        #Note that when an argument is passed in for pad, a "truncated" convolution will occur in which only fully overlapped regions are convolved. This feature is more usefull in the conv2d, but available here as well.

        #Initallize paramaters and set up return variable
        if h.shape[0]>y.shape[0]: temp = h; h=y; y=temp
        if pad is None:
            pad = h.shape[0]-1

        #Calculating convolution in 3 steps, overlapping, overlapped, and unoverlapping
        if pad != 0:
            temp = y
            y = padVal*np.ones(temp.shape[0]+2*pad)
            y[pad:y.shape[0]-pad] = temp
        h=np.flip(h)
        ans+=np.tensordot(np.lib.stride_tricks.as_strided(y, shape=(y.shape[0]-h.shape[0]+1,h.shape[0]),strides=(y.strides*2)),h,([1],[0]))
        return ans

    @classmethod
    def conv_ndim(cls, ans, h, y, pad=None, padVal=0):
        #n Dimensional Convolution
        #Pad is a padding width to apply to all sides of the signal of value padVal
        #Note that when an argument is passed in for pad, a "truncated" convolution will occur, in which only fully overlapped regions are convolved. This can be useful in giving edge pixel more weight in the convolutions

        #Initallize paramaters and set up return variable ans
        if h.shape[0]>y.shape[0]: temp = h; h=y; y=temp
        for i in range(h.ndim):
            if h.shape[i]>y.shape[i]:
                raise IndexError("One array must be smaller in size in all dimensions")

        if pad is None:
            pad = np.empty(h.ndim, np.int)
            for i in range(h.ndim):
                pad[i]=h.shape[i]-1
        elif np.isscalar(pad):
            pad = pad*np.ones(h.ndim, np.int)

        #Iteratively calls itself until its down to one dimension
        if h.ndim>2:
            #Calculating convolution in 3 steps, overlapping, overlapped, and unoverlapping
            if padVal == 0:
                h=np.flip(h,0)
                if pad[0]-h.shape[0]+1>0:
                    ans[:pad[0]-h.shape[0]+1,...] = 0
                if pad[0]-h.shape[0]+1<0:
                    for j in range(0,pad[0]):
                        cls.conv_ndim(ans[j,...], h[-1,...],y[j-pad[0]+h.shape[0]-1,...],pad[1:])
                        for i in range(1,j-pad[0]+h.shape[0]):
                            cls.conv_ndim(ans[j,...], h[-1-i,...],y[j-pad[0]+h.shape[0]-1-i,...],pad[1:])
                else:
                    for j in range(pad[0]-h.shape[0]+1,pad[0]):
                        cls.conv_ndim(ans[j,...], h[-1,...],y[j-pad[0]+h.shape[0]-1,...],pad[1:])
                        for i in range(1,j-pad[0]+h.shape[0]):
                            cls.conv_ndim(ans[j,...], h[-1-i,...],y[j-pad[0]+h.shape[0]-1-i,...],pad[1:])
                for j in range(pad[0],y.shape[0]-h.shape[0]+pad[0]+1):
                    cls.conv_ndim(ans[j,...], h[0,...],y[j-pad[0],...],pad[1:])
                    for i in range(1,h.shape[0]):
                        cls.conv_ndim(ans[j,...], h[i,...],y[j-pad[0]+i,...],pad[1:])
                if ans.shape[0]<=y.shape[0]+pad[0]:
                    for j in range(y.shape[0]-h.shape[0]+pad[0]+1, ans.shape[0]):
                        cls.conv_ndim(ans[j,...], h[0,...],y[j-pad[0],...],pad[1:])
                        for i in range(1,y.shape[0]-(j-pad[0])):
                            cls.conv_ndim(ans[j,...], h[i,...],y[j-pad[0]+i,...],pad[1:])
                else:
                    for j in range(y.shape[0]-h.shape[0]+pad[0]+1, y.shape[0]+pad[0]):
                        cls.conv_ndim(ans[j,...], h[0,...],y[j-pad[0],...],pad[1:])
                        for i in range(1,y.shape[0]-(j-pad[0])):
                            cls.conv_ndim(ans[j,...], h[i,...],y[j-pad[0]+i,...],pad[1:])
                ans[y.shape[0]+pad[0]:,...]=0

            else:
                y=np.pad(y,(np.ones((2,pad.size),np.int)*pad).T,'constant', constant_values=padVal)
                return cls.conv_ndim(ans, h,y,0)
        else:
            if padVal == 0:
                h=np.flip(h,0)
                if pad[0]-h.shape[0]+1>0:
                    ans[:pad[0]-h.shape[0]+1,...] = 0
                if pad[0]-h.shape[0]+1<0:
                    for j in range(0,pad[0]):
                        cls.conv1d(ans[j,...], h[-1,...],y[j-pad[0]+h.shape[0]-1,...],pad[1])
                        for i in range(1,j-pad[0]+h.shape[0]):
                            cls.conv1d(ans[j,...], h[-1-i,...],y[j-pad[0]+h.shape[0]-1-i,...],pad[1])
                else:
                    for j in range(pad[0]-h.shape[0]+1,pad[0]):
                        cls.conv1d(ans[j,...], h[-1,...],y[j-pad[0]+h.shape[0]-1,...],pad[1])
                        for i in range(1,j-pad[0]+h.shape[0]):
                            cls.conv1d(ans[j,...], h[-1-i,...],y[j-pad[0]+h.shape[0]-1-i,...],pad[1])
                for j in range(pad[0],y.shape[0]-h.shape[0]+pad[0]+1):
                    cls.conv1d(ans[j,...], h[0,...],y[j-pad[0],...],pad[1])
                    for i in range(1,h.shape[0]):
                        cls.conv1d(ans[j,...], h[i,...],y[j-pad[0]+i,...],pad[1])
                if ans.shape[0]<=y.shape[0]+pad[0]:
                    for j in range(y.shape[0]-h.shape[0]+pad[0]+1, ans.shape[0]):
                        cls.conv1d(ans[j,...], h[0,...],y[j-pad[0],...],pad[1])
                        for i in range(1,y.shape[0]-(j-pad[0])):
                            cls.conv1d(ans[j,...], h[i,...],y[j-pad[0]+i,...],pad[1])
                else:
                    for j in range(y.shape[0]-h.shape[0]+pad[0]+1, y.shape[0]+pad[0]):
                        cls.conv1d(ans[j,...], h[0,...],y[j-pad[0],...],pad[1])
                        for i in range(1,y.shape[0]-(j-pad[0])):
                            cls.conv1d(ans[j,...], h[i,...],y[j-pad[0]+i,...],pad[1])
                ans[y.shape[0]+pad[0]:,...]=0

            else:
                y=np.pad(y,(np.ones((2,pad.size),np.int)*pad).T,'constant', constant_values=padVal)
                return cls.conv_ndim(ans, h,y,0)
        return ans

    @classmethod
    def normConv(cls, h, y, pad=None, pady=None, padVal=0):
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
        return cls.conv(h, y, pad, pady, padVal)

    @classmethod
    def convShape(cls, h, y, pad=None, pady=None):
        #Function that returns the shape of arrays returned given the above parameters
        if h.ndim==1:
            if pad is None:
                pad = np.min((h.shape[0], y.shape[0]))-1
            return np.array([y.shape[0]-h.shape[0]+2*pad+1])
        elif h.ndim>1:
            if h.shape[0]>y.shape[0]: temp = h; h=y; y=temp
            for i in range(h.ndim):
                if h.shape[i]>y.shape[i]:
                    raise IndexError("One array must be smaller in size in all dimensions")
            if pad is None:
                pad = np.empty(h.ndim, np.int)
                for i in range(h.ndim):
                    pad[i]=h.shape[i]-1
            elif np.isscalar(pad):
                pad = pad*np.ones(h.ndim, np.int)
            return np.subtract(np.add(y.shape,2*pad+1),h.shape)

    @classmethod
    def toInt(cls, a, type):
        infob = np.iinfo(type)
        return np.interp(a,(a.min(), a.max()), (infob.min, infob.max)).astype(type)
