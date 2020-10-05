import numpy as np
class CNN:
    X=0;
    def __init__(self):
        print("A new CNN class was created!")

    @classmethod
    def conv(cls, x, y):
        #One dimesional Convolution
        if x.ndim==1 and y.ndim==1:
            #Prepping ans variable, ensuring h is the flipped version of the smaller array
            ans = np.zeros(x.size+y.size-1)
            if x.size>y.size: temp = x; x=y; x=temp
            h=np.flip(x)

            #Calculating convolution in 3 steps, overlapping, overlapped, and unoverlapping
            for i in range(h.size-1): #Overlapping
                ans[i]=np.multiply(h[h.size-i-1:],y[:i+1]).sum()
            ans[h.size-1:y.size] = cls.convTrunc(x,y) #Overlapped
            for i in reversed(range(h.size-1)): #UnOverlapping
                ans[h.size-1-i+y.size-1] = np.multiply(h[0:i+1],y[y.size-1-i:]).sum()
            return ans

        #Two Dimensional Array
        elif x.ndim==2 and y.ndim==2:
            #Checks that one matrix is smaller in both dimesions and swaps if neccessary
            #Calls convTrunc method with extra paddings of zero to take care of regions
            #where full overlap does not occur
            if x.shape[0]>y.shape[0] and x.shape[1]>y.shape[1]: #Swap to ensure h is smaller
                ans = cls.convTrunc(y,x,y.shape[0]-1, 0, y.shape[1]-1)
                return ans
            elif x.shape[0]<=y.shape[0] and x.shape[1]<=y.shape[1]:
                ans = cls.convTrunc(x,y,x.shape[0]-1, 0, x.shape[1]-1)
                return ans
            else:
                raise ValueError("One array must be smaller in size in all dimensions")


    @classmethod
    def convTrunc(cls, h, y, pad=None, padVal=0, pady=None):
        #One dimensional convolution only considering fully overlapped regions
        #Optional padding size to give earlier and later points more weights
        if h.ndim==1 and y.ndim==1:
            #Flip h and y if h is the larger array
            if h.size>y.size:
                temp = h; h=y; y=temp
            #Add padding if argument was passed in and greater than 0
            if pad is not None and pad>0:
                temp = y;
                y = padVal*np.ones(temp.size+2*pad)
                y[pad:y.size-pad] = temp
            #Reverse h for flip and slip and calculate sliding weighted sum
            h=np.flip(h)
            ans = np.zeros(y.size-h.size+1)
            for i in range(y.size-h.size+1): #Overlapped
                ans[i] = np.multiply(h,y[i:i+h.size]).sum()
            return ans

        #Two dimensional convolution only considering fully overlapped regions
        #Optional padding size to give earlier and later points more weights
        elif h.ndim==2 and y.ndim==2:
            #Ensure h is the smaller array, and in both dimensions; otherwise return error
            if h.shape[0]>y.shape[0] and h.shape[1]>y.shape[1]:
                temp = h; h=y; y=temp
            if h.shape[0]<=y.shape[0] and h.shape[1]<=y.shape[1]:
                #Add padding if arguments passed in
                if pad is not None:
                    if pad<0: raise Warning("Padding cannot be negative, Padding of 0 was used instead");
                    if pady == None:
                        pady = pad
                    else:
                        if pad<0: raise Warning("Padding cannot be negative, Padding of 0 was used instead");
                    temp = y
                    y = padVal*np.ones((y.shape[0]+2*pad, y.shape[1]+2*pady))
                    y[pad:y.shape[0]-pad,pady:y.shape[1]-pady] = temp

                #Flip matrix for flip and slip across both axis
                h=np.flipud(h)
                h=np.fliplr(h)
                ans = np.zeros((y.shape[0]-h.shape[0]+1, y.shape[1]-h.shape[1]+1))

                #Conduct sliding wieghted sum
                for i in range(y.shape[0]-h.shape[0]+1):
                    for j in range(y.shape[1]-h.shape[1]+1):
                        ans[i][j] = np.multiply(h,y[i:i+h.shape[0],j:j+h.shape[1]]).sum()
                return ans
            else:
                raise ValueError("One array must be smaller in size in all dimensions")


    @classmethod
    def normConv(cls, h, y, pad=None, padVal=0, pady=None):
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
        return cls.convTrunc(h, y, pad, padVal, pady)
