#%%
import numpy as np
import matplotlib.pyplot as plt
from Deep_Utils import simple_sliding_window, simple_inv_sliding_window
from scipy.io import savemat
#%% pad zeros to complete the tomogram

mat = np.ones((586,896,960),dtype=int)*255

#%%
num_zeros = 64
pad_width = ((0, 0), (0, 0), (0, num_zeros))
mat= np.pad(mat, pad_width, mode='edge')
print(mat.shape)

trimmed_array = mat[:, :, :mat.shape[2]-num_zeros]
print(trimmed_array.shape)
#%% test for simple sliding window

pol = 0
tomData = mat[:,:,:,pol,:]
del mat
#%%
tomShape = np.shape(tomData)
slidingYSize = 128
slidingXSize = 128
strideY = 128
strideX = 128
slices = simple_sliding_window(tomData,tomShape,slidingYSize,slidingXSize,strideY,strideX)
tomDataOver = simple_inv_sliding_window(slices, tomShape, slidingYSize, slidingXSize, strideY, strideX)
#%%
plt.imshow(mat[64,:,:,0,0],vmin=0,vmax=(255))
#%%
mat = np.ones((586,896,896),dtype=int)*255
mat2 = np.ones((586,896,128),dtype=int)*255
#%%

#%%
a = np.ones((4,4))
mdic = {"a": a, "label": "tomData"}
savemat("matlab_matrix.mat", mdic)

