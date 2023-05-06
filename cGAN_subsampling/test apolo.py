#%%
import numpy as np
import matplotlib.pyplot as plt
from Deep_Utils import simple_sliding_window, simple_inv_sliding_window
#%% pad zeros to complete the tomogram
mat = np.ones((586,896,960))
num_zeros = 64
pad_width = ((0, 0), (0, 0), (0, num_zeros))
padded_array = np.pad(mat, pad_width, mode='constant', constant_values=0)
print(padded_array.shape)

#%% test for simple sliding window
tomShape = (1,896,1024,1)
tomData = np.ones(tomShape)*255
slidingYSize = 128
slidingXSize = 128
strideY = 128
strideX = 128
slices = simple_sliding_window(tomData,tomShape,slidingYSize,slidingXSize,strideY,strideX)
tomDataOver = simple_inv_sliding_window(slices, tomShape, slidingYSize, slidingXSize, strideY, strideX)
#%%
plt.imshow(padded_array[:,:,1000],vmin=0,vmax=(1))

# np.save('mat',mat)
# x = np.load('mat.npy')
# print(x)
