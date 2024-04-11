#%%
import os
import numpy as np
import matplotlib.pyplot as plt
path = r'D:\DLOCT\TomogramsDataAcquisition\[DepthWrap]\[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]'
file = 'tomDataOver_z=560_x=1024_y=512.npy'
tomOver = np.load(os.path.join(path,file))
#%%
n = 20
plot = 10*np.log10(abs(tomOver[n,:,:])**2)
plt.imshow(plot)