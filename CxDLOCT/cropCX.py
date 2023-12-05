#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
path = r'C:\Users\USER\Documents\GitHub\depth_chicken_breast'
real = '\Tom_Real_CC_z=3584_x=1152_y=512.bin'
completePath = path+real
tomReal = np.fromfile(completePath,dtype='single')
#%%
imag = '\Tom_Imag_CC_z=3584_x=1152_y=512.bin'
completePath = path+imag
tomImag = np.fromfile(completePath,dtype='single')
#%%
# tomImag=
#%%
tomReal = np.reshape(tomReal,(3584,1152,512),order='F')