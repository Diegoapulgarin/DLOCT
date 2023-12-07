#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
path = r'C:\Users\USER\Documents\GitHub\Experimental_Data_complex\depth_nail'
real = '\Tom_Real_z=2048_x=1152_y=512.bin'
completePath = path+real
tomReal = np.fromfile(completePath,dtype='single')
imag = '\Tom_Imag_z=2048_x=1152_y=512.bin'
completePath = path+imag
tomImag = np.fromfile(completePath,dtype='single')
tomReal = np.reshape(tomReal,(2048,1152,512),order='F')
tomImag = np.reshape(tomReal,(2048,1152,512),order='F')
#%%
tomReal2 = tomReal[380:380+1024,0:1024,:]
tomImag2 = tomImag[380:380+1024,0:1024,:]
# %%
tomReal2.tofile("Tom_Real_CC_z=1024_x=1024_y=512.bin")
tomImag2.tofile("Tom_Imag_CC_z=1024_x=1024_y=512.bin")
#%%
bscan = 256
z = 524
plot =10*np.log10(abs(tomReal2[z,:,:]+1j*tomImag2[z,:,:])**2)
plt.imshow(plot,cmap='gray',vmax=120,vmin=80)

plot2 =10*np.log10(abs(tomReal2[:,:,bscan]+1j*tomImag2[:,:,bscan])**2)
plt.imshow(plot2,cmap='gray',vmax=120,vmin=80)