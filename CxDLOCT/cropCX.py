#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
path = r'C:\Users\USER\Documents\GitHub\Experimental_Data_complex\depth_wrap_nail'
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
tomImag = np.reshape(tomReal,(3584,1152,512),order='F')
tomReal = tomReal[1280:2304,0:1024,:]
tomImag = tomImag[1280:2304,0:1024,:]
# %%
tomReal.tofile("Tom_Real_CC_z=1024_x=1024_y=512.bin")
tomImag.tofile("Tom_Imag_CC_z=1024_x=1024_y=512.bin")
#%%
bscan = 256
z = 524
plot =10*np.log10(abs(tomReal[z,:,:]+1j*tomImag[z,:,:])**2)
plt.imshow(plot,cmap='gray',vmax=120,vmin=80)

plot2 =10*np.log10(abs(tomReal[:,:,bscan]+1j*tomImag[:,:,bscan])**2)
plt.imshow(plot2,cmap='gray',vmax=120,vmin=80)



