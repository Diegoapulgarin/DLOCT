import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\USER\Documents\GitHub\Experimental_Data_complex\tomogram_no_artifacts\depth_fovea'
real = '\Tom_Real_z=1024_x=1024_y=720.bin'
completePath = path+real
tomReal = np.fromfile(completePath,dtype='single')

imag = '\Tom_Imag_z=1024_x=1024_y=720.bin'
completePath = path+imag
tomImag = np.fromfile(completePath,dtype='single')

tomReal = np.reshape(tomReal,(1024,1024,720),order='F')
tomImag = np.reshape(tomReal,(1024,1024,720),order='F')

tomReal.tofile(path + "\Tom_Real_CC_z=1024_x=1024_y=720.bin")
tomImag.tofile(path + "\Tom_Imag_CC_z=1024_x=1024_y=720.bin")

bscan = 0
z = 0
plot =10*np.log10(abs(tomReal[:,:,bscan]+1j*tomImag[:,:,bscan])**2)
plt.imshow(plot,cmap='gray',vmax=120,vmin=55)