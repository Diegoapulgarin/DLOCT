#%%
import numpy as np 
import os
from os.path import join
import matplotlib.pyplot as plt 
#%%

path = r'C:\Users\USER\Documents\data oct\simulados_valen'
path2 = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscanNoartifacts'

filename_tom1_real = 'tom1_real_h1_z=512_x=512_y=25.bin'
filename_tom1_imag = 'tom1_imag_h1_z=512_x=512_y=25.bin'

z = 512
x = 512
y = 25

tom1_real = np.fromfile(join(path,filename_tom1_real),'single')
tom1_imag = np.fromfile(join(path,filename_tom1_imag),'single')

tom1_real = tom1_real.reshape((z,x,y),order='F')
tom1_imag = tom1_imag.reshape((z,x,y),order='F')
tom1 = tom1_real + 1j*tom1_imag
del tom1_real, tom1_imag
tomInt = abs(tom1)**2
plt.imshow(20* np.log10(abs(tom1[:,:,0])))
#%%
newTom = np.zeros((2*z,x,y),dtype=complex)
solap = 64
newTom[solap:512+solap,:,:] = tom1
newTom[512-solap:1024-solap,:,:] = newTom[512-solap:1024-solap,:,:] + np.flip(tom1,axis=0)
plt.imshow(20* np.log10(abs(newTom[:,:,0])))

#%%

fileNameExReal = 'Tom_Real_z=2304_x=1024_y=11.bin'
fileNameExImag = 'Tom_Imag_z=2304_x=1024_y=11.bin'

z1 = 2304
x1 = 1024
y1 = 11

tom2_real = np.fromfile(join(path2,fileNameExReal),'single')
tom2_imag = np.fromfile(join(path2,fileNameExImag),'single')
tom2_real = tom2_real.reshape((z1,x1,y1),order='F')
tom2_imag = tom2_imag.reshape((z1,x1,y1),order='F')
tom2 = tom2_real + 1j*tom2_imag
del tom2_real, tom2_imag
tom2Flip = np.flip(tom2,axis=0)

plt.imshow(20* np.log10(abs(tom2Flip[:,:,0]+tom2[:,:,0])))
