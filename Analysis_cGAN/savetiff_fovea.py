#%%
import os 
import numpy as np
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
from Deep_Utils import tiff_3Dsave

#%%
path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'

filename = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]_TomoriginalFlat_z=400_x=896_y=960_pol=2'
real = '_real.bin'
imag = '_imag.bin'
nZbin = 400
nXbin = 896
nYbin = 960
npol = 2
tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(100,928,960),order='F')
tom = np.reshape(tom,(nZbin,nXbin,nYbin,npol),order='F')
tom = np.sum(tom,axis=3)

tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(100,928,960),order='F')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin,npol),order='F')
tomi = np.sum(tomi,axis=3)

tomOriginal = abs(tom+1j*tomi)
del tom, tomi
#%%

filename = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]_TomFlat_z=400_x=896_y=960_pol=2'
real = '_real.bin'
imag = '_imag.bin'
nZbin = 400
nXbin = 896
nYbin = 960
npol = 2
tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(100,928,960),order='F')
tom = np.reshape(tom,(nZbin,nXbin,nYbin,npol),order='F')
tom = np.sum(tom,axis=3)

tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(100,928,960),order='F')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin,npol),order='F')
tomi = np.sum(tomi,axis=3)

tomReconstructed = abs(tom+1j*tomi)
del tom, tomi

#%%

tomSubsampled = tomOriginal[:,:,1::2]

#%%
tomSubsampled = np.pad(tomSubsampled, ((0, 0), (0, 0), (240, 240)), mode='constant', constant_values=1)

#%%
original_array = np.array([tomOriginal,
          tomReconstructed,
          tomSubsampled])
del tomOriginal, tomSubsampled, tomReconstructed
#%%
original_array = np.transpose(original_array, (1, 2, 3, 0))
#%%
# Redimensionamos los volúmenes para que tengan tres dimensiones (x, y, 2z o 3z)
volume1 = original_array[:,:,:,0].reshape((400, 896, -1))
volume2 = original_array[:,:,:,1].reshape((400, 896, -1))
volume3 = original_array[:,:,:,2].reshape((400, 896, -1))  # Extrae y redimensiona tomSub
del original_array
#%%
# Concatenamos los volúmenes a lo largo del eje Y
compare = np.concatenate((volume1, volume2, volume3), axis=2)
# compare = np.transpose(compare, (2, 0, 1))
del volume1, volume2, volume3
#%%

filename = 'compare_original_sub_cGAN.tiff'
tiff_3Dsave(10*np.log10(abs(np.float32(compare))**2),os.path.join(path,filename))
