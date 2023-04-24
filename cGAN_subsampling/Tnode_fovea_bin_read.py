#%% import libraries 
import numpy as np
import tifffile as tif
#%% Load bin files

rootFolder = 'D:/DLOCT/TomogramsDataAcquisition/Fovea/TNode/'
fnameTom = 'TNodeIntFlattenRPE.bin' 

""" Shape of each tomogram, as tuples (Z, X, Y)"""
tomShape = [(586,896,960)]

fname = rootFolder + fnameTom

tom = np.fromfile(fname,'single') # quit single for porcine cornea and put single for s_eye
tom = tom.reshape(tomShape[0], order='F')  # reshape using
# Fortran style to import according to MATLAB

#%%
tif.imwrite(path+'\\reconstruct2seye.tif', tomdatapabs)


