import numpy as np
from scipy.io import savemat
path = '/home/dapulgaris/data/tomDataOver_Fovea_pol'
ext = '.npy'
pol = ['1','2']

for i in pol:
    np.load(path+i+ext)
    
    
    print(path+i+ext)
