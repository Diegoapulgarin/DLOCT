#%%
import numpy as np
#%%
path = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\No_motion_corrected' # own pc
tomData1 = np.load(path + '\\tomDataOver_586_896_0%3A896_pol1.npy')
tomData2 = np.load(path + '\\tomDataOver_586_896_64%3A960_pol1.npy')
#test