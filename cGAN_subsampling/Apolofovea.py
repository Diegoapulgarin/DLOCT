#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
path = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\No_motion_corrected' # own pc
# tomData = np.load(path + '\\tomDataOver_Fovea.npy')
tomData2 = np.load(path + '\\tomDataOver_586_896_0%3A896_pol1.npy')
#%%
z = 895
plot = 10*np.log(abs(10*np.log(tomData2[:,z,:,0]+1j*tomData2[:,z,:,1]))**2)
plt.imshow(plot)
