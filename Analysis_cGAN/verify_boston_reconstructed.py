#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

path = r'D:\DLOCT\ultimo experimento subsampling paper\Fovea'
file = 'tomFlatDataOver_z=400_x=896_y=960_pol1.npy'
# file2 = 'tomDataOver_z=560_x=1024_y=1024_pol2.npy'
# file2 = 'tomDataOver_z=560_x=1024_y=1024.npy'
# file3 = 'Tom_z=1152_x=1024_y=1024.npy'
# tomOver = np.load(os.path.join(path,file))
tomOver1 = np.load(os.path.join(path,file))
# tomOver2 = np.load(os.path.join(path,file2))
# tomOver = tomOver1 + 0.5*tomOver2
# tomdatas2 = np.load(os.path.join(path,file3))
# tomdatas2 = tomdatas2[400:960,:,:]
#%%
# fnameTom = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomFlat_z=400_x=896_y=960_pol=2' # fovea
# tomShape = [(400,896,960,2)]# porcine cornea
# fname = os.path.join(path, fnameTom)
# # Names of all real and imag .bin files
# fnameTomReal = [fname + '_real.bin' ]
# fnameTomImag = [fname + '_imag.bin' ]
# tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
# tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using

# tomImag = np.fromfile(fnameTomImag[0],'single')
# tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using

# tomDatas = np.stack((tomReal,tomImag), axis=4)
# del tomImag, tomReal
# tomDatas = tomDatas[:,:,:,:,0] + 1j* tomDatas[:,:,:,:,1]
#%%
# tomOver1 = tomOver1[:,:,:,0]+1j*tomOver1[:,:,:,1]
#%%
save_path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\partial_results\segunda revisión\foveaflat'
dpi = 300
n = 150
plot = 10*np.log10(abs(tomOver1[:,:,n])**2)
plt.rcParams['figure.dpi']=dpi
plt.imshow(plot,cmap='gray',vmax=120,vmin=60)
plt.axis('off')
plt.savefig(os.path.join(save_path,f'nofoveaFlat_reconstructed_y={n}_{np.shape(plot)[0]}x{np.shape(plot)[1]}')
            , dpi=dpi, format=None, metadata=None,
            bbox_inches='tight', pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None,
        )
#%%
