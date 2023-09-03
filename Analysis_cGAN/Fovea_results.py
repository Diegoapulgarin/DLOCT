'''
Z depth = 3.9 um for each pixel
X depth = 14 um for each pixel
Y depth = 28 um for each pixel
'''
#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
from Deep_Utils import Correlation
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import scipy.io as sio

#%%
CentralWavelength = 870e-9
bandwith = 50e-9
pixel = (2*np.log(2)/np.pi)*(CentralWavelength**2/bandwith)

#%%
path = r'C:\Users\USER\Documents\GitHub\Fovea'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_Tomint_z=(295..880)_x=(65..960)_y=(1..960)_reconstructed.bin'
tom = np.fromfile(path+'\\'+filename,'single')
nZbin = 586
nXbin = 896
nYbin = 960
tomReconstructed = np.reshape(tom,(nZbin,nXbin,nYbin,2),order='F')
tomReconstructed = np.sum(tomReconstructed,axis=3)
del tom
#%%
# path = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected\Final\Int_8x8x8x0_3x3x3x0_150_0_50_unitary'
# filename = 'TNodeIntFlattenRPE.bin'
# tom = np.fromfile(path+'\\'+filename,'single')
# xSlice = 587
# ySlice = 587
# zSlice = 586
# tomTNode = np.reshape(tom,(zSlice,xSlice,ySlice),order='F')
# del tom

#%%
path = r'C:\Users\USER\Documents\GitHub\Fovea'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)'
real = '_real.bin'
imag = '_imag.bin'
tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(100,928,960),order='F')
tom = np.reshape(tom,(nZbin,nXbin,nYbin,2),order='F')
tom = np.sum(tom,axis=3)

tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(100,928,960),order='F')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin,2),order='F')
tomi = np.sum(tomi,axis=3)

tomOriginal = np.stack((tom, tomi), axis=3)
del tom, tomi
#%%
# n=190
# fig = px.imshow(10*np.log10(tomOriginal[n,155:741,187:773]),color_continuous_scale='gray',zmin=80,zmax=115)
# fig.show()
# fig.write_html('original.html')
# #%%
# n = 190
# fig = px.imshow(10*np.log10(tomTNode[n,:,:]),color_continuous_scale='gray',zmin=70,zmax=160)
# fig.show()
# # fig.write_html('Tnode.html')
# #%%
# n = 190
# out = r'C:\Data\partial results'
# plot1 = 10*np.log10(tomOriginal[:,155:742,187+n])
# plot2 = 10*np.log10(tomTNode[:,:,n])
# #%%
# # create_and_save_subplot(plot2,plot1,
# #                         'Tomogram with TNode',
# #                         'Original Tomogram',
# #                         output_path=out,
# #                         zmin=70,zmax=160,
# #                         file_name='Fovea compare with TNode')

#%%
z = 400
# plot1 = 10*np.log10(tomOriginal[n,155:742,186:773])
plot1 = 10*np.log10(abs(tomReconstructed[:,:,z]))
plot2 = 10*np.log10(abs(tomOriginal[:,:,z,0]+1j*tomOriginal[:,:,z,1])**2)
plt.imshow(plot2, cmap='gray',vmin=85,vmax=110)
#%%

fig, ax = plt.subplots()
ax.imshow(plot2, cmap='gray',vmin=81,vmax=110)  # Puedes cambiar 'viridis' por el colormap que prefieras.
# Elimina los ejes y bordes blancos
ax.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

# Guarda la imagen
plt.savefig("ZX_Fovea_Original.png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()

#%%
path = r'C:\Users\USER\Documents\GitHub\Fovea'
filename = 'tomDataOverpol0.mat'
mat_contents = sio.loadmat(path+'/'+filename)
tompol0 = mat_contents['tomDataOver']
filename = 'tomDataOverpol1.mat'
mat_contents = sio.loadmat(path+'/'+filename)
tompol1 = mat_contents['tomDataOver']
tomReconstructed = np.stack((tompol0,tompol1),axis=3)
tomReconstructed = np.sum(tomReconstructed,axis=3)
del mat_contents, tompol0, tompol1
#%%
z = 170
enfaceReconstructedReal = np.real(tomReconstructed[z,:,:])
enfaceReconstructedImag=np.imag(tomReconstructed[z,:,:])
enfaceReconstructed = np.stack((enfaceReconstructedReal,enfaceReconstructedImag),axis=2)
correlationx,correlationy = Correlation(enfaceReconstructed,savename='Z170_reconstructed')

