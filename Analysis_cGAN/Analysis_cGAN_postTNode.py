#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
import os
from Deep_Utils import create_and_save_subplot, tiff_3Dsave
#%%
path = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected\Final'
file = 'TNodeIntFlattenRPE.bin'
nXBin = 587
nYBin = 587
nZBin = 586
tomint =[] 
os.chdir(path)
for filename in os.listdir(os.getcwd()):
    tom = np.fromfile(path+'\\'+filename+'\\'+file,'single')
    print(path+'\\'+filename+'\\'+file,'single')
    tom = tom.reshape((nZBin,nXBin,nYBin),order='F')
    tomint.append(tom)
    del tom
tomint = np.array(tomint) # 
#%%
thisbscan=256
plot_cGAN = 10*np.log10(abs(tomint[1,:,:,thisbscan]))
plot_orig = 10*np.log10(abs(tomint[0,:,:,thisbscan])**2)
zmax=250
zmin = 160
file = 'ZX_256'
output = r'C:\Data\partial_results'
create_and_save_subplot(plot_cGAN,plot_orig,
                        title1='Resampled with cGAN and TNode',
                        title2='original with TNode',
                        output_path=output
                        ,zmax=zmax,zmin=zmin,
                        file_name=file)
#%%
thisbscan=170
plot_cGAN = 10*np.log10(abs(tomint[1,thisbscan,:,:]))
plot_orig = 10*np.log10(abs(tomint[0,thisbscan,:,:])**2)
zmax=250
zmin = 160
file = 'XY_256'

create_and_save_subplot(plot_cGAN,plot_orig,
                        title1='Resampled with cGAN and TNode',
                        title2='original with TNode',
                        output_path=output
                        ,zmax=zmax,zmin=zmin,
                        file_name=file)
#%%
thisbscan=160
plot_cGAN = 10*np.log10(abs(tomint[1,:,thisbscan,:]))
plot_orig = 10*np.log10(abs(tomint[0,:,thisbscan,:])**2)
zmax=250
zmin = 160
file = 'ZY_256'

create_and_save_subplot(plot_cGAN,plot_orig,
                        title1='Resampled with cGAN and TNode',
                        title2='original with TNode',
                        output_path=output
                        ,zmax=zmax,zmin=zmin,
                        file_name=file)
#%%
#%%
original_array = np.transpose(tomint, (1, 2, 3, 0))

# Redimensionamos los volúmenes para que tengan tres dimensiones (x, y, 2z)
volume1 = original_array[:,:,:,0].reshape((586, 587, -1))
volume2 = original_array[:,:,:,1].reshape((586, 587, -1))

# Concatenamos los volúmenes a lo largo del eje Y
compare = np.concatenate((volume1**2, volume2), axis=1)
compare = np.transpose(compare,(2,0,1))
del volume1, volume2
#%%
import plotly.express as px
#%%
# plot_test = 10*np.log10(abs(compare[240,:,:])**2)
plot_cGAN = 10*np.log10((tomint[1,thisbscan,:,:]))
fig = px.imshow(plot_cGAN,color_continuous_scale='gray')
fig.show()
#%%
filename = '\cGANtomintTNode.tiff'
tiff_3Dsave(10*np.log10(tomint[1,:,:,:]),output+filename)
filename = '\OriginaltomintTNode.tiff'
tiff_3Dsave(10*np.log10(tomint[0,:,:,:]),output+filename)
#%%
filename = '\compare_original_cGAN.tiff'
tiff_3Dsave(10*np.log10((compare)),output+filename)