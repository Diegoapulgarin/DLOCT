#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
import os
from Deep_Utils import create_and_save_subplot, tiff_3Dsave,save_image
import plotly.express as px
import matplotlib.pyplot as plt
#%%
path = r'C:\Users\USER\Documents\GitHub\Fovea\[p.SHARP][s.Eye2a][10-09-2019_13-14-42]\z=(1..586)_x=(1..896)_y=(1..960)-nSpec='
file = 'TNodeIntFlattenRPE.bin'
nZBin = 586
nXBin = 896
nYBin = 960
tomint =[] 
os.chdir(path)
for filename in os.listdir(os.getcwd()):
    tom = np.fromfile(path+'\\'+filename+'\\'+file,'single')
    print(path+'\\'+filename+'\\'+file,'single')
    tom = tom.reshape((nZBin,nXBin,nYBin),order='F')
    tomint.append(tom)
    del tom
tomint = np.array(tomint) 
#%% sub sampled
path = r'C:\Users\USER\Documents\GitHub\Fovea\[p.SHARP][s.Eye2a][10-09-2019_13-14-42]\z=(1..586)_x=(1..512)_y=(1..960)-nSpec=\Int_8x8x8x0_3x3x3x0_250_0_50_unitary'
file = 'TNodeIntFlattenRPE.bin'
nZBin = 586
nXBin = int(896/2)
nYBin = 960
tom = np.fromfile(path+'\\'+file,'single')
tomSub = tom.reshape((nZBin,nXBin,nYBin),order='F')
del tom
#%%
tomSub = np.pad(tomSub, ((0, 0), (224, 224), (0, 0)), mode='constant', constant_values=1)
tomint = np.concatenate((tomint, tomSub[np.newaxis, :, :, :]), axis=0)
del tomSub
#%%
thisbscan=308
plot_cGAN = 10*np.log10(abs(tomint[1,:,:,thisbscan])**2)
plot_orig = 10*np.log10(abs(tomint[0,:,:,thisbscan])**2)
vmin = 165
vmax = 235
plt.imshow(plot_cGAN,cmap='gray',vmin = vmin, vmax=vmax)
save_image(plot_cGAN,file_name='TNode_cGAN_ZX308',vmin=vmin,vmax=vmax)
print('cGAN saved')
save_image(plot_orig,file_name='TNode_orig_ZX308',vmin=vmin,vmax=vmax)
print('original saved')
#%%
thisbscan=int(308)
vmin = 165
vmax = 235
plot_sub = 10*np.log10(abs(tomSub[:,:,thisbscan])**2)
plt.imshow(plot_sub,cmap='gray',vmin = vmin, vmax=vmax)
save_image(plot_sub,file_name='TNode_sub_ZX308',vmin=vmin,vmax=vmax)
#%%
# zmax=250
# zmin = 160
# file = 'ZX_256'
# output = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\partial_results'
# create_and_save_subplot(plot_cGAN,plot_orig,
#                         title1='Resampled with cGAN and TNode',
#                         title2='original with TNode',
#                         output_path=output
#                         ,zmax=zmax,zmin=zmin,
#                         file_name=file)
#%%
thisbscan=190
plot_cGAN = 10*np.log10(abs(tomint[1,thisbscan,:,:])**2)
plot_orig = 10*np.log10(abs(tomint[0,thisbscan,:,:])**2)
vmin = 165
vmax = 235
plt.imshow(plot_orig,cmap='gray',vmin = vmin, vmax=vmax)
save_image(plot_cGAN,file_name='TNode_cGAN_XY190',vmin=vmin,vmax=vmax)
print('cGAN saved')
save_image(plot_orig,file_name='TNode_orig_XY190',vmin=vmin,vmax=vmax)
print('original saved')
#%%
thisbscan=int(190)
vmin = 165
vmax = 235
plot_sub = 10*np.log10(abs(tomSub[thisbscan,:,:])**2)
plt.imshow(plot_sub,cmap='gray',vmin = vmin, vmax=vmax)
save_image(plot_sub,file_name='TNode_sub_XY190',vmin=vmin,vmax=vmax)


#%%
# zmax=250
# zmin = 160
# file = 'XY_256'

# create_and_save_subplot(plot_cGAN,plot_orig,
#                         title1='Resampled with cGAN and TNode',
#                         title2='original with TNode',
#                         output_path=output
#                         ,zmax=zmax,zmin=zmin,
#                         file_name=file)
#%%
thisbscan=180
plot_cGAN = 10*np.log10(abs(tomint[1,:,thisbscan,:])**2)
plot_orig = 10*np.log10(abs(tomint[0,:,thisbscan,:])**2)
vmin = 165
vmax = 235
plt.imshow(plot_orig,cmap='gray',vmin = vmin, vmax=vmax)
save_image(plot_cGAN,file_name='TNode_cGAN_ZY180',vmin=vmin,vmax=vmax)
print('cGAN saved')
save_image(plot_orig,file_name='TNode_orig_ZY180',vmin=vmin,vmax=vmax)
print('original saved')
#%%

thisbscan=int(180/2)
vmin = 165
vmax = 235
plot_sub = 10*np.log10(abs(tomSub[:,thisbscan,:])**2)
plt.imshow(plot_sub,cmap='gray',vmin = vmin, vmax=vmax)
save_image(plot_sub,file_name='TNode_sub_ZY180',vmin=vmin,vmax=vmax)

# zmax=250
# zmin = 160
# file = 'ZY_256'

# create_and_save_subplot(plot_cGAN,plot_orig,
#                         title1='Resampled with cGAN and TNode',
#                         title2='original with TNode',
#                         output_path=output
#                         ,zmax=zmax,zmin=zmin,
#                         file_name=file)
#%%
original_array = np.transpose(tomint, (1, 2, 3, 0))

# Redimensionamos los volúmenes para que tengan tres dimensiones (x, y, 2z o 3z)
volume1 = original_array[:,:,:,0].reshape((586, 896, -1))
volume2 = original_array[:,:,:,1].reshape((586, 896, -1))
volume3 = original_array[:,:,:,2].reshape((586, 896, -1))  # Extrae y redimensiona tomSub

# Concatenamos los volúmenes a lo largo del eje Y
compare = np.concatenate((volume1, volume2, volume3), axis=1)
compare = np.transpose(compare, (2, 0, 1))

# Liberar memoria
del volume1, volume2, volume3
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
output = r'C:\Users\USER\Documents\GitHub\Fovea'
filename = '\compare_original_sub_cGAN.tiff'
tiff_3Dsave(10*np.log10(abs(compare)**2),output+filename)

#%%

