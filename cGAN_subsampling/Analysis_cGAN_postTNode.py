#%%
import numpy as np
import os
from Deep_Utils import create_and_save_subplot, tiff_3Dsave
#%%
path = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\No_motion_corrected\Final para analizar\z=(1..586)_x=(155..741)_y=(187..773)-nSpec=cGAN'
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
tomint = np.array(tomint) # index 0 = cGAN, index 1 = original
#%%
thisbscan=256
plot_cGAN = 10*np.log10(abs(tomint[0,:,:,thisbscan])**2)
plot_orig = 10*np.log10(abs(tomint[1,:,:,thisbscan])**2)
zmax=250
zmin = 160
file = 'ZX_256'
output = r'C:\Users\diego\Documents\Github\Result fovea cGAN subsampling'
create_and_save_subplot(plot_cGAN,plot_orig,
                        title1='Resampled with cGAN and TNode',
                        title2='original with TNode',
                        output_path=output
                        ,zmax=zmax,zmin=zmin,
                        file_name=file)
#%%
thisbscan=160
plot_cGAN = 10*np.log10(abs(tomint[0,thisbscan,:,:])**2)
plot_orig = 10*np.log10(abs(tomint[1,thisbscan,:,:])**2)
zmax=250
zmin = 160
file = 'XY_256'
output = r'C:\Users\diego\Documents\Github\Result fovea cGAN subsampling'
create_and_save_subplot(plot_cGAN,plot_orig,
                        title1='Resampled with cGAN and TNode',
                        title2='original with TNode',
                        output_path=output
                        ,zmax=zmax,zmin=zmin,
                        file_name=file)
#%%
thisbscan=160
plot_cGAN = 10*np.log10(abs(tomint[0,:,thisbscan,:])**2)
plot_orig = 10*np.log10(abs(tomint[1,:,thisbscan,:])**2)
zmax=250
zmin = 160
file = 'ZY_256'
output = r'C:\Users\diego\Documents\Github\Result fovea cGAN subsampling'
create_and_save_subplot(plot_cGAN,plot_orig,
                        title1='Resampled with cGAN and TNode',
                        title2='original with TNode',
                        output_path=output
                        ,zmax=zmax,zmin=zmin,
                        file_name=file)
#%%

filename = '\cGANtomintTNode.tiff'
tiff_3Dsave(tomint[0,:,:,:],output+filename)
#%%
filename = '\OriginaltomintTNode.tiff'
tiff_3Dsave(tomint[1,:,:,:],output+filename)





