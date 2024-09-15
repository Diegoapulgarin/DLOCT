#%%
import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from Deep_Utils import dbscale

def read_tomogram(file_path, dimensions):
    depth, height, width = dimensions
    with open(file_path, 'rb') as file:
        tomogram = np.fromfile(file, dtype='single')
        tomogram = tomogram.reshape((depth, height, width),order='F')
    return tomogram
#%%
path_tnode = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea\[p.SHARP][s.Eye2a][10-09-2019_13-14-42]\z=(1..400)_x=(1..896)_y=(1..960)-nSpec='
folder = 'Int_8x8x8x0_3x3x3x0_110_0_50_unitary_original'
tomshape = [400,896,960]
tomName = 'TNodeIntFlattenRPE.bin'
tomOriginal = read_tomogram(os.path.join(path_tnode,folder,tomName),tomshape)
print('original loaded')
folder = 'Int_8x8x8x0_3x3x3x0_110_0_50_unitary_recons'
tomReconstruct = read_tomogram(os.path.join(path_tnode,folder,tomName),tomshape)
print('Reconstructed loaded')

#%%
tomSubsampled = tomOriginal[:,:,1::3]
nZbin,nXbin,nYbin = tomshape
tomSubsampledInterp = np.zeros((nZbin,nXbin,nYbin))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    tomSubsampledInterp[z,:,:] = cv2.resize(tomSubsampled[z,:,:],
                dsize=(int(np.shape(tomSubsampled)[2]*3),np.shape(tomSubsampled)[1]),
                interpolation=cv2.INTER_NEAREST)
print('tom linearlly interpolated')

tomSubsampledInterpBi = np.zeros((nZbin, nXbin, nYbin))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    tomSubsampledInterpBi[z, :, :] = cv2.resize(
        tomSubsampled[z, :, :], 
        dsize=(int(np.shape(tomSubsampled)[2]*3), np.shape(tomSubsampled)[1]),
        interpolation=cv2.INTER_CUBIC)
print('tom cubic interpolated')
#%%
z = 45

vmin = 80
vmax= 120
fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(20,10))
axs[0].imshow(dbscale(tomOriginal[z,:,:]),vmin=vmin,vmax=vmax,cmap='gray')
axs[1].imshow(dbscale(tomReconstruct[z,:,:]),vmin=vmin,vmax=vmax,cmap='gray')
axs[2].imshow(dbscale(tomSubsampledInterp[z,:,:]),vmin=vmin,vmax=vmax,cmap='gray')
axs[3].imshow(dbscale(tomSubsampledInterpBi[z,:,:]),vmin=vmin,vmax=vmax,cmap='gray')
fig.show()
x = 800
fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(20,10))
axs[0].imshow(dbscale(tomOriginal[:,x,:]),vmin=vmin,vmax=vmax,cmap='gray')
axs[1].imshow(dbscale(tomReconstruct[:,x,:]),vmin=vmin,vmax=vmax,cmap='gray')
axs[2].imshow(dbscale(tomSubsampledInterp[:,x,:]),vmin=vmin,vmax=vmax,cmap='gray')
axs[3].imshow(dbscale(tomSubsampledInterpBi[:,x,:]),vmin=vmin,vmax=vmax,cmap='gray')
fig.show()
y = 200
fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(20,10))
axs[0].imshow(dbscale(tomOriginal[:,:,y]),vmin=vmin,vmax=vmax,cmap='gray')
axs[1].imshow(dbscale(tomReconstruct[:,:,y]),vmin=vmin,vmax=vmax,cmap='gray')
axs[2].imshow(dbscale(tomSubsampledInterp[:,:,y]),vmin=vmin,vmax=vmax,cmap='gray')
axs[3].imshow(dbscale(tomSubsampledInterpBi[:,:,y]),vmin=vmin,vmax=vmax,cmap='gray')
fig.show()
#%%
xint = 150
xfin = 350
yint = 500
yfin = 750
miniOriginalz = tomOriginal[z,xint:xfin,yint:yfin]
miniReconstructz = tomReconstruct[z,xint:xfin,yint:yfin]
miniInterpz = tomSubsampledInterp[z,xint:xfin,yint:yfin]
miniinterBiz = tomSubsampledInterpBi[z,xint:xfin,yint:yfin]

fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(20,10))
axs[0].imshow(dbscale(miniOriginalz),vmin=vmin,vmax=vmax,cmap='gray')
axs[1].imshow(dbscale(miniReconstructz),vmin=vmin,vmax=vmax,cmap='gray')
axs[2].imshow(dbscale(miniInterpz),vmin=vmin,vmax=vmax,cmap='gray')
axs[3].imshow(dbscale(miniinterBiz),vmin=vmin,vmax=vmax,cmap='gray')
fig.show()

zint = 0
zfin = 100
yint = 180
yfin = 380

miniOriginalx = tomOriginal[zint:zfin,x,yint:yfin]
miniReconstructx = tomReconstruct[zint:zfin,x,yint:yfin]
miniInterpx = tomSubsampledInterp[zint:zfin,x,yint:yfin]
miniinterBix = tomSubsampledInterpBi[zint:zfin,x,yint:yfin]

fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(20,10))
axs[0].imshow(dbscale(miniOriginalx),vmin=vmin,vmax=vmax,cmap='gray')
axs[1].imshow(dbscale(miniReconstructx),vmin=vmin,vmax=vmax,cmap='gray')
axs[2].imshow(dbscale(miniInterpx),vmin=vmin,vmax=vmax,cmap='gray')
axs[3].imshow(dbscale(miniinterBix),vmin=vmin,vmax=vmax,cmap='gray')
fig.show()

zint = 50
zfin = 150
xint = 180
xfin = 380

miniOriginaly = tomOriginal[zint:zfin,xint:xfin,:]
miniReconstructy = tomReconstruct[zint:zfin,xint:xfin,:]
miniInterpy = tomSubsampledInterp[zint:zfin,xint:xfin,:]
miniinterBiy = tomSubsampledInterpBi[zint:zfin,xint:xfin,:]

fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(20,10))
axs[0].imshow(dbscale(miniOriginaly),vmin=vmin,vmax=vmax,cmap='gray')
axs[1].imshow(dbscale(miniOriginaly),vmin=vmin,vmax=vmax,cmap='gray')
axs[2].imshow(dbscale(miniInterpy),vmin=vmin,vmax=vmax,cmap='gray')
axs[3].imshow(dbscale(miniinterBiy),vmin=vmin,vmax=vmax,cmap='gray')
fig.show()