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
path_tnode = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea\[p.SHARP][s.Eye2a][10-09-2019_13-14-42] old _ no_flat'
folder = 'z=(1..586)_x=(1..896)_y=(1..960)-nSpec='
subfolder = 'Int_8x8x8x0_3x3x3x0_150_0_50_unitary'
tomshape = [586,896,960]
tomName = 'TNodeIntFlattenRPE.bin'
tomOriginal = np.sqrt(read_tomogram(os.path.join(path_tnode,folder,subfolder,tomName),tomshape))
print('original loaded')
subfolder = 'Int_8x8x8x0_3x3x3x0_170_0_50_unitary'
tomReconstruct = np.sqrt(read_tomogram(os.path.join(path_tnode,folder,subfolder,tomName),tomshape))
print('Reconstructed loaded')
folder = 'z=(1..586)_x=(1..896)_y=(1..480)-nSpec='
subfolder = 'Int_8x8x8x0_3x3x3x0_250_0_50_unitary'
tomshapesub = [586,896,480]
tomSubsampled = np.sqrt(read_tomogram(os.path.join(path_tnode,folder,subfolder,tomName),tomshapesub))
nZbin,nXbin,nYbin = tomshapesub
print('Subsampled loaded')
#%%
tomSubsampledInterp = np.zeros((nZbin,nXbin,nYbin*2))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    tomSubsampledInterp[z,:,:] = cv2.resize(tomSubsampled[z,:,:],
                dsize=(int(np.shape(tomSubsampled)[2]*2),np.shape(tomSubsampled)[1]),
                interpolation=cv2.INTER_NEAREST)
print('tom linearlly interpolated')

tomSubsampledInterpBi = np.zeros((nZbin, nXbin, nYbin*2))
for z in tqdm(range(np.shape(tomSubsampled)[0])):
    tomSubsampledInterpBi[z, :, :] = cv2.resize(
        tomSubsampled[z, :, :], 
        dsize=(int(np.shape(tomSubsampled)[2]*2), np.shape(tomSubsampled)[1]),
        interpolation=cv2.INTER_CUBIC)
print('tom cubic interpolated')
#%%
z = 170

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
y = 0
fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(20,10))
axs[0].imshow(dbscale(tomOriginal[:,:,y]),vmin=vmin,vmax=vmax,cmap='gray')
axs[1].imshow(dbscale(tomReconstruct[:,:,y]),vmin=vmin,vmax=vmax,cmap='gray')
axs[2].imshow(dbscale(tomSubsampledInterp[:,:,y]),vmin=vmin,vmax=vmax,cmap='gray')
axs[3].imshow(dbscale(tomSubsampledInterpBi[:,:,y]),vmin=vmin,vmax=vmax,cmap='gray')
fig.show()
#%%
xint = 400
xfin = 600
yint = 180
yfin = 380
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

zint = 300
zfin = 450
yint = 350
yfin = 600

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

zint = 100
zfin = 300
xint = 600
xfin = 800

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

#%%
savefig = True
pathSave = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea\cortes\corte6'
path = os.path.join(pathSave,'tnode')
def saveIndividuals(image,path,figname,cmap='gray',vmin=80,vmax=120):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap,vmin=vmin,vmax=vmax)  # 
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(path,figname),bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
    plt.close()

if savefig:
        image = dbscale(tomOriginal[z,:,:])
        figname = f'Tnode_original_Z={z}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomReconstruct[z,:,:])
        figname = f'Tnode_Recosntruct_Z={z}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomSubsampledInterp[z,:,:])
        figname = f'Tnode_Subsampledinterp_Z={z}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomSubsampledInterpBi[z,:,:])
        figname = f'Tnode_tomSubsampledInterpBi_Z={z}.svg'
        saveIndividuals(image,path,figname)        

        image = dbscale(miniOriginalz)
        figname = f'Tnode_minioriginal_Z={z}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniReconstructz)
        figname = f'Tnode_miniRecosntruct_Z={z}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniInterpz)
        figname = f'Tnode_miniSubsampledinterp_Z={z}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniinterBiz)
        figname = f'Tnode_minitomSubsampledInterpBi_Z={z}.svg'
        saveIndividuals(image,path,figname)

###########################################################

        image = dbscale(tomOriginal[:,x,:])
        figname = f'Tnode_original_X={x}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomReconstruct[:,x,:])
        figname = f'Tnode_Recosntruct_X={x}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomSubsampledInterp[:,x,:])
        figname = f'Tnode_Subsampledinterp_X={x}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomSubsampledInterpBi[:,x,:])
        figname = f'Tnode_tomSubsampledInterpBi_X={x}.svg'
        saveIndividuals(image,path,figname)        

        image = dbscale(miniOriginalx)
        figname = f'Tnode_minioriginal_X={x}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniReconstructx)
        figname = f'Tnode_miniRecosntruct_X={x}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniInterpx)
        figname = f'Tnode_miniSubsampledinterp_Z={x}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniinterBix)
        figname = f'Tnode_minitomSubsampledInterpBi_Z={x}.svg'
        saveIndividuals(image,path,figname)
########################################################
 
        image = dbscale(tomOriginal[:,:,y])
        figname = f'Tnode_original_y={y}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomReconstruct[:,:,y])
        figname = f'Tnode_Recosntruct_y={y}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomSubsampledInterp[:,:,y])
        figname = f'Tnode_Subsampledinterp_y={y}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(tomSubsampledInterpBi[:,:,y])
        figname = f'Tnode_tomSubsampledInterpBi_y={y}.svg'
        saveIndividuals(image,path,figname)        

        image = dbscale(miniOriginaly)
        figname = f'Tnode_minioriginal_y={y}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniReconstructy)
        figname = f'Tnode_miniRecosntruct_y={y}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniInterpy)
        figname = f'Tnode_miniSubsampledinterp_y={y}.svg'
        saveIndividuals(image,path,figname)

        image = dbscale(miniinterBiy)
        figname = f'Tnode_minitomSubsampledInterpBi_y={y}.svg'
        saveIndividuals(image,path,figname)