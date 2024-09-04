#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from Deep_Utils import dbscale

from scipy.optimize import curve_fit

def gauss(x, A, sigma, offtset):
    return A * np.exp(-(x)**2/(2*sigma**2)) + offtset


def gaussfit(binscenters, counts, p0):
    
    popt, pcov = curve_fit(gauss, binscenters, counts, p0)
    residuals = counts - gauss(binscenters, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((counts - np.mean(counts))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    hist_fit = gauss(binscenters, *popt)
    return hist_fit, r_squared, popt
#%%

path = r'E:\DLOCT\ultimo experimento subsampling paper\OpticNerve'
fnameTomImag = 'Tom_Imag_z=800_x=1024_y=992_pol2.bin' # fovea
fnameTomReal = 'Tom_Real_z=800_x=1024_y=992_pol2.bin' # fovea
tomShape = [(800,1024,992,2)]# porcine cornea
fnameReal = os.path.join(path, fnameTomReal)
fnameImag = os.path.join(path, fnameTomImag)
tomReal = np.fromfile(fnameReal,'single') 
tomReal = tomReal.reshape(tomShape[0], order='F')
tomReal = tomReal[250:750,:,:,0]

tomImag = np.fromfile(fnameImag,'single')
tomImag = tomImag.reshape(tomShape[0], order='F')
tomImag = tomImag[250:750,:,:,0]
tomOriginal = np.stack((tomReal,tomImag),axis=3)
del tomReal,tomImag
print('Tom original loaded')
#%%
optimumFilter = True
if optimumFilter:
    print('loading tom with OF')
    fileName = 'TomOverOptimumFilter.npy'
    tomReconstructed = np.load(os.path.join(path,fileName))
    tomReconstructed = np.stack((tomReconstructed.real,tomReconstructed.imag),axis=3)
else:
    print('loading tom')
    fnameTomImag = 'TomOver_Imag_z=500_x=1024_y=992_pol2.bin' # fovea
    fnameTomReal = 'TomOver_Real_z=500_x=1024_y=992_pol2.bin' # fovea
    tomShape = [(500,1024,992,2)]# porcine cornea
    fnameReal = os.path.join(path, fnameTomReal)
    fnameImag = os.path.join(path, fnameTomImag)
    tomReal = np.fromfile(fnameReal,'float') 
    tomReal = tomReal.reshape(tomShape[0])
    tomReal = tomReal[:,:,:,0]

    tomImag = np.fromfile(fnameImag,'float')
    tomImag = tomImag.reshape(tomShape[0])
    tomImag = tomImag[:,:,:,0]
    tomReconstructed = np.stack((tomReal,tomImag),axis=3)
    del tomReal,tomImag

print('Tom Reconstructed loaded')

#%%
z = 250
x = 519
pathSave = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
folder = 'cortes'
subfolder = f'corte{6}'
savefig = True
vmin= 65
vmax = 120
fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
axs[0,0].imshow(dbscale(tomOriginal[z,0:600,0:512,:]),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect='auto')
axs[0,0].axis('off')
axs[0,1].imshow(dbscale(tomReconstructed[z,0:600,0:512,:]),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect = 'auto')
axs[0,1].axis('off')
axs[0,2].imshow(dbscale(tomReconstructed[z,0:600,:,:]),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect = 'auto')
axs[0,2].axis('off')

axs[1,0].imshow(dbscale(tomOriginal[:,x,0:512,:]),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect = 'auto')
axs[1,0].axis('off')
axs[1,1].imshow(dbscale(tomReconstructed[:,x,0:512,:]),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect='auto')
axs[1,1].axis('off')
axs[1,2].imshow(dbscale(tomReconstructed[:,x,:,:]),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect = 'auto')
axs[1,2].axis('off')
fig.show()

xint = 0
xfin = 600 
if savefig:

        image = dbscale(tomOriginal[z,xint:xfin,0:512,:])
        figname = f'enfaceOriginal_Z={z}.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()

        image = dbscale(tomOriginal[:,x,0:512,:])
        figname = f'transversalOriginal_X={x}.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()
    #####################################################################################

        image = dbscale(tomReconstructed[z,xint:xfin,0:512,:])
        figname = f'enfaceReconstructed_Z={z}.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()

        image = dbscale(tomReconstructed[:,x,0:512,:])
        figname = f'transversalReconstructed_X={x}.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()
    #######################################################################################

        image = dbscale(tomReconstructed[z,xint:xfin,:,:])
        figname = f'enfaceReconstructed_Z={z}T.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()

        image = dbscale(tomReconstructed[:,x,:,:])
        figname = f'transversalReconstructed_X={x}T.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()
#%%

miniOriginal = tomOriginal[z,220:420,252:452,:]
miniReconstructed = tomReconstructed[z,220:420,252:452,:]
miniReconstructedt = tomReconstructed[z,220:420,252:452,:]

xminiOriginal = tomOriginal[200:400,x,112:312,:]
xminiReconstructed = tomReconstructed[200:400,x,112:312,:]
xminiReconstructedt = tomReconstructed[200:400,x,112:312,:]

fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
axs[0,0].imshow(dbscale(miniOriginal),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect='auto')
axs[0,0].axis('off')
axs[0,1].imshow(dbscale(miniReconstructed),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect = 'auto')
axs[0,1].axis('off')
axs[0,2].imshow(dbscale(miniReconstructedt),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect = 'auto')
axs[0,2].axis('off')

axs[1,0].imshow(dbscale(xminiOriginal),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect = 'auto')
axs[1,0].axis('off')
axs[1,1].imshow(dbscale(xminiReconstructed),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect='auto')
axs[1,1].axis('off')
axs[1,2].imshow(dbscale(xminiReconstructedt),
                vmin=vmin,
                vmax=vmax,
                cmap='gray',
                aspect = 'auto')
axs[1,2].axis('off')
fig.show()

if savefig:

        image = dbscale(miniOriginal)
        figname = f'minienfaceOriginal_Z={z}.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()

        image = dbscale(xminiOriginal)
        figname = f'minitransversalOriginal_X={x}.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()
    #####################################################################################

        image = dbscale(miniReconstructed)
        figname = f'minienfaceReconstructed_Z={z}.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()

        image = dbscale(xminiReconstructed)
        figname = f'minitransversalReconstructed_X={x}.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()
    #######################################################################################

        image = dbscale(miniReconstructedt)
        figname = f'minienfaceReconstructed_Z={z}T.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()

        image = dbscale(xminiReconstructedt)
        figname = f'minitransversalReconstructed_X={x}T.svg'
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray',vmin=vmin,vmax=vmax)  # 
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(pathSave,folder,subfolder,figname), 
                    bbox_inches='tight', pad_inches=0, dpi=150,format='svg')
        plt.close()