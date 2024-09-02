#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
import os
from Deep_Utils import create_and_save_subplot, tiff_3Dsave,save_image, calculate_ssim,calculate_mse,calculate_psnr,relative_error,histogram_difference, dbscale
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots
from matplotlib.colors import Normalize as cmnorm
from matplotlib.cm import ScalarMappable
#%%
path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea\[p.SHARP][s.Eye2a][10-09-2019_13-14-42]\z=(1..400)_x=(1..896)_y=(1..960)-nSpec='
original_path = 'Int_8x8x8x0_3x3x3x0_110_0_50_unitary_original'
reconstructed_path = 'Int_8x8x8x0_3x3x3x0_110_0_50_unitary_recons'
z = 400
x = 896
y = 960
file_name = 'TNodeIntFlattenRPE.bin'
tomOriginal = np.fromfile(os.path.join(path,original_path,file_name),'single')
tomOriginal = np.reshape(tomOriginal,(z,x,y),order='F')
tomReconstructed = np.fromfile(os.path.join(path,reconstructed_path,file_name),'single')
tomReconstructed = np.reshape(tomReconstructed,(z,x,y),order='F')

path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
fnameTom = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]_TomoriginalFlat_z=400_x=896_y=960_pol=2' # fovea
tomShape = [(400,896,960,2)]# porcine cornea
fname = os.path.join(path, fnameTom)
# Names of all real and imag .bin files
fnameTomReal = [fname + '_real.bin' ]
fnameTomImag = [fname + '_imag.bin' ]
tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using

tomImag = np.fromfile(fnameTomImag[0],'single')
tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using

tomDatas = np.stack((tomReal,tomImag), axis=4)
del tomImag, tomReal
tomDatas = tomDatas[:,:,:,:,0] + 1j* tomDatas[:,:,:,:,1]
tomDatas = tomDatas.sum(axis=3)

fnameTom = '[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]_TomFlat_z=400_x=896_y=960_pol=2'
fname = os.path.join(path, fnameTom)
fnameTomReal = [fname + '_real.bin' ]
fnameTomImag = [fname + '_imag.bin' ]
tomReal = np.fromfile(fnameTomReal[0],'single') # quit single for porcine cornea and put single for s_eye
tomReal = tomReal.reshape(tomShape[0], order='F')  # reshape using

tomImag = np.fromfile(fnameTomImag[0],'single')
tomImag = tomImag.reshape(tomShape[0], order='F')  # reshape using

tomDatar = np.stack((tomReal,tomImag), axis=4)
tomDatar = tomDatar[:,:,:,:,0] + 1j* tomDatar[:,:,:,:,1]
tomDatar = tomDatar.sum(axis=3)
del tomImag, tomReal
#%%
nZBin = z
metrics_tnode = []
for i in tqdm(range(nZBin)):
    enfaceOriginal = 10*np.log10(abs(tomOriginal[i,:,:])**2)
    enfaceOriginal = enfaceOriginal/np.max(enfaceOriginal)
    enfaceReconstructed=10*np.log10(abs(tomReconstructed[i,:,:])**2)
    enfaceReconstructed = enfaceReconstructed/np.max(enfaceReconstructed)
    ssim_result =calculate_ssim(enfaceOriginal,enfaceReconstructed)
    mse = calculate_mse(enfaceOriginal,enfaceReconstructed)
    psnr = calculate_psnr(enfaceOriginal,enfaceReconstructed)
    hdiff = histogram_difference(np.float32(enfaceOriginal),
                                 np.float32(enfaceReconstructed),
                                 method='cosine-similarity')
    metrics_results = np.array((ssim_result,mse,psnr,hdiff))
    metrics_tnode.append(metrics_results)
metrics_tnode = np.array(metrics_tnode)


metrics_pretnode = []
for i in tqdm(range(nZBin)):
    enfaceOriginal = 10*np.log10(abs(tomDatas[i,:,:])**2)
    enfaceOriginal = enfaceOriginal/np.max(enfaceOriginal)
    enfaceReconstructed=10*np.log10(abs(tomDatar[i,:,:])**2)
    enfaceReconstructed = enfaceReconstructed/np.max(enfaceReconstructed)
    ssim_result =calculate_ssim(enfaceOriginal,enfaceReconstructed)
    mse = calculate_mse(enfaceOriginal,enfaceReconstructed)
    psnr = calculate_psnr(enfaceOriginal,enfaceReconstructed)
    hdiff = histogram_difference(np.float32(enfaceOriginal),
                                 np.float32(enfaceReconstructed),
                                 method='cosine-similarity')
    metrics_results = np.array((ssim_result,mse,psnr,hdiff))
    metrics_pretnode.append(metrics_results)
metrics_pretnode = np.array(metrics_pretnode)
#%%

fig = make_subplots(rows=1, cols=1)
pretnode_mean = np.mean(metrics_pretnode[:,0])
tnode_mean = np.mean(metrics_tnode[:,0])
pretnode_std = np.std(metrics_pretnode[:,0])
tnode_std = np.std(metrics_tnode[:,0])

yaxis_config = {
    'title': 'SSIM',
    'range': [0.5, 1],
    'titlefont': {'family': 'Times New Roman', 'size': 25, 'color': 'black'},
    'tickfont': {'family': 'Times New Roman', 'size': 20, 'color': 'black'}
}

fig.add_trace(
    go.Scatter(y=metrics_tnode[:,0], 
               mode='lines', 
               name='SSIM Post TNode',
               line=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(go.Scatter(y=[tnode_mean]*len(metrics_pretnode[:,0]), name=f'SSIM Post TNode mean',
                            mode='lines', line=dict(color='blue', dash='dash')))

fig.add_trace(
    go.Scatter(y=metrics_pretnode[:,0], 
               mode='lines', 
               name='SSIM Pre TNode',
               line=dict(color='red')),
               
    row=1, col=1
)


fig.add_trace(go.Scatter(y=[pretnode_mean]*len(metrics_pretnode[:,0]), 
                         name=f'SSIM Pre TNode mean',
                         mode='lines', line=dict(color='red', dash='dash')))

fig.add_annotation(x=len(metrics_pretnode[:, 0]) / 2, y=tnode_mean - 0.01,
                   text=f"Mean: {tnode_mean:.2f} \u00B1 {tnode_std:.2f}",
                   showarrow=False,
                   font=dict(family="Times New Roman", size=14, color="blue"))

fig.add_annotation(x=len(metrics_pretnode[:, 0]) / 2, y=pretnode_mean- 0.01,
                   text=f"Mean: {pretnode_mean:.2f} \u00B1 {pretnode_std:.2f}",
                   showarrow=False,
                   font=dict(family="Times New Roman", size=14, color="red"))

fig.update_layout(
    title_text="Comparison of SSIM before and after TNode",
    title={
        'font': {'family': 'Times New Roman', 'size': 30, 'color': 'black'}
    },
    height=600,  # Ajusta la altura si es necesario
    showlegend=True,
    yaxis=yaxis_config,
    legend={
        'font': {'family': 'Times New Roman', 'size': 18, 'color': 'black'}
    }
)

fig.update_xaxes(
    title_text="en-face slices",
    titlefont={'family': 'Times New Roman', 'size': 25, 'color': 'black'},
    tickfont={'family': 'Times New Roman', 'size': 20, 'color': 'black'},
    row=1, col=1
)

# fig.show()
save_path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea\cortes\corte6'
fig.write_html(os.path.join(save_path,'ssim pre and post tnode compare.html'))
#%%
min_ssim = 144
max_ssim = 180
pretnode_min_o =dbscale(tomDatas[min_ssim,:,:])
pretnode_max_o =dbscale(tomDatas[max_ssim,:,:])
pretnode_min_r =dbscale(tomDatar[min_ssim,:,:])
pretnode_max_r =dbscale(tomDatar[max_ssim,:,:])

tnode_min_o =dbscale(tomOriginal[min_ssim,:,:])
tnode_max_o =dbscale(tomOriginal[max_ssim,:,:])
tnode_min_r =dbscale(tomReconstructed[min_ssim,:,:])
tnode_max_r =dbscale(tomReconstructed[max_ssim,:,:])

#%%
vmin = 75
vmax = 120
savefig = False
path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
folder = 'cortes'
subfolder = f'corte{6}'
fig, axs = plt.subplots(2, 2, figsize=(40, 40))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0,0].imshow(pretnode_min_o, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0,0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[0,1].imshow(pretnode_min_r, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0,1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[1,0].imshow(tnode_min_o, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1,0].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[1,1].imshow(tnode_min_r, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1,1].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0.01)

figname = f'comparision min ssim'

# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()
#%%

fig, axs = plt.subplots(2, 2, figsize=(40, 40))
cmap= plt.cm.gray
norm = cmnorm(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap, norm=norm) 
axs[0,0].imshow(pretnode_max_o, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0,0].axis('off')
# axs[0].set_title(f'Original sharpness= {np.int32(bsh)}')

axs[0,1].imshow(pretnode_max_r, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[0,1].axis('off') 
# axs[1].set_title(f'cGAN reconstructed sharpness= {np.int32(ash)}')

axs[1,0].imshow(tnode_max_o, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1,0].axis('off')
# axs[2].set_title(f'Subsampled interpolated sharpeness= {np.int32(dsh)}')

axs[1,1].imshow(tnode_max_r, cmap='gray',vmin=vmin,vmax=vmax,aspect='auto')
axs[1,1].axis('off')
# axs[3].set_title(f'Subsampled sharpeness= {np.int32(csh)}')
plt.subplots_adjust(wspace=0.01, hspace=0.01)

figname = f'comparision max ssim'

# cbar = fig.colorbar(sm, aspect=10, orientation='vertical', ax=axs[3], label='dB')  
if savefig:
    plt.savefig(os.path.join(path,folder,subfolder,figname), dpi=300)
    print('fig saved')
plt.show()

#%%
# ssim_result,mse,psnr,hdiff
print(f'mean ssim pretnode:{np.mean(metrics_pretnode[:,0])}')
print(f'mean mse pretnode:{np.mean(metrics_pretnode[:,1])}')
print(f'mean psnr pretnode:{np.mean(metrics_pretnode[:,2])}')
print(f'mean hdiff pretnode:{np.mean(metrics_pretnode[:,3])}')

print(f'std ssim pretnode:{np.std(metrics_pretnode[:,0])}')
print(f'std mse pretnode:{np.std(metrics_pretnode[:,1])}')
print(f'std psnr pretnode:{np.std(metrics_pretnode[:,2])}')
print(f'std hdiff pretnode:{np.std(metrics_pretnode[:,3])}')

print(f'mean ssim tnode:{np.mean(metrics_tnode[:,0])}')
print(f'mean mse tnode:{np.mean(metrics_tnode[:,1])}')
print(f'mean psnr tnode:{np.mean(metrics_tnode[:,2])}')
print(f'mean hdiff tnode:{np.mean(metrics_tnode[:,3])}')

print(f'std ssim tnode:{np.std(metrics_tnode[:,0])}')
print(f'std mse tnode:{np.std(metrics_tnode[:,1])}')
print(f'std psnr tnode:{np.std(metrics_tnode[:,2])}')
print(f'std hdiff tnode:{np.std(metrics_tnode[:,3])}')

#%%

ssim_synthetic = np.load(os.path.join(path,'ssim synthetic fovea flat.npy'))
ssim_experimental = np.load(os.path.join(path,'ssim experimental fovea flat.npy'))
#%%
fig = make_subplots(rows=1, cols=1)
pretnode_mean = np.mean(metrics_pretnode[:, 0])
pretnode_std = np.std(metrics_pretnode[:, 0])
exper_mean = np.mean(ssim_experimental)
exper_std = np.std(ssim_experimental)
syn_mean = np.mean(ssim_synthetic)
syn_std = np.std(ssim_synthetic)

yaxis_config = {
    'title': 'SSIM',
    'range': [0.4, 0.9],
    'titlefont': {'family': 'Times New Roman', 'size': 25, 'color': 'black'},
    'tickfont': {'family': 'Times New Roman', 'size': 20, 'color': 'black'}
}

fig.add_trace(
    go.Scatter(y=ssim_experimental, 
               mode='lines', 
               name='SSIM - experimental model',
               line=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(go.Scatter(y=[exper_mean]*len(metrics_pretnode[:,0]), 
                         name=f'Average SSIM - experimental model',
                            mode='lines', 
                            line=dict(color='blue', dash='dash')))

fig.add_trace(
    go.Scatter(y=metrics_pretnode[:,0], 
               mode='lines', 
               name='SSIM - syn. + exper. model',
               line=dict(color='red')),
               
    row=1, col=1
)


fig.add_trace(go.Scatter(y=[pretnode_mean]*len(metrics_pretnode[:,0]), 
                         name=f'Average SSIM - syn. + exper. model',
                         mode='lines', line=dict(color='red', dash='dash')))


fig.add_trace(
    go.Scatter(y=ssim_synthetic, 
               mode='lines', 
               name='SSIM - synthetic model',
               line=dict(color='green')),
               
    row=1, col=1
)


fig.add_trace(go.Scatter(y=[syn_mean]*len(metrics_pretnode[:,0]), 
                         name=f'Average SSIM - synthetic model',
                         mode='lines', line=dict(color='green', dash='dash')))


# Añadir anotaciones para los valores promedios
fig.add_annotation(x=len(metrics_pretnode[:, 0]) / 2, y=exper_mean - 0.01,
                   text=f"Mean: {exper_mean:.2f} \u00B1 {exper_std:.2f}",
                   showarrow=False,
                   font=dict(family="Times New Roman", size=14, color="blue"))

fig.add_annotation(x=len(metrics_pretnode[:, 0]) / 2, y=pretnode_mean- 0.01,
                   text=f"Mean: {pretnode_mean:.2f} \u00B1 {pretnode_std:.2f}",
                   showarrow=False,
                   font=dict(family="Times New Roman", size=14, color="red"))

fig.add_annotation(x=len(metrics_pretnode[:, 0]) / 2, y=syn_mean- 0.01,
                   text=f"Mean: {syn_mean:.2f} \u00B1 {syn_std:.2f}",
                   showarrow=False,
                   font=dict(family="Times New Roman", size=14, color="green"))

fig.update_layout(
    title_text="Comparison of SSIM between different experiments",
    title={
        'font': {'family': 'Times New Roman', 'size': 30, 'color': 'black'}
    },
    height=600,  # Ajusta la altura si es necesario
    showlegend=True,
    yaxis=yaxis_config,
    legend={
        'font': {'family': 'Times New Roman', 'size': 18, 'color': 'black'}
    }
)

fig.update_xaxes(
    title_text="en-face slices",
    titlefont={'family': 'Times New Roman', 'size': 25, 'color': 'black'},
    tickfont={'family': 'Times New Roman', 'size': 20, 'color': 'black'},
    row=1, col=1
)

# fig.show()
save_path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea\cortes\corte6'
fig.write_html(os.path.join(save_path,'ssim exper and synthetics.html'))