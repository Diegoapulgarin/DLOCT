#%%
import sys
sys.path.append(r'C:\Data\DLOCT\cGAN_subsampling\Functions')
import numpy as np 
from Deep_Utils import create_and_save_subplot,Powerspectrum,MPS_single
import scipy.io as sio
import plotly.graph_objects as go
import plotly.express as px
#%%
pathorig = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)'
real = '_real.bin'
imag = '_imag.bin'
tomreal = np.fromfile(pathorig+'\\'+filename+real,'single')
tomreal = np.reshape(tomreal,(586,896,960,2),order='F')
tomreal = np.sum(tomreal,axis=3)
tomimag = np.fromfile(pathorig+'\\'+filename+imag,'single')
tomimag = np.reshape(tomimag,(586,896,960,2),order='F')
tomimag = np.sum(tomimag,axis=3)
z = 128
enface_original = tomreal[z,:,:]+1j*tomimag[z,:,:]
del tomimag, tomreal
#%% predicted by network
pathcGAN = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected\cGAN_dr'
filename = 'tomDataOverpol'
mat_contents = sio.loadmat(pathcGAN+'\\'+filename+'0')
tomDataOver0 = mat_contents['tomDataOver']
filename = 'tomDataOverpol'
mat_contents = sio.loadmat(pathcGAN+'\\'+filename+'1')
tomDataOver1 = mat_contents['tomDataOver']
tomDataover = np.stack((tomDataOver0,tomDataOver1),axis=3)
tomDataover = np.sum(tomDataover,axis=3)
enface_over = tomDataover[z,:,:]
#%%
del tomDataover, tomDataOver0, tomDataOver1
#%%
plot_orig = 10*np.log10(abs(enface_original)**2)
plot_over = 10*np.log10(abs(enface_over)**2)
#%%
create_and_save_subplot(plot_orig,plot_over,
                        'Original','Reconstructed',
                        pathcGAN,zmin=80,zmax=180)

#%%
meandim = 1
mps_orig = MPS_single(enface_original,meandim=meandim)
mps_reconstructed = MPS_single(enface_over,meandim=meandim)
nx, ny = enface_original.shape
latSamp = 1
if meandim == 0: # I keep the dimension that I do not average
    faxis = np.linspace(-0.5, 0.5 - 1/ny, ny) * (1/latSamp)
else:
    faxis = np.linspace(-0.5, 0.5 - 1/nx, nx) * (1/latSamp)
#%%
fig = go.Figure()
fig.add_trace(go.Line(x= faxis,y=mps_orig,name='Original'))
fig.add_trace(go.Line(x=faxis,y=mps_reconstructed,name='Reconstructed'))
fig.update_layout(legend_title_text = "Tomogram")
fig.update_xaxes(title_text= r'$\text{Frecuency }\text{mm}^{-1}$')
fig.update_yaxes(title_text="Y profile [dB]")
fig.write_html(pathcGAN+'\\'+'mps.html')
fig.show()
#%%

ps_original,_ = Powerspectrum(enface_original)
ps_spectrum,_ = Powerspectrum(enface_over)

#%%
create_and_save_subplot(ps_original,ps_spectrum,
                        'Original','Reconstructed',
                        output_path=pathcGAN,file_name='powespectrum',
                        zmax=1,zmin=0,colorscale='viridis')
