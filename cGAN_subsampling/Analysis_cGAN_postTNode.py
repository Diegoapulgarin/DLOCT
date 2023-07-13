#%%
import numpy as np
import os
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
def plot_images(array1, array2, zmin, zmax,tittle,save=False):
    fig = sp.make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Heatmap(z=np.flipud(array1), zmin=zmin, zmax=zmax, colorscale='Gray'),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=np.flipud(array2), zmin=zmin, zmax=zmax, colorscale='Gray'),
        row=1, col=2
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)  
    fig.update_layout(height=600, width=800, title_text=tittle)
    if save:
        fig.write_html(tittle +'.html')
    fig.show()
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
#%%
zmin = 160
zmax = 250
tittle = 'Comparision en face plane between original and resampled with cGAN'
plot_images(plot_orig,plot_cGAN,zmin,zmax,tittle,save=False)
