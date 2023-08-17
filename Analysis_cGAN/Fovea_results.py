'''
Z depth = 3.9 um for each pixel
X depth = 14 um for each pixel
Y depth = 28 um for each pixel
'''
#%%
import numpy as np
# from Deep_Utils import tiff_3Dsave,create_and_save_subplot
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import plotly.graph_objs as go

#%%
CentralWavelength = 870e-9
bandwith = 50e-9
pixel = (2*np.log(2)/np.pi)*(CentralWavelength**2/bandwith)

#%%
path = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected'
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomintOriginal_z=(295..880)_x=(65..960)_y=(1..960).bin'
tom = np.fromfile(path+'\\'+filename,'single')
nZbin = 586
nXbin = 896
nYbin = 960
tomOriginal = np.reshape(tom,(nZbin,nXbin,nYbin,2),order='F')
tomOriginal = np.sum(tomOriginal,axis=3)
#%%
path = r'D:\DLOCT\TomogramsDataAcquisition\Fovea\Motion_corrected\Final\Int_8x8x8x0_3x3x3x0_150_0_50_unitary'
filename = 'TNodeIntFlattenRPE.bin'
tom = np.fromfile(path+'\\'+filename,'single')
xSlice = 587
ySlice = 587
zSlice = 586
tomTNode = np.reshape(tom,(zSlice,xSlice,ySlice),order='F')
del tom
#%%
n=190
fig = px.imshow(10*np.log10(tomOriginal[n,155:741,187:773]),color_continuous_scale='gray',zmin=70,zmax=160)
fig.show()
# fig.write_html('original.html')
#%%
n = 190
fig = px.imshow(10*np.log10(tomTNode[n,:,:]),color_continuous_scale='gray',zmin=70,zmax=160)
fig.show()
# fig.write_html('Tnode.html')
#%%
n = 190
out = r'C:\Data\partial results'
plot1 = 10*np.log10(tomOriginal[:,155:742,187+n])
plot2 = 10*np.log10(tomTNode[:,:,n])
#%%
# create_and_save_subplot(plot2,plot1,
#                         'Tomogram with TNode',
#                         'Original Tomogram',
#                         output_path=out,
#                         zmin=70,zmax=160,
#                         file_name='Fovea compare with TNode')

#%%

image1 = np.flipud(plot2)
image2 = np.flipud(plot1)
title1 = 'Tomogram with TNode'
title2 =  'Original Tomogram'
title_size = 32
title_color = 'black'
colorscale = 'gray'
file_name='ZX Fovea compare with TNode'
output_path = r'C:\Data\partial_results'
#%%
fig = make_subplots(rows=1, cols=2)
zmin=70
zmax=160
fig.add_annotation(dict(text=title1, xref="x1", yref="paper",
                            x=int(image1.shape[0]/2),
                            y=1.07,
                            showarrow=False,
                            font=dict(size=title_size,color=title_color)))
fig.add_annotation(dict(text=title2, xref="x2", yref="paper",
                        x=int(image1.shape[0]/2),
                        y=1.07,
                        showarrow=False,
                        font=dict(size=title_size,color=title_color)))

fig.add_trace(go.Heatmap(z=image1, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=True,
                            colorbar=dict(y=0.5, len=(image1.shape[0]/(1*image1.shape[0])), yanchor="middle")), row=1, col=1)
fig.add_trace(go.Heatmap(z=image2, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=False), row=1, col=2)

fig.update_xaxes(scaleanchor = 'x',showticklabels=False, visible=False,showgrid=False)
fig.update_yaxes(scaleanchor = 'x',showticklabels=False, visible=False,showgrid=False)

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=20),
        coloraxis_colorbar_x=0.83,
        font_family="Serif",
        font_size=24)
html_name = '\\'+file_name+'.html'
# svg_name = file_name+'.svg'
# fig.update_layout(title_text="Subplot of Two Images")
# pio.write_image(fig, os.path.join(output_path, svg_name), format="svg", scale=dpi/72)
fig.write_html(output_path + html_name)
fig.show()
