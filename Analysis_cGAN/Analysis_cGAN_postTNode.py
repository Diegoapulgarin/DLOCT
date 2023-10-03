#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\cGAN_subsampling\Functions')
import numpy as np
import os
from Deep_Utils import create_and_save_subplot, tiff_3Dsave,save_image, calculate_ssim,calculate_mse,calculate_psnr,relative_error,histogram_difference
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
#%%
z=170
enfaceOriginal = 10*np.log10(abs(tomint[0,z,:,:]))
enfaceOriginal = enfaceOriginal/np.max(enfaceOriginal)
enfaceReconstructed=10*np.log10(abs(tomint[1,z,:,:]))
enfaceReconstructed = enfaceReconstructed/np.max(enfaceReconstructed)
ssim_result =calculate_ssim(enfaceOriginal,enfaceReconstructed)
mse = calculate_mse(enfaceOriginal,enfaceReconstructed)
psnr = calculate_psnr(enfaceOriginal,enfaceReconstructed)
rerror = relative_error(enfaceOriginal,enfaceReconstructed)
hdiff = histogram_difference(enfaceOriginal,enfaceReconstructed)
print("SSIM:", ssim_result)
print("MSE:", mse)
print("PSNR:", psnr)
print("relative error:", rerror)
print("histogram difference:", hdiff)
#%%
metrics_complete = []
for i in range(nZBin):
    enfaceOriginal = 10*np.log10(abs(tomint[0,i,:,:]))
    enfaceOriginal = enfaceOriginal/np.max(enfaceOriginal)
    enfaceReconstructed=10*np.log10(abs(tomint[1,i,:,:]))
    enfaceReconstructed = enfaceReconstructed/np.max(enfaceReconstructed)
    ssim_result =calculate_ssim(enfaceOriginal,enfaceReconstructed)
    mse = calculate_mse(enfaceOriginal,enfaceReconstructed)
    psnr = calculate_psnr(enfaceOriginal,enfaceReconstructed)
    rerror = relative_error(enfaceOriginal,enfaceReconstructed)
    hdiff = histogram_difference(enfaceOriginal,enfaceReconstructed)
    metrics_results = np.array((ssim_result,mse,psnr,rerror,hdiff))
    metrics_complete.append(metrics_results)
    print('Results ok z: ',i)
#%%
path_save = r'C:\Users\USER\Documents'
metrics = np.array(metrics_complete)
plt.plot(metrics[:,0])
np.save(path_save+'\\metrics_TNode',metrics)
#%%



# Datos
ssim_values = metrics[:, 0]
x = np.arange(1, 587)  # Números de plano enface
mean_ssim = np.mean(ssim_values)
std_ssim = np.std(ssim_values)


# Crear figura
fig = go.Figure()

# Agregar datos de SSIM
fig.add_trace(go.Scatter(x=x, y=ssim_values, mode='lines', name='SSIM'))

# Agregar línea de promedio
fig.add_shape(
    type="line",
    x0=1,
    y0=mean_ssim,
    x1=586,
    y1=mean_ssim,
    line=dict(color="Red", width=2, dash="dash"),
    name="Mean"
)

# Agregar banda de desviación estándar
fig.add_shape(
    type="rect",
    x0=1,
    y0=mean_ssim - std_ssim,
    x1=586,
    y1=mean_ssim + std_ssim,
    line=dict(color="Green", width=1),
    fillcolor="Green",
    opacity=0.2,
    name="standard deviation"
)

# Configuraciones adicionales
fig.update_layout(
    title='SSIM ',
    xaxis=dict(title='Enface planes',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    yaxis=dict(title='SSIM Value',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    legend_title_text='Metric',
    font=dict(family="Times New Roman", size=25),
    font_color='black',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Mostrar gráfico
fig.show()
fig.write_html(path_save+'\\ssim.html')
#%%

# Datos
mse_values = metrics[:, 1]
x = np.arange(1, 587)  # Números de plano enface
mean_mse = np.mean(mse_values)
std_mse = np.std(mse_values)


# Crear figura
fig = go.Figure()

# Agregar datos de SSIM
fig.add_trace(go.Scatter(x=x, y=mse_values, mode='lines', name='MSE'))

# Agregar línea de promedio
fig.add_shape(
    type="line",
    x0=1,
    y0=mean_mse,
    x1=586,
    y1=mean_mse,
    line=dict(color="Red", width=2, dash="dash"),
    name="Mean"
)

# Agregar banda de desviación estándar
fig.add_shape(
    type="rect",
    x0=1,
    y0=mean_mse - std_mse,
    x1=586,
    y1=mean_mse + std_mse,
    line=dict(color="Green", width=1),
    fillcolor="Green",
    opacity=0.2,
    name="standard deviation"
)

# Configuraciones adicionales
fig.update_layout(
    title='MSE ',
    xaxis=dict(title='Enface planes',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    yaxis=dict(title='MSE Value',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    legend_title_text='Metric',
    font=dict(family="Times New Roman", size=25),
    font_color='black',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Mostrar gráfico
fig.show()
fig.write_html(path_save+'\\mse.html')

#%%

# Datos
psnr_values = metrics[:, 2]
x = np.arange(1, 587)  # Números de plano enface
mean_psnr = np.mean(psnr_values)
std_psnr = np.std(psnr_values)


# Crear figura
fig = go.Figure()

# Agregar datos de SSIM
fig.add_trace(go.Scatter(x=x, y=psnr_values, mode='lines', name='PSNR'))

# Agregar línea de promedio
fig.add_shape(
    type="line",
    x0=1,
    y0=mean_psnr,
    x1=586,
    y1=mean_psnr,
    line=dict(color="Red", width=2, dash="dash"),
    name="Mean"
)

# Agregar banda de desviación estándar
fig.add_shape(
    type="rect",
    x0=1,
    y0=mean_psnr - std_psnr,
    x1=586,
    y1=mean_psnr + std_psnr,
    line=dict(color="Green", width=1),
    fillcolor="Green",
    opacity=0.2,
    name="standard deviation"
)

# Configuraciones adicionales
fig.update_layout(
    title='PSNR',
    xaxis=dict(title='Enface planes',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    yaxis=dict(title='PNSR Value [dB]',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    legend_title_text='Metric',
    font=dict(family="Times New Roman", size=25),
    font_color='black',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Mostrar gráfico
fig.show()
fig.write_html(path_save+'\\psnr.html')
#%%

# Datos
rerror_values = metrics[:, 3]
x = np.arange(1, 587)  # Números de plano enface
mean_rerror = np.mean(rerror_values)
std_rerror = np.std(rerror_values)


# Crear figura
fig = go.Figure()

# Agregar datos de SSIM
fig.add_trace(go.Scatter(x=x, y=rerror_values, mode='lines', name='rerror'))

# Agregar línea de promedio
fig.add_shape(
    type="line",
    x0=1,
    y0=mean_rerror,
    x1=586,
    y1=mean_rerror,
    line=dict(color="Red", width=2, dash="dash"),
    name="Mean"
)

# Agregar banda de desviación estándar
fig.add_shape(
    type="rect",
    x0=1,
    y0=mean_rerror - std_rerror,
    x1=586,
    y1=mean_rerror + std_rerror,
    line=dict(color="Green", width=1),
    fillcolor="Green",
    opacity=0.2,
    name="standard deviation"
)

# Configuraciones adicionales
fig.update_layout(
    title='Relative Error ',
    xaxis=dict(title='Enface planes',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    yaxis=dict(title='Relative Error',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    legend_title_text='Metric',
    font=dict(family="Times New Roman", size=25),
    font_color='black',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Mostrar gráfico
fig.show()
fig.write_html(path_save+'\\rerror.html')
#%%
# Datos
hdiff_values = metrics[:, 4]
x = np.arange(1, 587)  # Números de plano enface
mean_hdiff = np.mean(hdiff_values)
std_hdiff = np.std(hdiff_values)


# Crear figura
fig = go.Figure()

# Agregar datos de SSIM
fig.add_trace(go.Scatter(x=x, y=hdiff_values, mode='lines', name='hdiff'))

# Agregar línea de promedio
fig.add_shape(
    type="line",
    x0=1,
    y0=mean_hdiff,
    x1=586,
    y1=mean_hdiff,
    line=dict(color="Red", width=2, dash="dash"),
    name="Mean"
)

# Agregar banda de desviación estándar
fig.add_shape(
    type="rect",
    x0=1,
    y0=mean_hdiff - std_hdiff,
    x1=586,
    y1=mean_hdiff + std_hdiff,
    line=dict(color="Green", width=1),
    fillcolor="Green",
    opacity=0.2,
    name="standard deviation"
)

# Configuraciones adicionales
fig.update_layout(
    title='Histogram difference ',
    xaxis=dict(title='Enface planes',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    yaxis=dict(title='Histogram difference',showgrid=True, zeroline=False, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),#, tickformat=".1e"
    legend_title_text='Metric',
    font=dict(family="Times New Roman", size=25),
    font_color='black',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Mostrar gráfico
fig.show()
fig.write_html(path_save+'\\hdiff.html')
#%%
print('SSIM Mean: ',mean_ssim)
print('SSIM STD: ', std_ssim)
print('MSE Mean: ',mean_mse)
print('MSE STD: ', std_mse)
print('PSNR Mean: ',mean_psnr)
print('PSNR STD: ', std_psnr)
print('Relative error Mean: ',mean_rerror)
print('relative error STD: ', std_rerror)
print('histogram difference Mean: ',mean_hdiff)
print('histogram difference STD: ', std_hdiff)
#%% sub sampled
path = r'C:\Users\USER\Documents\GitHub\Fovea\[p.SHARP][s.Eye2a][10-09-2019_13-14-42]\z=(1..586)_x=(1..896)_y=(1..480)-nSpec=\Int_8x8x8x0_3x3x3x0_250_0_50_unitary'
file = 'TNodeIntFlattenRPE.bin'
nZBin = 586
nXBin = 896
nYBin = int(960/2)
tom = np.fromfile(path+'\\'+file,'single')
tomSub = tom.reshape((nZBin,nXBin,nYBin),order='F')
del tom
#%%
tomSub = np.pad(tomSub, ((0, 0), (0, 0), (240, 240) ), mode='constant', constant_values=1)
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
thisbscan=int(308/2)
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

thisbscan=int(180)
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
compare = np.concatenate((volume1, volume2, volume3), axis=2)
# compare = np.transpose(compare, (2, 0, 1))

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

