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
import cv2
from scipy.spatial.distance import cosine as simcos
#%%

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
def histogram_difference(image1, image2, method="chi-squared"):
    # Calcular histogramas
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 1]) # asumiendo que la imagen está normalizada entre 0 y 1
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 1])

    # Normalizar histogramas si se va a usar Kullback-Leibler
    if method == "kullback-leibler":
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()
    
    # Calcular diferencia de histogramas
    if method == "chi-squared":
        # Usar distancia chi-cuadrado
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    elif method == "kullback-leibler":
        # Usar divergencia de Kullback-Leibler
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)
    elif method == 'cosine-similarity':
        return simcos(np.ravel(hist1),np.ravel(hist2))
    else:
        raise ValueError("Método no reconocido")
i=20
enfaceOriginal = 10*np.log10(abs(tomDatas[i,:,:])**2)
enfaceOriginal = enfaceOriginal/np.max(enfaceOriginal)
enfaceReconstructed=10*np.log10(abs(tomDatar[i,:,:])**2)
enfaceReconstructed = enfaceReconstructed/np.max(enfaceReconstructed)

ssim_result =calculate_ssim(enfaceOriginal,enfaceReconstructed)
mse = calculate_mse(enfaceOriginal,enfaceReconstructed)
psnr = calculate_psnr(enfaceOriginal,enfaceReconstructed)
rerror = relative_error(enfaceOriginal,enfaceReconstructed)
hdiff = histogram_difference(np.float32(enfaceOriginal),np.float32(enfaceReconstructed),method='cosine-similarity')
print("SSIM:", ssim_result)
print("MSE:", mse)
print("PSNR:", psnr)
print("relative error:", rerror)
print("histogram difference:", hdiff)
#%%
nZBin = 400
metrics_complete = []
for i in tqdm(range(nZBin)):
    enfaceOriginal = 10*np.log10(abs(tomDatas[i,:,:])**2)
    enfaceOriginal = enfaceOriginal/np.max(enfaceOriginal)
    enfaceReconstructed=10*np.log10(abs(tomDatar[i,:,:])**2)
    enfaceReconstructed = enfaceReconstructed/np.max(enfaceReconstructed)
    ssim_result =calculate_ssim(enfaceOriginal,enfaceReconstructed)
    mse = calculate_mse(enfaceOriginal,enfaceReconstructed)
    psnr = calculate_psnr(enfaceOriginal,enfaceReconstructed)
    # rerror = relative_error(enfaceOriginal,enfaceReconstructed)
    hdiff = histogram_difference(np.float32(enfaceOriginal),np.float32(enfaceReconstructed),method='cosine-similarity')
    metrics_results = np.array((ssim_result,mse,psnr,hdiff))
    metrics_complete.append(metrics_results)
#%%
path_save = r'C:\Users\USER\Documents'
metrics = np.array(metrics_complete)
plt.plot(metrics[:,0])
# np.save(path_save+'\\metrics_TNode',metrics)
#%%
import plotly.graph_objects as go
import numpy as np

# Suponemos que 'metrics' es tu array numpy con la forma (512, 4)
# metrics = np.random.random((512, 4))  # Ejemplo de datos, reemplaza con tus datos reales

# Etiquetas para las métricas
metric_labels = ["SSIM", "MSE", "PSNR", "histogram difference"]

# Función para crear el gráfico con dos ejes Y y rangos opcionales
def create_dual_axis_plot(title, y1_data, y1_label, y2_data, y2_label, filename, y1_range=None, y2_range=None):
    # Calcular los promedios
    # Calcular los promedios y desviaciones estándar
    y1_mean = np.mean(y1_data)
    y1_std = np.std(y1_data)
    y2_mean = np.mean(y2_data)
    y2_std = np.std(y2_data)
    
    x_values = np.arange(len(y1_data))
    
    # Crear las zonas de desviación estándar para y1
    y1_upper = y1_mean + y1_std
    y1_lower = y1_mean - y1_std
    
    # Crear las zonas de desviación estándar para y2
    y2_upper = y2_mean + y2_std
    y2_lower = y2_mean - y2_std
    # Crear figura
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate((x_values, x_values[::-1])),  # x, luego x reversa
        y=np.concatenate((y1_lower * np.ones(len(y1_data)), (y1_upper * np.ones(len(y1_data)))[::-1])),  # y1 inferior, luego y1 superior reversa
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name=f'{y1_label} STD',
    ))

    # Añadir la primera línea con el eje Y primario
    fig.add_trace(go.Scatter(x=np.arange(len(y1_data)), y=y1_data, name=y1_label, mode='lines',
                             line=dict(color='blue')))
    # Añadir línea punteada para el promedio de y1
    fig.add_trace(go.Scatter(x=np.arange(len(y1_data)), y=[y1_mean]*len(y1_data), name=f'{y1_label} mean',
                             mode='lines', line=dict(color='blue', dash='dash')))
    
    # Configurar eje Y primario
    yaxis_config = {
        'title': y1_label,
        'titlefont': dict(color='blue'),
        'tickfont': dict(color='blue')
    }
    if y1_range:
        yaxis_config['range'] = y1_range
    
    fig.update_layout(
        title=title,
        xaxis_title='En-face slices',
        yaxis=yaxis_config
    )

    fig.add_trace(go.Scatter(
        x=np.concatenate((x_values, x_values[::-1])),  # x, luego x reversa
        y=np.concatenate((y2_lower * np.ones(len(y2_data)), (y2_upper * np.ones(len(y2_data)))[::-1])),  # y2 inferior, luego y2 superior reversa
        fill='toself',
        fillcolor='rgba(240,100,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name=f'{y2_label} STD',
        yaxis='y2'  # Importante para asignar al eje y correcto
    ))
    # Añadir la segunda línea con el eje Y secundario
    fig.add_trace(go.Scatter(x=np.arange(len(y2_data)), y=y2_data, name=y2_label, mode='lines',
                             line=dict(color='red'), yaxis='y2'))
    # Añadir línea punteada para el promedio de y2
    fig.add_trace(go.Scatter(x=np.arange(len(y2_data)), y=[y2_mean]*len(y2_data), name=f'{y2_label} mean',
                             mode='lines', line=dict(color='red', dash='dash'), yaxis='y2'))
    
    # Configurar eje Y secundario
    yaxis2_config = {
        'title': y2_label,
        'titlefont': dict(color='red'),
        'tickfont': dict(color='red'),
        'overlaying': 'y',
        'side': 'right'
    }
    if y2_range:
        yaxis2_config['range'] = y2_range
    
    fig.update_layout(
        yaxis2=yaxis2_config
    )

    # Guardar la figura como un archivo HTML
    fig.write_html(f'{filename}.html')
    # fig.show()

# Ejemplo de uso con rangos personalizados
create_dual_axis_plot('Comparación de SSIM y MSE', metrics[:, 0], 'SSIM', metrics[:, 1], 'MSE', 'ssim_mse_comparison_flat', y2_range=[0, 0.2])
create_dual_axis_plot('Comparación de PSNR y histogram difference', metrics[:, 2], 'PSNR', metrics[:, 3], 'histogram difference', 'psnr_histogram_comparison_flat', y1_range=[0, 30], y2_range=[0, 2])

#%%
i=250
enfaceOriginal = 10*np.log10(abs(tomDatas[:,i,:])**2)
# enfaceOriginal = enfaceOriginal/np.max(enfaceOriginal)
enfaceReconstructed=10*np.log10(abs(tomDatar[:,i,:])**2)
# enfaceReconstructed = enfaceReconstructed/np.max(enfaceReconstructed)
save_path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\partial_results\segunda revisión\nail'
dpi = 300
n = 700
plot = (enfaceOriginal)
plt.rcParams['figure.dpi']=dpi
plt.imshow(plot,cmap='gray',vmax=120,vmin=60)
plt.axis('off')
plt.savefig(os.path.join(save_path,f'nail_original_x={i}_{np.shape(plot)[0]}x{np.shape(plot)[1]}'), dpi=dpi, format=None, metadata=None,
        bbox_inches='tight', pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None,
       )

plot = (enfaceReconstructed)
plt.rcParams['figure.dpi']=dpi
plt.imshow(plot,cmap='gray',vmax=120,vmin=60)
plt.axis('off')
plt.savefig(os.path.join(save_path,f'nail_reconstructed_x={i}_{np.shape(plot)[0]}x{np.shape(plot)[1]}'), dpi=dpi, format=None, metadata=None,
        bbox_inches='tight', pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None,
       )

