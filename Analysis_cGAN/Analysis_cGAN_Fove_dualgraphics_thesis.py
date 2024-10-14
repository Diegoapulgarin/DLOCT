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
path_save = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Bibliografía complex conjugate mirror terms\imagenes_tesis'
metrics = np.array(metrics_complete)
plt.plot(metrics[:,0])
# np.save(path_save+'\\metrics_TNode',metrics)
#%%
import plotly.graph_objects as go
import numpy as np

metric_labels = ["SSIM", "MSE", "PSNR", "histogram difference"]

# Función para crear el gráfico con dos ejes Y y rangos opcionales
def create_dual_axis_plot(title, y1_data, y1_label, y2_data, y2_label, filename, path_save,xannot1,yannot1,xannot2,yannot2 ,y1_range=None, y2_range=None):
    
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
        x=np.concatenate((x_values, x_values[::-1])),
        y=np.concatenate((y1_lower * np.ones(len(y1_data)), (y1_upper * np.ones(len(y1_data)))[::-1])),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name=f'{y1_label} STD',
    ))

    # Añadir la primera línea con el eje Y primario
    fig.add_trace(go.Scatter(x=np.arange(len(y1_data)), y=y1_data, name=y1_label, mode='lines',
                             line=dict(color='red')))
    # Añadir línea punteada para el promedio de y1
    fig.add_trace(go.Scatter(x=np.arange(len(y1_data)), y=[y1_mean]*len(y1_data), name=f'Promedio del {y1_label}',
                             mode='lines', line=dict(color='red', dash='dash')))
    
    # Configurar eje Y primario
    yaxis_config = {
        'title': y1_label,
        'titlefont': dict(color='red'),
        'tickfont': dict(color='red')
    }
    if y1_range:
        yaxis_config['range'] = y1_range
    
    fig.update_layout(
        title=title,
        xaxis_title='Planos <i>En-face</i>',
        yaxis=yaxis_config,
        font=dict(family='Times New Roman', size=22),  # Ajustar la fuente
    )

    fig.add_trace(go.Scatter(
        x=np.concatenate((x_values, x_values[::-1])),
        y=np.concatenate((y2_lower * np.ones(len(y2_data)), (y2_upper * np.ones(len(y2_data)))[::-1])),
        fill='toself',
        fillcolor='rgba(240,100,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name=f'{y2_label} STD',
        yaxis='y2'
    ))

    # Añadir la segunda línea con el eje Y secundario
    fig.add_trace(go.Scatter(x=np.arange(len(y2_data)), y=y2_data, name=y2_label, mode='lines',
                             line=dict(color='green'), yaxis='y2'))
    # Añadir línea punteada para el promedio de y2
    fig.add_trace(go.Scatter(x=np.arange(len(y2_data)), y=[y2_mean]*len(y2_data), name=f'Promedio de la {y2_label}',
                             mode='lines', line=dict(color='green', dash='dash'), yaxis='y2'))
    
    # Configurar eje Y secundario
    yaxis2_config = {
        'title': y2_label,
        'titlefont': dict(color='green'),
        'tickfont': dict(color='green'),
        'overlaying': 'y',
        'side': 'right'
    }
    if y2_range:
        yaxis2_config['range'] = y2_range

    fig.add_annotation(x=xannot1, y=yannot1,
                   text=f"Media: {y1_mean:.2f} \u00B1 {y1_std:.2f}",
                    showarrow=False,
                    font=dict(family="Times New Roman", size=24, color="red")) 

    fig.add_annotation(x=xannot2, y=yannot2,
                   text=f"Media: {y2_mean:.2f} \u00B1 {y2_std:.2f}",
                   showarrow=False,
                   font=dict(family="Times New Roman", size=24, color="green"))

    fig.update_layout(
        yaxis2=yaxis2_config,
        legend=dict(
            font=dict(
                family="Times New Roman",
                size=22  # Aumentar el tamaño de la fuente de la leyenda
            ),
            orientation="h",  # Orientación horizontal
            yanchor="bottom",  # Anclaje de la leyenda
            y=-0.3,  # Ajustar la posición vertical (más abajo)
            xanchor="center",  # Anclaje horizontal centrado
            x=0.5,  # Posición horizontal centrada,
            borderwidth=1  # Borde alrededor de la leyenda (opcional)
        )
    )
    fig.write_html(os.path.join(path_save, f'{filename}.html'))
    # fig.show()


# Ejemplo de uso con rangos personalizados
create_dual_axis_plot('Comparación de SSIM y MSE', 
                      metrics[:, 0], 
                      'SSIM', metrics[:, 1], 
                      'MSE', 
                      'ssim_mse_comparison_flat', 
                      y1_range=[0.5, 0.8], y2_range=[0, 0.2],
                      xannot1=350,xannot2=350,
                      yannot1=0.67,yannot2=0.57,
                      path_save=path_save)
create_dual_axis_plot('Comparación de PSNR y histogram difference', 
                      metrics[:, 2], 
                      'PSNR', metrics[:, 3], 
                      'Diferencia de histograma', 
                      'psnr_histogram_comparison_flat', 
                      y1_range=[0, 30], y2_range=[0, 2],
                      xannot1=350,xannot2=350,
                      yannot1=23,yannot2=6.8,
                      path_save=path_save)
