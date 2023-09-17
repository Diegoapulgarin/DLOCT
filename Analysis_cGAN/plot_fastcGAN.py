#%%
import numpy as np 
import plotly.graph_objects as go
path = r'C:\Users\USER\Documents\GitHub\Fovea'
file = '\\cGAN_1_metrics_log.npy'
metrics = np.load(path+file)
#%%
'''
    metrics in file
    0 ssims
    1 ssims_std
    2 ssims_uncertainty
    3 phasemetric
    4 phasemetricCorrected
    5 mse
    6 mse_std
    7 mse_uncertainty

'''
ssim_metric = metrics[0:81,0]
ssim_std = metrics[0:81,1]
mse_values = metrics[0:81,5]
mse_std_values = metrics[0:81,5]

fig = go.Figure()
fig.add_trace(go.Scatter( y=ssim_metric, mode='lines', name='SSIM', line=dict(width=4.5)))
fig.add_trace(go.Scatter( y=ssim_metric + ssim_std, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='+1 STD',showlegend=False))
fig.add_trace(go.Scatter( y=ssim_metric - ssim_std, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='-1 STD',showlegend=False))


# Valor de SSIM en la época 11
epoch_11_value = ssim_metric[11]
epoch_04_value = ssim_metric[4]
epoch_20_value = ssim_metric[20]

# Agregar anotación con el valor de SSIM en la época 11
annotation_text = f"SSIM: {epoch_11_value:.4f}" 
fig.add_annotation(
    x=11, 
    y=epoch_11_value+0.002*epoch_11_value, 
    text=annotation_text,
    showarrow=True,
    arrowhead=3,
    ax=0,
    ay=-20, 
)
annotation_text = f"SSIM: {epoch_04_value:.4f}" 
fig.add_annotation(
    x=4, 
    y=epoch_04_value+0.002*epoch_04_value, 
    text=annotation_text,
    showarrow=True,
    arrowhead=3,
    ax=0,
    ay=-20, 
)
annotation_text = f"SSIM: {epoch_20_value:.4f}" 
fig.add_annotation(
    x=20, 
    y=epoch_20_value+0.002*epoch_20_value, 
    text=annotation_text,
    showarrow=True,
    arrowhead=3,
    ax=0,
    ay=-20, 
)
font_dict = dict(family="Times New Roman", size=14, color="black")
fig.update_layout(
    yaxis=dict(range=[0.7, 1], title='SSIM Value', showgrid=True, zeroline=True, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    xaxis=dict(title='Epoch', showgrid=True, zeroline=True, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    font=font_dict,
    plot_bgcolor='rgba(0,0,0,0)',
)
fig.show()
fig.write_html(path+'\\ssim.html')
#%%

fig = go.Figure()


fig.add_trace(go.Scatter(y=mse_values, mode='lines', name='MSE', line=dict(width=3.5)))  # Engrosamos la línea aquí


fig.add_trace(go.Scatter(y=mse_values + mse_std_values, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='+1 STD', showlegend=False))
fig.add_trace(go.Scatter(y=mse_values - mse_std_values, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='+1 STD', showlegend=False))

epoch_11_mse_value = mse_values[11]
annotation_text = f"MSE: {epoch_11_mse_value:.4f}"
fig.add_annotation(
    x=11, 
    y=epoch_11_mse_value+0.02*epoch_11_mse_value, 
    text=annotation_text,
    showarrow=True,
    arrowhead=4,
    ax=0,
    ay=-20 
)

epoch_04_mse_value = mse_values[4]
annotation_text = f"MSE: {epoch_04_mse_value:.4f}"
fig.add_annotation(
    x=4, 
    y=epoch_04_mse_value+0.02*epoch_04_mse_value, 
    text=annotation_text,
    showarrow=True,
    arrowhead=4,
    ax=0,
    ay=-20 
)

epoch_20_mse_value = mse_values[20]
annotation_text = f"MSE: {epoch_20_mse_value:.4f}"
fig.add_annotation(
    x=20, 
    y=epoch_20_mse_value+0.02*epoch_20_mse_value, 
    text=annotation_text,
    showarrow=True,
    arrowhead=4,
    ax=0,
    ay=-20 
)

# Configuraciones adicionales para el gráfico
fig.update_layout(
    yaxis=dict(title='MSE Value', range=[0, 0.04], showgrid=True, zeroline=True, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'), # Añadimos la malla y la línea del eje y
    xaxis=dict(title='Epoch', showgrid=True, zeroline=True, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'), # Añadimos la malla y la línea del eje x
    font=dict(family="Times New Roman", size=14),
    plot_bgcolor='rgba(0,0,0,0)',
)

fig.show()
fig.write_html(path+'\\mse.html')
#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Crear la figura con subplots
fig = make_subplots(rows=2, cols=1, subplot_titles=("SSIM Metric", "MSE Metric"))

# Agregar trazas para SSIM
fig.add_trace(go.Scatter(y=ssim_metric, mode='lines', name='SSIM', line=dict(width=4.5)), row=1, col=1)
fig.add_trace(go.Scatter(y=ssim_metric + ssim_std, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='+1 STD',showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(y=ssim_metric - ssim_std, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='-1 STD',showlegend=False), row=1, col=1)

# Agregar trazas para MSE
fig.add_trace(go.Scatter(y=mse_values, mode='lines', name='MSE', line=dict(width=3.5)), row=2, col=1)
fig.add_trace(go.Scatter(y=mse_values + mse_std_values, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='+1 STD', showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(y=mse_values - mse_std_values, fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='-1 STD',showlegend=False), row=2, col=1)

for epoch, value in [(11, mse_values[11]), (4, mse_values[4]), (20, mse_values[20])]:
    annotation_text = f"MSE: {value:.4f}"
    fig.add_annotation(x=epoch, y=value+0.02*value, text=annotation_text, showarrow=True, arrowhead=4, ax=0, ay=-20, row=2, col=1)

for epoch, value in [(11, ssim_metric[11]), (4, ssim_metric[4]), (20, ssim_metric[20])]:
    annotation_text = f"MSE: {value:.4f}"
    fig.add_annotation(x=epoch, y=value+0.002*value, text=annotation_text, showarrow=True, arrowhead=4, ax=0, ay=-20, row=1, col=1)


# Configuración general para la figura
fig.update_layout(
    yaxis1=dict(title='SSIM Value', range=[0.7, 1],showgrid=True, zeroline=True, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    yaxis2=dict(title='MSE Value', range=[0, 0.04], showgrid=True, zeroline=True, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    xaxis1=dict(title='Epoch', showgrid=True, zeroline=True, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    xaxis2=dict(title='Epoch', showgrid=True, zeroline=True, showline=True, gridcolor='rgba(128,128,128,0.3)', zerolinecolor='rgba(128,128,128,0.5)', linecolor='rgba(0,0,0,1)'),
    font=dict(family="Times New Roman", size=14),
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.show()
fig.write_html(path+'\\combined.html')
