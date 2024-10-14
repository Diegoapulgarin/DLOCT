#%%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import hilbert
import plotly.express as px
import os
#%%
fs = 300  # Frecuencia de muestreo
t = np.arange(0, 1, 1/fs)  # Vector de tiempo

# Parámetros de las señales
A1 = 1
f1 = 10  # Primera frecuencia
w1 = 2*np.pi*f1
phase1 = 0

A2 = 0.5
f2 = -20  # Segunda frecuencia muy cercana a la primera
w2 = 2*np.pi*f2
phase2 = np.pi/8  # Desfase para crear interferencia

# Generación de señales coseno con dos frecuencias distintas
srv1 = A1*np.cos(w1*t+phase1)
srv2 = A2*np.cos(w2*t+phase2)
srv = srv1 + srv2  # Señal combinada

# Señal compleja con dos frecuencias distintas
scv = A1*np.exp(1j*(w1*t+phase1)) + A2*np.exp(1j*(w2*t+phase2))

real_signal = np.real(scv) 
analytic_signal = srv + 1j*hilbert(srv)
# imaginary_signal = analytic_signal.imag

fftsrv = abs(np.fft.fftshift(np.fft.fft(srv)))
fftscv = abs(np.fft.fftshift(np.fft.fft(scv)))
fftrec = abs(np.fft.fftshift(np.fft.fft(analytic_signal)))
srvfreq = np.fft.fftshift(np.fft.fftfreq((len(fftscv)),1/fs))

fig = make_subplots(rows=3, cols=1)

# Añadir las trazas
fig.add_trace(go.Scatter(y=srv, x=t, name='Señal Real'), row=1, col=1)
fig.add_trace(go.Scatter(y=np.real(scv), x=t, line=dict(dash='dash'), name='Componente real de la señal compleja'), row=1, col=1)
fig.add_trace(go.Scatter(y=np.imag(scv), x=t, line=dict(dash='dot'), name='Componente imaginaria de la señal compleja'), row=1, col=1)
fig.add_trace(go.Scatter(y=fftsrv, x=srvfreq, name='FFT señal real'), row=2, col=1)
fig.add_trace(go.Scatter(y=fftscv, x=srvfreq, name='FFT señal compleja'), row=3, col=1)

# Actualizar el diseño para centrar la leyenda en la parte inferior y ajustar el tamaño
fig.update_layout(
    font=dict(
        family="Times New Roman",  # Cambiar la fuente
        size=20,  # Cambiar el tamaño de la fuente
    ),
    legend=dict(
        font=dict(
            family="Times New Roman",
            size=20  # Aumentar el tamaño de la fuente de la leyenda
        ),
        orientation="h",  # Orientación horizontal
        yanchor="bottom",  # Anclaje de la leyenda
        y=-0.3,  # Ajustar la posición vertical (más abajo)
        xanchor="center",  # Anclaje horizontal centrado
        x=0.5,  # Posición horizontal centrada
        bgcolor="rgba(255, 255, 255, 0.8)",  # Fondo semitransparente para la leyenda (opcional)
        borderwidth=1  # Borde alrededor de la leyenda (opcional)
    ),
    xaxis=dict(
        title="Tiempo",
        title_font=dict(size=20),  # Tamaño de la fuente del título del eje x
        automargin=True,  # Ajustar automáticamente el margen
    ),
    
    yaxis=dict(
        title="Amplitud",  # Corregir el título del eje y
        title_font=dict(size=22),
        tickfont=dict(size=20)
    ),    
    # Ajustar los márgenes de la figura para evitar superposición
    margin=dict(
        t=1,  # Margen superior
        b=5,  # Margen inferior
    ),
    
    # Actualizar los ejes de las filas 2 y 3
    xaxis2=dict(
        title="Frecuencia",
        title_font=dict(size=20),  # Tamaño de fuente para el título de los ejes
        tickfont=dict(size=20)
    ),
    yaxis2=dict(
        title="Magnitud FFT",
        title_font=dict(size=20),
        tickfont=dict(size=20)
    ),
    
    xaxis3=dict(
        title="Frecuencia",
        title_font=dict(size=20),
        tickfont=dict(size=20)
    ),
    yaxis3=dict(
        title="Magnitud FFT",
        title_font=dict(size=20),
        tickfont=dict(size=20)
    )
)

# Mostrar la gráfica
fig.show()


# Guardar el archivo HTML
path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Bibliografía complex conjugate mirror terms\imagenes_tesis'
fig.write_html(os.path.join(path, 'Basic_complex_problem.html'))

#%%

intensity = np.abs(scv)*np.cos(np.angle(scv))
fft_intensity = abs(np.fft.fftshift(np.fft.fft(intensity)))
fig=px.line(y = fft_intensity,x=srvfreq)
fig.show()

#%%
analytic_signal2 = hilbert(intensity)
recscv2 = intensity + 1j*analytic_signal2.imag
fft_intensity2 = abs(np.fft.fftshift(np.fft.fft(recscv2)))
fig=px.line(y = fft_intensity2, x = srvfreq)
fig.show()

#%%
amp = abs(recscv2)
phase = np.angle(recscv2)
crec = amp*np.cos(phase)
fig = make_subplots(rows=2,cols=1)
fig.add_trace(go.Line(y=amp,x=t,name='Amplitude'),row=1,col=1)
fig.add_trace(go.Line(y=phase,x=t,name='phase'),row=1,col=1)
fig.add_trace(go.Line(y=crec,x=t,name = 'cosine'),row=2,col=1)
fig.show()
