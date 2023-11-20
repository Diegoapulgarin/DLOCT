import numpy as np
from scipy.signal import hilbert
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft
from PyEMD import EMD
import matplotlib.pyplot as plt

# Parámetros de muestreo
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

# Parámetros de la modulación
f_carrier = 180  # Frecuencia de la portadora para la modulación

carrier = np.cos(2 * np.pi * f_carrier * t)  # Señal portadora
# Modulación de la señal
modulated_signal = srv * carrier
# FFT de las señales
fftsrv = np.abs(np.fft.fftshift(np.fft.fft(srv)))
fftscv = np.abs(np.fft.fftshift(np.fft.fft(scv)))
fft_modulated_shifted = np.abs(np.fft.fftshift(np.fft.fft(modulated_signal)))



# Frecuencias para el eje x del FFT
srvfreq = np.fft.fftshift(np.fft.fftfreq(len(srv), 1/fs))

# Gráficas
fig = make_subplots(rows=4, cols=1)

# Señal en el tiempo
fig.add_trace(go.Scatter(y=modulated_signal, x=t, mode='lines', name='Cosine'), row=1, col=1)
fig.add_trace(go.Scatter(y=np.real(scv), x=t, mode='lines', name='Real Part'), row=1, col=1)
fig.add_trace(go.Scatter(y=np.imag(scv), x=t, mode='lines', name='Imaginary Part'), row=1, col=1)

# FFT de la señal real
fig.add_trace(go.Scatter(y=fftsrv, x=srvfreq, mode='lines', name='FFT Real Signal'), row=2, col=1)

# FFT de la señal compleja
fig.add_trace(go.Scatter(y=fftscv, x=srvfreq, mode='lines', name='FFT Complex Signal'), row=3, col=1)

# FFT de la señal real al aplicarle el ventaneo
fig.add_trace(go.Scatter(y=fft_modulated_shifted, x=srvfreq, mode='lines', name='FFT real signal modulated with 10Hz'), row=4, col=1)

# Ajustes finales y mostrar la figura
fig.update_layout(height=600, width=800, title_text="Signal Analysis")
fig.show()

f, t2, Zxx = stft(srv, fs, nperseg=128)


# Graficar el espectrograma con Matplotlib
plt.figure(figsize=(10, 6))
plt.pcolormesh(t2, f, abs(Zxx), shading='gouraud')
plt.colorbar(label='Intensity')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('STFT Magnitude')
plt.tight_layout()
plt.show()

emd = EMD()
IMFs = emd(srv)

# Graficar los IMFs
plt.figure(figsize=(12, 9))
for i, IMF in enumerate(IMFs, start=1):
    plt.subplot(len(IMFs), 1, i)
    plt.plot(t, IMF)
    plt.title('IMF ' + str(i))
    plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()

from scipy.fft import fft, fftfreq

# Suponiendo que IMFs es un array donde cada fila es un IMF
# y t es tu vector de tiempo correspondiente

# Preparar el entorno de la figura para múltiples subplots
plt.figure(figsize=(15, 10))

# Número total de IMFs
num_imfs = IMFs.shape[0]

# Iterar sobre cada IMF
for i in range(num_imfs):
    # Calcular la FFT del IMF actual
    yf = fft(IMFs[i])
    
    # Generar el vector de frecuencias correspondiente
    xf = fftfreq(t.size, 1 / fs)
    
    # Solo tomar la mitad del espectro para el gráfico
    # xf = xf[:t.size//2]
    yf = np.abs(yf)
    
    # Crear un subplot para el IMF actual
    plt.subplot(num_imfs, 1, i + 1)
    plt.plot(xf, yf)
    plt.title(f'Espectro de Frecuencia para IMF {i+1}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')

# Ajustar los subplots y mostrar la figura
plt.tight_layout()
plt.show()


