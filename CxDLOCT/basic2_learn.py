import numpy as np
from scipy.signal import hilbert
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft
from PyEMD import EMD
import matplotlib.pyplot as plt
import pywt

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
f_carrier = 40  # Frecuencia de la portadora para la modulación

carrier = np.cos(2 * np.pi * f_carrier * t)  # Señal portadora
# Modulación de la señal
modulated_signal = scv * carrier


# FFT de las señales
fftsrv = np.abs(np.fft.fftshift(np.fft.fft(srv)))
fftscv = np.abs(np.fft.fftshift(np.fft.fft(scv)))
fft_modulated_shifted = np.abs(np.fft.fftshift(np.fft.fft(modulated_signal)))



# Frecuencias para el eje x del FFT
srvfreq = np.fft.fftshift(np.fft.fftfreq(len(srv), 1/fs))


# f, t2, Zxx = stft(srv, fs, nperseg=128)


# # Graficar el espectrograma con Matplotlib
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(t2, f, abs(Zxx), shading='gouraud')
# plt.colorbar(label='Intensity')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title('STFT Magnitude')
# plt.tight_layout()
# plt.show()

# emd = EMD()
# IMFs = emd(srv)

# Graficar los IMFs
# plt.figure(figsize=(12, 9))
# for i, IMF in enumerate(IMFs, start=1):
#     plt.subplot(len(IMFs), 1, i)
#     plt.plot(t, IMF)
#     plt.title('IMF ' + str(i))
#     plt.xlabel('Time [s]')
# plt.tight_layout()
# plt.show()


# Suponiendo que 'signal' es tu señal de entrada
# Aplica EMD para obtener IMFs
# emd = EMD()
# IMFs = emd.emd(srv)

# # Procesa cada IMF
# for i, IMF in enumerate(IMFs):
#     # Obtén la señal analítica
#     analytic_signal = hilbert(IMF)
#     # Calcula la amplitud instantánea
#     instantaneous_amplitude = np.abs(analytic_signal)
#     # Calcula la fase instantánea
#     instantaneous_phase = np.unwrap(np.angle(analytic_signal))
#     # Calcula la frecuencia instantánea
#     instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (1/fs))

    # Grafica la amplitud y la frecuencia instantánea
    # plt.figure(figsize=(12, 8))

    # plt.subplot(211)
    # plt.plot(t, instantaneous_amplitude)
    # plt.title(f'Instantaneous Amplitude of IMF {i+1}')

    # plt.subplot(212)
    # plt.plot(t[1:], instantaneous_frequency)
    # plt.title(f'Instantaneous Frequency of IMF {i+1}')
    # plt.show()

# Asumiendo que tienes una lista de IMFs y su tiempo correspondiente 't'
# y que 'fs' es la frecuencia de muestreo

# reconstructed_signal_real = np.zeros_like(t)
# reconstructed_signal_imag = np.zeros_like(t)

# for IMF in IMFs:  # selected_IMFs es una lista de los IMFs seleccionados
#     analytic_signal = hilbert(IMF)
#     instantaneous_amplitude = np.abs(analytic_signal)
#     instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
#     reconstructed_signal_real += instantaneous_amplitude * np.cos(instantaneous_phase)
#     reconstructed_signal_imag += instantaneous_amplitude * np.sin(instantaneous_phase)

# # Combinar las partes real e imaginaria para formar la señal compleja
# reconstructed_signal_complex = reconstructed_signal_real + 1j * reconstructed_signal_imag


# fft_reconstructed_IMFs = np.abs(np.fft.fftshift(np.fft.fft(reconstructed_signal_complex)))

# Gráficas
fig = make_subplots(rows=4, cols=1)

# Señal en el tiempo
# fig.add_trace(go.Scatter(y=modulated_signal, x=t, mode='lines', name='Cosine'), row=1, col=1)
fig.add_trace(go.Scatter(y=np.real(scv), x=t, mode='lines', name='Real Part'), row=1, col=1)
fig.add_trace(go.Scatter(y=np.imag(scv), x=t, mode='lines', name='Imaginary Part'), row=1, col=1)

# FFT de la señal real
fig.add_trace(go.Scatter(y=fftsrv, x=srvfreq, mode='lines', name='FFT Real Signal'), row=2, col=1)

# FFT de la señal compleja
fig.add_trace(go.Scatter(y=fftscv, x=srvfreq, mode='lines', name='FFT Complex Signal'), row=3, col=1)

# FFT de la señal real al aplicarle el ventaneo
# fig.add_trace(go.Scatter(y=fft_modulated_shifted, x=srvfreq, mode='lines', name='FFT modulated signal'), row=4, col=1)

# Ajustes finales y mostrar la figura
fig.update_layout(height=600, width=800, title_text="Signal Analysis")
fig.show()

# Realiza una CWT
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(srv, scales, 'cmor')

# Grafica la transformada wavelet
plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='jet', aspect='auto',
           vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
plt.show()

plt.plot(srvfreq,fft_modulated_shifted)
