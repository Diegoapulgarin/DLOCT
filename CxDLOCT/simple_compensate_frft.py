#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift, ifft
import sys
from tqdm import tqdm # for progress bars
from statistics import mean, stdev
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Repositorio
sys.path.append(r'C:\Users\USER\Documents\GitHub\frft') 
import torch
import frft
import frft_gpu as frft_g
import time
#%%

# Parámetros de muestreo
fs = 300 # Frecuencia de muestreo
t = np.arange(0, 1, 1/fs)  # Vector de tiempo
A1 = 0.5
f1 = 50  # Primera frecuencia
w1 = 2*np.pi*f1
phase1 = 0
A2 = 1
f2 = -20  # Segunda frecuencia muy cercana a la primera
w2 = 2*np.pi*f2
phase2 = np.pi/8  # Desfase para crear interferencia
srv1 = A1*np.cos(w1*t+phase1)
srv2 = A2*np.cos(w2*t+phase2)
srv = srv1 + srv2  # Señal combinada valores reales
scv = A1*np.exp(1j*(w1*t+phase1)) + A2*np.exp(1j*(w2*t+phase2)) # Señal compleja con dos frecuencias distintas valores complejos 
# FFT de las señales
fftsrv = (fftshift(fft(srv)))
fftscv = (fftshift(fft(scv)))

srvfreq = np.fft.fftshift(np.fft.fftfreq(len(srv), 1/fs)) # Frecuencias para el eje x del FFT
nSnapshots = fs
alpha = np.linspace( 0., 2.,nSnapshots)
obj_1d_shifted_gpu = torch.from_numpy(srv).cuda()
results = []
gputime = []
for al in tqdm( alpha, total=alpha.size ):
    start = time.time()
    fobj_1d = frft_g.frft( obj_1d_shifted_gpu, al )
    results.append( fftshift(torch.Tensor.numpy(torch.Tensor.cpu(fobj_1d))))
    t_gpu = time.time() - start
    gputime.append( t_gpu*1.e6 )
print( 'Mean GPU time = %f μs'%mean( gputime ) )
n = int(nSnapshots/2)
phasefrft = np.angle(results[n])
peaks, _ = find_peaks(phasefrft, height=0)
# fig,ax = plt.subplots(3,1)
# ax[1].plot(abs(results[n]))
# ax[2].plot(phasefrft)
# ax[0].plot(abs(fftscv))
# plt.plot()
#%%
# %matplotlib auto
z = 150
frftarray = np.array(results)
phasefrft = np.angle(frftarray)

fig = make_subplots(rows=1, cols=3)
fig.add_trace(
    go.Scatter(y=np.abs(fftscv), mode='lines', name='target signal'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=np.abs(frftarray[n, :]), mode='lines', name='frft transform with alpha aprox 1'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=np.abs(np.unwrap(phasefrft[n, :])), mode='lines', name='unwrapped phase'),
    row=1, col=3
)
fig.update_layout(
    title_text="Análisis de Señales",
    height=900,  # Ajusta la altura si es necesario
    showlegend=False
)
fig.update_xaxes(title_text="Señal objetivo", row=1, col=1)
fig.update_xaxes(title_text="Transformada FRFT con alpha ~1", row=1, col=2)
fig.update_xaxes(title_text="Fase desenvuelta", row=1, col=3)
fig.show()
# plt.imshow(abs(frftarray),cmap='viridis')
#%%
n = int(nSnapshots / 2)
alpha_central = frftarray[n, :]
phase_central = phasefrft[n, :]
unwrapped_phase = abs(np.unwrap(phase_central))
peaks, _ = find_peaks(np.abs(fftsrv),height=0.5)  # review criteria
range_to_search = len(fftsrv) // 2 
fftsrv_compensated = np.copy(fftsrv)
for peak in peaks:
    is_real_peak = unwrapped_phase[peak] > unwrapped_phase[range_to_search * 2 - peak]
    peak_to_attenuate = peak if not is_real_peak else range_to_search * 2 - peak
    N = len(fftsrv)
    if peak_to_attenuate < N / 2:
        frequency = (peak_to_attenuate / N) * fs
    else:
        frequency = ((peak_to_attenuate - N) / N) * fs
    A = np.abs(fftsrv[peak_to_attenuate])
    attenuation_signal = fft(A * np.exp(-1j * (2 * np.pi * frequency * t)))
    fftsrv_compensated -= attenuation_signal
signal_compensated = np.fft.ifft(fftsrv_compensated)
#%%
fig = make_subplots(rows=1, cols=2)
fig.add_trace(
    go.Scatter(y=np.abs(fftscv), mode='lines', name='target signal'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=np.abs(fftsrv_compensated), mode='lines', name='compensated signal'),
    row=1, col=2
)
fig.update_layout(
    title_text="Results",
    height=900,  # Ajusta la altura si es necesario
    showlegend=False
)
fig.update_xaxes(title_text="Señal objetivo", row=1, col=1)
fig.update_xaxes(title_text="compensated signal", row=1, col=2)

fig.show()