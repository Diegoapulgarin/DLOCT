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
fig,ax = plt.subplots(3,1)
ax[1].plot(abs(results[n]))
ax[2].plot(phasefrft)
ax[0].plot(abs(fftscv))
plt.plot()
#%%
# %matplotlib auto
z = 150
frftarray = np.array(results)
phasefrft = np.angle(frftarray)

fig = make_subplots(rows=3, cols=1)
fig.add_trace(
    go.Scatter(y=np.abs(fftscv), mode='lines', name='target signal'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=np.abs(frftarray[n, :]), mode='lines', name='frft transform with alpha aprox 1'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=np.abs(np.unwrap(phasefrft[n, :])), mode='lines', name='unwrapped phase'),
    row=3, col=1
)
fig.update_layout(
    title_text="Análisis de Señales",
    height=900,  # Ajusta la altura si es necesario
    showlegend=False
)
fig.update_xaxes(title_text="Señal objetivo", row=1, col=1)
fig.update_xaxes(title_text="Transformada FRFT con alpha ~1", row=2, col=1)
fig.update_xaxes(title_text="Fase desenvuelta", row=3, col=1)
fig.show()
#%%
# plt.imshow(abs(frftarray),cmap='viridis')
