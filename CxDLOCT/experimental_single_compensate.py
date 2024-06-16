#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\frft')
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\Analysis_cGAN')
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift, ifft
from tqdm import tqdm
from statistics import mean, stdev
from scipy.signal import find_peaks
# Repositorio
from Deep_Utils import dbscale
import torch
import frft
import frft_gpu as frft_g
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_dimensions(file_name):
    parts = file_name.split('_')
    dimensions = []
    for part in parts:
        if 'z=' in part or 'x=' in part or 'y=' in part:
            number = int(part.split('=')[-1])
            dimensions.append(number)
    return tuple(dimensions)

def read_tomogram(file_path, dimensions):
    depth, height, width = dimensions
    with open(file_path, 'rb') as file:
        tomogram = np.fromfile(file, dtype='single')
        tomogram = tomogram.reshape((depth, height, width),order='F')
    return tomogram
#%%

pathcomplex = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscan'
artifact_files = os.listdir(pathcomplex)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(pathcomplex, real_file)
        imag_file_path = os.path.join(pathcomplex, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tomcc = tomReal + 1j * tomImag
        del tomImag, tomReal
# fringescc = fftshift(ifft(fftshift(tom,axes=0),axis=0),axes=0)
fringescc = fftshift(ifft(tomcc,axis=0),axes=0)

pathcomplex = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscanNoartifacts'
artifact_files = os.listdir(pathcomplex)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(pathcomplex, real_file)
        imag_file_path = os.path.join(pathcomplex, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        del tomImag, tomReal
# fringescc = fftshift(ifft(fftshift(tom,axes=0),axis=0),axes=0)
fringes = fftshift(ifft(tom,axis=0),axes=0)
#%%
bscan = 0
x = 512
plot = dbscale(tomcc[:,:,bscan])
signal = (fringescc[:,x,bscan])
signalOriginal = abs(fringes[:,x,bscan])
nSnapshots = len(signal)
fs = nSnapshots
alpha = np.linspace( 0., 2.,nSnapshots)
obj_1d_shifted_gpu = torch.from_numpy(signal).cuda()
results = []
gputime = []
for al in tqdm( alpha, total=alpha.size ):
    start = time.time()
    fobj_1d = frft_g.frft( obj_1d_shifted_gpu, al )
    results.append( fftshift(torch.Tensor.numpy(torch.Tensor.cpu(fobj_1d))))
    t_gpu = time.time() - start
    gputime.append( t_gpu*1.e6 )
print( 'Mean GPU time = %f μs'%mean( gputime ) )

t = np.arange(0, fs)
n = int(nSnapshots/2)
frftarray = np.array(results)
phasefrft = np.angle(frftarray)
alpha_central = frftarray[n, :]
phase_central = phasefrft[n, :]
phase_original = np.angle((fringes[:,x,bscan]))
phase_complex = np.angle((fringescc[:,x,bscan]))
unwrapped_original = abs(np.unwrap(phase_original))
unwrapped_complex = abs(np.unwrap(phase_complex))
unwrapped_phase = abs(np.unwrap(phase_central))
peaks, _ = find_peaks(np.abs(tomcc[:,x,bscan]),height=0.5)  # review criteria
range_to_search = len(tomcc[:,x,bscan]) // 2 
fftsrv_compensated = np.copy(tomcc[:,x,bscan])

#%%
def normalize(array):
     return array/np.max(array)

offset = np.max(np.abs(tomcc[:,x,bscan]))-np.max(unwrapped_phase)
fig = make_subplots(rows=3, cols=1)
fig.add_trace(
    go.Scatter(y=np.abs(tomcc[:,x,bscan]), mode='lines', name='artifacts mirror'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(y=np.abs(tom[:,x,bscan]), mode='lines', name='target'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(y=np.abs(tomcc[:,x,bscan]-tom[:,x,bscan]), mode='lines', name='unwrapped phase'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(y=(normalize(unwrapped_phase)), mode='lines', name='frft'),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(y=(normalize(unwrapped_original)), mode='lines', name='target'),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(y=(normalize(unwrapped_complex)), mode='lines', name='cc'),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(y=abs(signal), mode='lines', name='complex'),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(y=(signalOriginal), mode='lines', name='target'),
    row=3, col=1
)

fig.update_layout(
    title_text="Comparative",
    height=900,  # Ajusta la altura si es necesario
    showlegend=False
)
fig.update_xaxes(title_text="artifact mirror", row=1, col=1)
fig.update_xaxes(title_text="free artifact", row=2, col=1)
fig.update_xaxes(title_text="phase unwrap", row=3, col=1)

# fig.show()
fig.write_html('phase_unwrap.html')

#%% compensate
# for peak in peaks:
#     is_real_peak = unwrapped_phase[peak] > unwrapped_phase[range_to_search * 2 - peak]
#     peak_to_attenuate = peak if not is_real_peak else range_to_search * 2 - peak
#     N = len(tomcc[:,x,bscan])
#     if peak_to_attenuate < N / 2:
#         frequency = (peak_to_attenuate / N) * fs
#     else:
#         frequency = ((peak_to_attenuate - N) / N) * fs
#     A = np.abs(tomcc[:,x,bscan][peak_to_attenuate])
#     attenuation_signal = fft(A * np.exp(-1j * (2 * np.pi * frequency * t)))
#     fftsrv_compensated -= attenuation_signal
# signal_compensated = fftshift(ifft(fftsrv_compensated))