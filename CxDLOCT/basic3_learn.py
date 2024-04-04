#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift, ifft
import sys
from tqdm import tqdm # for progress bars
from statistics import mean, stdev
from scipy.signal import find_peaks
# Repositorio
sys.path.append(r'C:\Users\USER\Documents\GitHub\frft') 
import torch
import frft
import frft_gpu as frft_g
import time

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

# Generación de señales coseno con dos frecuencias distintas
srv1 = A1*np.cos(w1*t+phase1)
srv2 = A2*np.cos(w2*t+phase2)
srv = srv1 + srv2  # Señal combinada valores reales

# Señal compleja con dos frecuencias distintas valores complejos 
scv = A1*np.exp(1j*(w1*t+phase1)) + A2*np.exp(1j*(w2*t+phase2))

# Parámetros de la modulación
f_carrier = 40  # Frecuencia de la portadora para la modulación

carrier = np.cos(2 * np.pi * f_carrier * t)  # Señal portadora
# Modulación de la señal
modulated_signal = scv * carrier

# FFT de las señales
fftsrv = (fftshift(fft(srv)))
fftscv = (fftshift(fft(scv)))
fft_modulated_shifted = np.abs(fftshift(fft(modulated_signal)))

# # Frecuencias para el eje x del FFT
srvfreq = np.fft.fftshift(np.fft.fftfreq(len(srv), 1/fs))
# fig,axs = plt.subplots(1,2)
# axs[0].plot(fftsrv)
# axs[1].plot(fftscv)
# fig.show()

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

# fig, ax = plt.subplots( nSnapshots, 2, sharex=True )
# for n in range( nSnapshots ):
#     ax[n,0].plot( np.absolute( results[n] )/np.absolute( results[n].max() ) )
#     ax[n,0].grid() 
#     ax[n,0].set_ylim( [ 0., 1. ] )
#     # ax[n,0].set_yticks( [] )
#     # ax[n,0].set_ylabel( r'$\alpha = %.2f$'%alpha[n] )
    
#     ax[n,1].plot( np.angle( results[n] ) )
#     ax[n,1].grid()

# ax[0,0].set_title( 'Normalized amplitude' )
# ax[0,1].set_title( 'Phase')
# fig.show()
# fig = plt.plot(srvfreq,abs(fftshift(fft(results[10]))))
n = int(nSnapshots/2)
phasefrft = np.angle(results[n])
peaks, _ = find_peaks(phasefrft, height=0)
fig,ax = plt.subplots(3,1)
ax[1].plot(abs(results[n]))
ax[2].plot(phasefrft)
# ax[2].plot(srvfreq[peaks], phasefrft[peaks], "x")
ax[0].plot(abs(fftscv))
plt.plot()
#%%
# signal recovered with differents alpha values
srfrft = abs(fftshift(fft(results[n])))
plt.plot(srvfreq,srfrft)
# unwrapped =np.unwrap(np.angle(results[n]))
# plt.plot(srvfreq,unwrapped)
#%%
#%matplotlib auto
z = 150
frftarray = np.array(results)
phasefrft = np.angle(frftarray)
fig,ax = plt.subplots(4,1)
ax[0].plot(abs(fftscv))
ax[1].plot(abs(frftarray[n,:]))
ax[2].plot(abs(frftarray[:,z]))
ax[3].plot(abs(np.unwrap(phasefrft[n,:])))
# ax[2].plot(srvfreq[peaks], phasefrft[peaks], "x")

#%%
plt.imshow(abs(frftarray),cmap='viridis')
# plt.imshow(phasefrft,cmap='viridis')
#%%
# # from matplotlib import cm
# # %matplotlib auto
# plt.style.use('_mpl-gallery')

# X = np.arange(min(srvfreq), max(srvfreq)+1, 1)
# Y = np.arange(0, len(alpha), 1)
# X, Y = np.meshgrid(X, Y)
# # Plot the surface
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X, Y,frftarray,cmap = 'viridis')

# ax.set(xticklabels=[],
#        yticklabels=[],
#        zticklabels=[])

# plt.show()
#%%  datos experimentales dejar de lado de momento
# def fast_reconstruct(array):
#     tom = fftshift(fft(fftshift(array,axes=0),axis=0),axes=0)
#     return tom
 
# def extract_dimensions(file_name):
#     parts = file_name.split('_')
#     dimensions = []
#     for part in parts:
#         if 'z=' in part or 'x=' in part or 'y=' in part:
#             number = int(part.split('=')[-1])
#             dimensions.append(number)
#     return tuple(dimensions)

# def read_tomogram(file_path, dimensions):
#     depth, height, width = dimensions
#     with open(file_path, 'rb') as file:
#         tomogram = np.fromfile(file, dtype='single')
#         tomogram = tomogram.reshape((depth, height, width),order='F')
#     return tomogram

# path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Data Boston\[DepthWrap]\[p.DepthWrap][s.Fovea][09-20-2023_11-15-34]'
# file = '[p.Calibration][s.Mirror][02-10-2023_15-17-52].dispersion'
# dispersion = np.fromfile(os.path.join(path,file))
# # plt.plot(dispersion)
# path = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscan'
# artifact_files = os.listdir(path)
# for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
#         real_file_path = os.path.join(path, real_file)
#         imag_file_path = os.path.join(path, imag_file)
#         dimensions = extract_dimensions(real_file[:-4])
#         tomReal = read_tomogram(real_file_path, dimensions)
#         tomImag = read_tomogram(imag_file_path, dimensions)
#         tom = tomReal + 1j * tomImag
#         del tomImag, tomReal
# fringescc = fftshift(ifft(tom,axis=0),axes=0)
# fringescc = fringescc[:,:,0:4]
# pathtarget = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscanNoartifacts'
# artifact_files = os.listdir(pathtarget)
# for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
#         real_file_path = os.path.join(pathtarget, real_file)
#         imag_file_path = os.path.join(pathtarget, imag_file)
#         dimensions = extract_dimensions(real_file[:-4])
#         tomReal = read_tomogram(real_file_path, dimensions)
#         tomImag = read_tomogram(imag_file_path, dimensions)
#         tom = tomReal + 1j * tomImag
#         del tomImag, tomReal
# fringesreal = fftshift(ifft(tom,axis=0),axes=0)
# fringesreal = fringesreal[:,:,0:4]
# # plt.imshow(20*np.log10(abs(tom[:,:,0])),cmap='gray')
# #%%
# nSnapshots = fs
# alpha = np.linspace( 0., 2.,nSnapshots)
# obj_1d_shifted_gpu = torch.from_numpy(fringescc[:,512,0]).cuda()
# results = []
# gputime = []
# for al in tqdm( alpha, total=alpha.size ):
#     start = time.time()
#     fobj_1d = frft_g.frft( obj_1d_shifted_gpu, al )
#     results.append( fftshift(torch.Tensor.numpy(torch.Tensor.cpu(fobj_1d))))
#     t_gpu = time.time() - start
#     gputime.append( t_gpu*1.e6 )
# print( 'Mean GPU time = %f μs'%mean(gputime))
# z = 1700
# frftarray = np.array(results)
# phasefrft = np.angle(frftarray)
# fig,ax = plt.subplots(4,1)
# ax[0].plot(abs(tom[:,512,0]))
# ax[1].plot(abs(fftshift(frftarray[n,:])))
# ax[2].plot(abs(fftshift(frftarray[:,z])))
# ax[3].plot(abs(np.unwrap(phasefrft[n,:])))