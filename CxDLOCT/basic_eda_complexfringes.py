#%%
import numpy as np
import os
import scipy.io as sio
from scipy.fft import fft, fftshift
from scipy.ndimage import shift
from numpy.random import randn
import plotly.graph_objs as go
import plotly.subplots as sp
from scipy.signal import hilbert

# %%
def reconstruct_tomogram(fringes1, zeroPadding=0, noiseFloorDb=0,z=2):
    nK = fringes1.shape[0]  # the size along the first dimension
    nZ, nX, nY = fringes1.shape  # fringes1 is 3D
    zRef = nZ / z  # zRef value
    zSize = 256  # zSize value

    # Apply hanning window along the first dimension
    fringes1 = fringes1 * np.hanning(nK)[:, np.newaxis, np.newaxis]

    # Pad the fringes
    fringes1_padded = np.pad(fringes1, ((zeroPadding, zeroPadding), (0, 0), (0, 0)), mode='constant')

    # Fourier Transform
    tom1True = fftshift(fft(fftshift(fringes1_padded, axes=0), axis=0), axes=0)
    tom1 = tom1True + (((10 ** (noiseFloorDb / 20)) / 1) * (randn(nZ, nX, nY) + 1j * randn(nZ, nX, nY)))

    refShift = int((2 * zRef + zSize) / zSize * nZ) // 2
    tom1 = np.roll(tom1, refShift, axis=0)
    tom1True = np.roll(tom1True, refShift, axis=0)
    
    return tom1True, tom1

def plot_images(array1, array2, zmin, zmax,tittle,save=False):
    fig = sp.make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Heatmap(z=np.flipud(array1), zmin=zmin, zmax=zmax, colorscale='Gray'),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(z=np.flipud(array2), zmin=zmin, zmax=zmax, colorscale='Gray'),
        row=1, col=2
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)  
    fig.update_layout(height=600, width=800, title_text=tittle)
    if save:
        fig.write_html(tittle +'.html')
    fig.show()

path = r'C:\Users\diego\Documents\Github\Simulated_Data_Complex'
# path = r'/home/haunted/Projects/DLOCT/CxDLOCT/Simulated_Data_Complex'
os.chdir(path)
fringes = []
for filename in os.listdir(os.getcwd()):
   mat_contents = sio.loadmat(path+'/'+filename)
   fringes1 = mat_contents['fringes1']
   divisions = int(fringes1.shape[2]/16)
   n = 0 
   for i in range(divisions):
       fringes_slice = fringes1[:, :, n:n+16]
       n = n + 16
       fringes.append(fringes_slice)
   print(filename)
fringes = np.array(fringes)
# %%
tom1 = fringes[6,:,:,:]
aline = tom1[:,0,0]
intensity = abs(aline)*np.cos(np.angle(aline))
recsignal = abs(intensity)*np.exp(1j*np.angle(aline))
fft_aline =abs(fftshift(fft(aline)))
fft_real = abs(fftshift(fft(intensity)))
fft_rec = abs(fftshift(fft(recsignal)))
fig = sp.make_subplots(rows=4,cols=1)
fig.add_trace(go.Line(y=aline.real,name='Real part'),row=1,col=1)
fig.add_trace(go.Line(y=intensity,line=dict(dash='dash'),name='intensity'),row=1,col=1)
fig.add_trace(go.Line(y=recsignal.real,line=dict(dash='dot'),name='Real part recovered'),row=1,col=1)
fig.add_trace(go.Line(y=fft_aline,name= 'fft complex fringe'),row=2,col=1)
fig.add_trace(go.Line(y=fft_real,name='fft real fringe'),row=3,col=1)
fig.add_trace(go.Line(y=fft_rec,name='fft recovered signal'),row=4,col=1)
fig.show()
# %%
hil_aline = hilbert(intensity)
recovered2 = abs(intensity)*np.exp(1j*hil_aline.imag)
diff = intensity - recsignal.real
fft_hil = abs(fftshift(fft(recsignal)))
fig = sp.make_subplots(rows=3,cols=1)
fig.add_trace(go.Line(y=diff,name='real signal difference'),row=1,col=1)
fig.add_trace(go.Line(y=recsignal.imag, name='Original phase'),row=2,col=1)
fig.add_trace(go.Line(y=fft_hil,name = 'estimated signal'),row=3,col=1)