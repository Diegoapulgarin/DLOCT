#%%
import os
import numpy as np
import struct
import matplotlib.pyplot as plt 
from numpy.fft import fft, fftshift, ifft
#%%
# Configura los parámetros y las rutas de los archivos
path = r'D:\DLOCT\TomogramsDataAcquisition\[DepthWrap]'
dataFileName = '[DepthWrap][ChickenBreast][09-18-2023_11-43-40]'
dir_name = os.path.join(path,dataFileName)
roi = [1, 10]
n_volumes = 1  # Valor predeterminado si no se define en el archivo de información
n_ch = 2  # Número de canales, si es conocido

# Encuentra los archivos relevantes en el directorio
for file in os.listdir(dir_name):
    if file.endswith('.dat'):
        config_name = file
    elif file.endswith('_info.txt'):
        info_name = file
    elif 'BG' in file:
        bgr_name = file
    elif file.endswith('.ofd') and 'FULLBG' not in file:
        data_name = file

# Manejo del archivo de información
info_path = os.path.join(dir_name, info_name)
with open(info_path, 'r') as f:
    lines = f.readlines()

# Extrae los parámetros numéricos, ignorando las líneas que contienen texto
scan_parms = []
for line in lines:
    try:
        # Intenta convertir la línea en un número flotante y agregarlo a la lista de parámetros
        scan_parms.append(float(line.strip()))
    except ValueError:
        # Si falla, continúa con la siguiente línea
        continue

# Asegúrate de que encontramos los parámetros requeridos
if len(scan_parms) < 3:
    raise ValueError('Not enough scan parameters found in the info file.')

n_samples, n_alines, n_bscans = scan_parms[:3]
n_volumes = int(scan_parms[3]) if len(scan_parms) > 3 else n_volumes
roi[1] = roi[1] if roi[1] is not None else n_bscans
n_bscans_to_read = roi[1] - roi[0] + 1

# Asumiendo que tienes otras configuraciones o parámetros basados en tu código MATLAB,
# como 'specTrim', 'cx', 'windowCh1', 'windowCh2', 'doOptimFilt', etc.
# ... (Agregar código para manejar estos parámetros)

# Lectura del archivo de datos .ofd
data_path = os.path.join(dir_name, data_name)
with open(data_path, 'rb') as file:
    # Calcula la cantidad de datos a leer por B-scan
    data_per_bscan = n_samples * n_alines * n_ch
    # Calcula el total de datos a leer según el ROI
    total_data_to_read = data_per_bscan * (roi[1] - roi[0] + 1)
    
    # Crea un array para almacenar los datos del ROI
    fringes = np.zeros((int(n_samples), int(n_alines), roi[1] - roi[0] + 1, n_ch), dtype=np.single)
    
    # Lee los datos del archivo .ofd
    for bscan_index in range(roi[0] - 1, roi[1]):
        # Calcula el desplazamiento al inicio del B-scan actual
        file_offset = 4 * (bscan_index * n_samples * n_alines)
        file.seek(int(file_offset))
        # Lee los datos del B-scan actual
        this_fringe = np.fromfile(file, dtype=np.uint16, count=int(data_per_bscan))
        # Reshape y almacenar en el array 'fringes'
        fringes[:, :, bscan_index - (roi[0] - 1), :] = this_fringe.reshape((int(n_samples), int(n_alines), n_ch), order='F')
#%%
# tom = fftshift(fft(fftshift(fringes, axes=0), axis=0), axes=0)     
#%%
# plt.imshow(10*np.log10(abs(tom[:,:,0,0])**2))

