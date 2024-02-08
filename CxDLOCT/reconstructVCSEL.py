#%%
import os
import numpy as np
import struct
import matplotlib.pyplot as plt 
from numpy.fft import fft, fftshift, ifft
import numpy as np
from scipy.fftpack import fft, ifft
from numpy.linalg import norm
# from scipy.signal import hanning, csaps, interp1d
#%%
# Configura los parámetros y las rutas de los archivos
path = r'C:\Users\USER\OneDrive - Universidad EAFIT\Eafit\Trabajo de grado\Data Boston\[DepthWrap]'
# dataFileName = '[DepthWrap][ChickenBreastWrap][09-18-2023_11-30-40]'
dataFileName = '[DepthWrap][ChickenBreast][09-18-2023_11-29-20]'
dir_name = os.path.join(path,dataFileName)
roi = [1, 5]
n_volumes = 1  # Valor predeterminado si no se define en el archivo de información
n_ch = 2  # Número de canales, si es conocido
# Definición del diccionario de parámetros 'parms' (equivalente a 'varargin' en MATLAB)
parms = {
    'useComplexFringes': True,
    'backgroundType': 'new'  # Incluimos este campo por defecto
}
# Luego sigue la lógica para establecer 'old_background' y 'calc_noise_floor'
old_background = parms.get('backgroundType', 'new') == 'old'
calc_noise_floor = False
if 'window' not in parms and 'windowCh1' not in parms and 'windowCh2' not in parms:
    window_ch1 = None
    window_ch2 = None
    do_optim_filt = False
elif parms.get('window', '').lower() == 'optimum':
    window_ch1 = None
    window_ch2 = None
    do_optim_filt = True
elif 'window' in parms and 'windowCh1' not in parms and 'windowCh2' not in parms:
    window_ch1 = parms['window']
    window_ch2 = parms['window']
    do_optim_filt = False
elif 'windowCh1' in parms and 'windowCh2' in parms:
    window_ch1 = parms['windowCh1']
    window_ch2 = parms['windowCh2']
    do_optim_filt = False
else:
    raise ValueError("Can't understand window requested")
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
n_samples = int(n_samples)
n_alines = int(n_alines)
n_bscans = int(n_bscans)
n_volumes = int(scan_parms[3]) if len(scan_parms) > 3 else n_volumes
roi[1] = roi[1] if roi[1] is not None else n_bscans
n_bscans_to_read = roi[1] - roi[0] + 1
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
# Detectar Alines saturados
saturated_alines = np.any((fringes <= 32) | (fringes >= (2 ** 16 - 32)), axis=(0, 1))
# Leer el archivo de fondo
bgr_name = os.path.join(dir_name, bgr_name)
bgr = np.zeros_like(fringes)  # Inicializar array de fondo
noise_floor = np.zeros_like(fringes)  # Inicializar array de piso de ruido, si es necesario
with open(bgr_name, 'rb') as file:
    if old_background:
        temp_bgr = np.fromfile(file, dtype=np.uint16, count=n_samples * n_alines * n_ch)
        temp_bgr1 = temp_bgr[:n_samples * n_alines].reshape(n_alines, n_samples).T
        temp_bgr2 = temp_bgr[n_samples * n_alines:].reshape(n_alines, n_samples).T
        bgr[:, :, :, 1] = np.tile(np.mean(temp_bgr1, axis=1, keepdims=True), (1, 1, fringes.shape[2]))
        bgr[:, :, :, 0] = np.tile(np.mean(temp_bgr2, axis=1, keepdims=True), (1, 1, fringes.shape[2]))
        if calc_noise_floor:
            noise_floor[:, :, :, 1] = temp_bgr1[:, :, np.newaxis]
            noise_floor[:, :, :, 0] = temp_bgr2[:, :, np.newaxis]
    else:
        n_alines_bgr = 512  # Valor fijo
        temp_bgr = np.fromfile(file, dtype=np.uint16, count=n_samples * n_alines_bgr * n_ch)
        bgr[:, :, :, 1] = np.tile(np.mean(temp_bgr[1::2].reshape(n_samples, n_alines_bgr, 1), axis=1, keepdims=True), (1, 1, fringes.shape[2]))
        bgr[:, :, :, 0] = np.tile(np.mean(temp_bgr[::2].reshape(n_samples, n_alines_bgr, 1), axis=1, keepdims=True), (1, 1, fringes.shape[2]))
        if calc_noise_floor:
            noise_floor[:, :, :, 1] = temp_bgr[1::2].reshape(n_samples, n_alines_bgr, 1)
            noise_floor[:, :, :, 0] = temp_bgr[::2].reshape(n_samples, n_alines_bgr, 1)
# Sustracción del fondo
fringes -= bgr
if calc_noise_floor:
    noise_floor_no_bgr = noise_floor - bgr
if window_ch1 is None or window_ch2 is None:
    window_ch1 = np.hanning(n_samples)
    window_ch2 = np.hanning(n_samples)
# Realizar la Transformada de Fourier
pre_tom = np.fft.fft(fringes, axis=0)
# Si se deben calcular los datos del piso de ruido
if calc_noise_floor:
    pre_noise_floor = np.fft.fft(noise_floor_no_bgr, axis=0)
# Si se están utilizando franjas complejas, realiza el recorte y el desplazamiento
if parms['useComplexFringes']:
    # Recorta la mitad del espectro
    pre_tom = pre_tom[:n_samples // 2]
    # Realiza un desplazamiento circular
    pre_tom = np.roll(pre_tom, n_samples // 4, axis=0)
    # Convierte los datos de vuelta al dominio del tiempo o del espacio
    fringes = np.fft.ifft(pre_tom, axis=0)
    # Si se deben calcular los datos del piso de ruido
    if calc_noise_floor:
        pre_noise_floor = pre_noise_floor[:n_samples // 2]
        pre_noise_floor = np.roll(pre_noise_floor, n_samples // 4, axis=0)
        noise_floor_no_bgr = np.fft.ifft(pre_noise_floor, axis=0)
#%%
# import numpy as np
# from numpy.lib import pad
# window = np.stack((window_ch1, window_ch2), axis=3)
# # Aplicación de la ventana
# fringes *= window
# if calc_noise_floor:
#     noise_floor_no_bgr *= window

# # Si se realiza el recorte espectral
# if spec_trim:
#     # Encuentra el inicio del espectro donde la ventana no es cero
#     spec_start = np.argmax(window_ch1 != 0, axis=0)
#     fringes_trimmed = np.zeros((spec_trimmed_size, *fringes.shape[1:]), dtype=fringes.dtype)
#     for this_window in range(window_ch1.shape[4]):
#         start_index = spec_start[this_window]
#         # Asegúrate de no salirte de los límites del array
#         end_index = min(fringes.shape[0], start_index + spec_trimmed_size)
#         fringes_trimmed[:, :, :, :, this_window] = fringes[start_index:end_index, :, :, :, this_window]
#     fringes = fringes_trimmed
#     # Realiza el relleno con ceros si se solicita
#     padding_amount = (nZ - spec_trimmed_size) // 2
#     fringes = np.pad(fringes, ((padding_amount, padding_amount), (0, 0), (0, 0), (0, 0)), 'constant')
# else:
#     # Realiza el relleno con ceros si se solicita
#     padding_amount = (nZ - fringes.shape[0]) // 2
#     fringes = np.pad(fringes, ((padding_amount, padding_amount), (0, 0), (0, 0), (0, 0)), 'constant')

# if calc_noise_floor:
#     noise_floor_no_bgr = np.pad(noise_floor_no_bgr, ((padding_amount, padding_amount), (0, 0), (0, 0), (0, 0)), 'constant')
# # Combina las dos ventanas
# window = np.stack((window_ch1, window_ch2), axis=3)
tom = fftshift(fft(fftshift(fringes, axes=0), axis=0), axes=0)     
plt.imshow(10*np.log10(abs(tom[:,:,0,0])**2))

#%%
def dispersion_compensation(volume, k_vector, dispersion_coefficients):
    """
    Applies dispersion compensation to an OCT volume data.
    Parameters:
    - volume: numpy array with shape (z, x, y, channels), OCT data.
    - k_vector: numpy array, linearized wave number vector.
    - dispersion_coefficients: dict, dispersion coefficients of the system.
    Returns:
    - compensated_volume: numpy array, volume with dispersion compensated.
    """
    # Apply FFT along the z-axis
    volume_f = fft(volume, axis=0)
    # Calculate the compensation phase
    compensation_phase = np.zeros_like(k_vector)
    for order, coefficient in dispersion_coefficients.items():
        compensation_phase += coefficient * k_vector**order
    # Adjust the shape of compensation_phase to match volume_f for broadcasting
    compensation_phase = np.exp(-1j * compensation_phase)
    compensation_phase = compensation_phase.reshape(-1, 1, 1, 1)
    # Check if compensation_phase length matches the z-axis of volume_f
    if compensation_phase.shape[0] != volume_f.shape[0]:
        raise ValueError("The length of compensation_phase does not match the z-axis of the volume_f. Adjustment required.")
    # Apply the phase compensation
    compensated_volume_f = volume_f * compensation_phase
    # Apply IFFT to return to the time domain
    compensated_volume = ifft(compensated_volume_f, axis=0)
    return compensated_volume
coeficientes_dispersion = {
    2: -1.0e-4,  # Coeficiente para el término cuadrático
    3: 3.0e-6,   # Coeficiente para el término cúbico
}
lambda_c = 1300e-9  # Longitud de onda central en metros
delta_lambda_sweep = 95e-9  # Rango de barrido en metros

# Calcular longitudes de onda mínima y máxima
lambda_min = lambda_c - (delta_lambda_sweep / 2)
lambda_max = lambda_c + (delta_lambda_sweep / 2)
z_dim = fringes.shape[0] 
# Convertir longitudes de onda a números de onda (k_min y k_max)
k_min = 2 * np.pi / lambda_max  # Usamos lambda_max para k_min debido a la inversión de la relación
k_max = 2 * np.pi / lambda_min  # Usamos lambda_min para k_max

# Generar el vector k
indices = np.arange(1, 1281, 2)  # Basado en el mapeo proporcionado
k = np.linspace(k_min, k_max, z_dim)

compensedFringes =dispersion_compensation(fringes,k,coeficientes_dispersion)
# Usar estos coeficientes en el algoritmo DEFR
factor = 1
# Modificar los coeficientes de dispersión para exagerar el ajuste
coeficientes_dispersion_adjusted = {order: coef * factor for order, coef in coeficientes_dispersion.items()}
