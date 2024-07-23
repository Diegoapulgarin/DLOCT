#%%
import sys
sys.path.append(r'C:\Users\USER\Documents\GitHub\DLOCT\Analysis_cGAN')
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift, ifft
import sys
from tqdm import tqdm # for progress bars
from statistics import mean, stdev
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
from Deep_Utils import dbscale
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input

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
        fringescc = fftshift(ifft(tomcc,axis=0),axes=0)
        fringescc = np.stack((fringescc.real,fringescc.imag),axis=3)
        del tomImag, tomReal
# fringescc = fftshift(ifft(fftshift(tom,axes=0),axis=0),axes=0)
# fringescc = fftshift(ifft(tomcc,axis=0),axes=0)

pathcomplex = r'C:\Users\USER\Documents\GitHub\[s.fovea]11bscanNoartifacts'
artifact_files = os.listdir(pathcomplex)
for imag_file, real_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(pathcomplex, real_file)
        imag_file_path = os.path.join(pathcomplex, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        fringes = fftshift(ifft(tom,axis=0),axes=0)
        fringes = np.stack((fringes.real,fringes.imag),axis=3)
        del tomImag, tomReal
# fringescc = fftshift(ifft(fftshift(tom,axes=0),axis=0),axes=0)
# fringes = fftshift(ifft(tom,axis=0),axes=0)

fringesccHilbert = hilbert(fringescc[:,:,:,0],axis=0)
fringesccAnalytic = fringescc[:,:,:,0] + 1j*fringesccHilbert
fringesccAnalytic = np.stack((fringesccAnalytic.real,fringesccAnalytic.imag),axis=3)
del fringesccHilbert
#%%
num_samples, num_alines, num_bscans, num_channels = fringesccAnalytic.shape

# Initialize arrays for normalized data
X_train_complete = np.zeros_like(fringesccAnalytic)
y_train_complete = np.zeros_like(fringes)

# Initialize arrays for storing energies
energies_X = np.zeros((num_alines, num_bscans, num_channels))
energies_y = np.zeros((num_alines, num_bscans, num_channels))

# Function to normalize by energy along the z-axis (axis=0)
def normalize_by_energy(signal):
    energy = np.sum(np.square(signal), axis=0, keepdims=True)
    normalized_signal = signal / np.sqrt(energy)
    return normalized_signal, np.sqrt(energy)

# Normalize each aline (z) individually for both input and output
for bscan in tqdm(range(num_bscans)):
    for aline in range(num_alines):
        for channel in range(num_channels):
            X_train_complete[:, aline, bscan, channel], energies_X[aline, bscan, channel] = normalize_by_energy(fringesccAnalytic[:, aline, bscan, channel])
            y_train_complete[:, aline, bscan, channel], energies_y[aline, bscan, channel] = normalize_by_energy(fringes[:, aline, bscan, channel])


# Verificar las formas de los datos
print("X_train shape:", X_train_complete.shape)
print("y_train shape:", y_train_complete.shape)
print("energies_X shape:", energies_X.shape)
print("energies_y shape:", energies_y.shape)

#%%
X_train_partial = X_train_complete[:, :, :-1, :]  # Excluyendo el último B-scan
y_train_partial = y_train_complete[:, :, :-1, :]  # Excluyendo el último B-scan

# Reorganizar los datos en la forma adecuada
X_list = []
y_list = []

for bscan in tqdm(range(X_train_partial.shape[2])):  # Iterar sobre B-scans
    for aline in range(X_train_partial.shape[1]):  # Iterar sobre alines
        X_list.append(X_train_partial[:, aline, bscan, :])
        y_list.append(y_train_partial[:, aline, bscan, :])

# Convertir listas a arrays numpy
X = np.array(X_list)  # Input: (n_samples, 2304, 2)
y = np.array(y_list)  # Output: (n_samples, 2304, 2)

# Dividir en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
def unwrap_phase(phase):
    phase_unwrapped = np.unwrap(phase)
    return phase_unwrapped

def magnitude_phase_loss(y_true, y_pred):
    # Calcular la magnitud
    mag_true = tf.sqrt(tf.square(y_true[..., 0]) + tf.square(y_true[..., 1]))
    mag_pred = tf.sqrt(tf.square(y_pred[..., 0]) + tf.square(y_pred[..., 1]))

    # Calcular la fase
    phase_true = tf.math.angle(tf.complex(y_true[..., 0], y_true[..., 1]))
    phase_pred = tf.math.angle(tf.complex(y_pred[..., 0], y_pred[..., 1]))

    # Envolver la fase utilizando tf.py_function para ejecutar np.unwrap
    phase_true_unwrapped = tf.py_function(func=unwrap_phase, inp=[phase_true], Tout=tf.float32)
    phase_pred_unwrapped = tf.py_function(func=unwrap_phase, inp=[phase_pred], Tout=tf.float32)

    # Normalizar la fase desenvuelta
    phase_true_unwrapped = (phase_true_unwrapped - tf.reduce_min(phase_true_unwrapped)) / (tf.reduce_max(phase_true_unwrapped) - tf.reduce_min(phase_true_unwrapped))
    phase_pred_unwrapped = (phase_pred_unwrapped - tf.reduce_min(phase_pred_unwrapped)) / (tf.reduce_max(phase_pred_unwrapped) - tf.reduce_min(phase_pred_unwrapped))

    # Calcular la diferencia en magnitud y fase
    mag_diff = tf.reduce_mean(tf.square(mag_true - mag_pred))
    phase_diff = tf.reduce_mean(tf.square(phase_true_unwrapped - phase_pred_unwrapped))

    return mag_diff + phase_diff

def fft_loss(y_true, y_pred):
    # Transformar las señales en el dominio de la frecuencia
    fft_true = tf.signal.fft(tf.complex(y_true[..., 0], y_true[..., 1]))
    fft_pred = tf.signal.fft(tf.complex(y_pred[..., 0], y_pred[..., 1]))

    # Calcular la magnitud de las FFTs
    mag_fft_true = tf.math.abs(fft_true)
    mag_fft_pred = tf.math.abs(fft_pred)

    # Calcular la diferencia entre las magnitudes de las FFTs
    fft_diff = tf.reduce_mean(tf.square(mag_fft_true - mag_fft_pred))

    return fft_diff

def combined_loss(y_true, y_pred, mag_phase_weight=0.5, fft_weight=0.5):
    mag_phase_loss_value = magnitude_phase_loss(y_true, y_pred)
    fft_loss_value = fft_loss(y_true, y_pred)
    
    return mag_phase_weight * mag_phase_loss_value + fft_weight * fft_loss_value


# Definir el modelo RNN con LSTM
model = Sequential([
    Input(shape=(2304, 2)),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='linear')  # Ensure output has 2 channels
])

# Compilar el modelo con la nueva función de pérdida combinada
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, mag_phase_weight=0.5, fft_weight=0.5))

# Verificar el resumen del modelo
model.summary()

#%%
# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,  # Ajusta el tamaño del lote según la capacidad de tu GPU
    verbose=1
)
# Evaluar el modelo en el conjunto de validación
val_loss = model.evaluate(X_val, y_val, verbose=1)
print("Validation Loss:", val_loss)
#%%
# Seleccionar la última B-scan para la prueba
X_test_partial = X_train_complete[:, :, -1, :]  # Input: (2304, 1024, 2)
y_test_partial = y_train_complete[:, :, -1, :]  # Output: (2304, 1024, 2)

# Reorganizar los datos en la forma adecuada
X_test_list = []
y_test_list = []

for aline in range(X_test_partial.shape[1]):  # Iterar sobre alines
    X_test_list.append(X_test_partial[:, aline, :])
    y_test_list.append(y_test_partial[:, aline, :])

# Convertir listas a arrays numpy
X_test = np.array(X_test_list)  # Input: (n_samples, 2304, 2)
y_test = np.array(y_test_list)  # Output: (n_samples, 2304, 2)

# Verificar las formas de los datos de prueba
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# Hacer predicciones
predictions = model.predict(X_test)

# Reorganizar las predicciones para que coincidan con las dimensiones originales
predictions_reorganized = np.zeros((2304, X_test.shape[0], 2))
for i in range(X_test.shape[0]):
    predictions_reorganized[:, i, :] = predictions[i]

# Verificar las formas de las predicciones
print("Predictions shape:", predictions_reorganized.shape)

# Comparar una aline específica (por ejemplo, la aline 0) entre las predicciones y los valores reales
aline_index = 0


plt.figure(figsize=(14, 14))

# Parte real
plt.subplot(4, 1, 1)
plt.plot(predictions_reorganized[:, aline_index, 0], label='Predicted Real')
plt.plot(y_test_partial[:, aline_index, 0], label='True Real')
plt.title('Comparison of Real Part')
plt.legend()

# Diferencia en la parte real
plt.subplot(4, 1, 2)
plt.plot(predictions_reorganized[:, aline_index, 0] - y_test_partial[:, aline_index, 0], label='Difference Real')
plt.title('Difference in Real Part')
plt.legend()

# Parte imaginaria
plt.subplot(4, 1, 3)
plt.plot(predictions_reorganized[:, aline_index, 1], label='Predicted Imaginary')
plt.plot(y_test_partial[:, aline_index, 1], label='True Imaginary')
plt.title('Comparison of Imaginary Part')
plt.legend()

# Diferencia en la parte imaginaria
plt.subplot(4, 1, 4)
plt.plot(predictions_reorganized[:, aline_index, 1] - y_test_partial[:, aline_index, 1], label='Difference Imaginary')
plt.title('Difference in Imaginary Part')
plt.legend()

plt.tight_layout()
plt.show()


#%%

fig, axs = plt.subplots(nrows=2,ncols=1)
axs[0].plot(abs(fft(predictions_reorganized[:,0,0]+1j*predictions_reorganized[:,0,1])))
axs[1].plot(abs(fft(y_test_partial[:,0,0]+1j*y_test_partial[:,0,1])))
fig.show()

fig, axs = plt.subplots(nrows=1,ncols=2)
axs[0].imshow(dbscale(fft(predictions_reorganized[:,:,0]+1j*predictions_reorganized[:,:,1],axis=0)))
axs[1].imshow(dbscale(fft(y_test_partial[:,:,0]+1j*y_test_partial[:,:,1],axis=0)))
fig.show()

#%%
# Guardar el modelo entrenado
model.save('model_rnn_combined_loss.h5')