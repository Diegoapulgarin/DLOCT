#%%
import numpy as np
import scipy.io as sio
import os
from scipy.fft import fft, fftshift
from numpy.random import randn

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization



#%%
path = r'C:\Users\USER\Documents\GitHub\Simulated_Data_Complex'
os.chdir(path)
fringes = []
for filename in os.listdir(os.getcwd()):
   print(path+'/'+filename)
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
del fringes1, fringes_slice
#%%


def normalize_aline(aline):
    min_val = np.min(aline)
    range_val = np.max(aline) - min_val
    normalized_aline = (aline - min_val) / range_val
    return normalized_aline, min_val, range_val

def normalize_volume_by_aline(volume):
    z, x, y = volume.shape
    normalized_volume = np.zeros_like(volume)
    min_vals = np.zeros((x, y))
    range_vals = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            normalized_volume[:, i, j], min_vals[i, j], range_vals[i, j] = normalize_aline(volume[:, i, j])
            
    return normalized_volume, min_vals, range_vals


def inverse_normalize_aline(normalized_aline, min_val, range_val):
    original_aline = normalized_aline * range_val + min_val
    return original_aline

def inverse_normalize_volume_by_aline(normalized_volume, min_vals, range_vals):
    z, x, y = normalized_volume.shape
    original_volume = np.zeros_like(normalized_volume)
    
    for i in range(x):
        for j in range(y):
            original_volume[:, i, j] = inverse_normalize_aline(normalized_volume[:, i, j], min_vals[i, j], range_vals[i, j])
            
    return original_volume


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


def paired_random_zero_padding(tomogram, target, target_z_size=1024):
    assert tomogram.shape == target.shape, "Ambos volúmenes deben tener la misma forma."

    z, x, y = tomogram.shape
    
    # Si el volumen ya es del tamaño deseado o mayor, lo truncamos.
    if z >= target_z_size:
        return tomogram[:target_z_size, :, :], target[:target_z_size, :, :]

    padding_size = target_z_size - z

    # Decidimos aleatoriamente si el padding va al principio, al final, o se divide.
    decision = np.random.choice(["start", "end", "both"])
    
    if decision == "start":
        start_padding = padding_size
        end_padding = 0
    elif decision == "end":
        start_padding = 0
        end_padding = padding_size
    else: # decision == "both"
        start_padding = np.random.randint(0, padding_size + 1)
        end_padding = padding_size - start_padding
    
    # Aplicamos el padding a ambos volúmenes.
    padded_tomogram = np.pad(tomogram, ((start_padding, end_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    padded_target = np.pad(target, ((start_padding, end_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)

    return padded_tomogram, padded_target


def consistent_zero_padding(volume, target_z_size, start_fraction=0.5):
    z, x, y = volume.shape
    
    if z >= target_z_size:
        return volume[:target_z_size, :, :]
    
    padding_size = target_z_size - z
    start_padding = int(padding_size * start_fraction)
    end_padding = padding_size - start_padding
    
    padded_volume = np.pad(volume, ((start_padding, end_padding), (0, 0), (0, 0)), mode='constant', constant_values=0)
    return padded_volume




#%%
normalized_volume_complex = np.zeros(np.shape(fringes))
min_vals_list =[]
range_vals_list =[]
for i in range(np.shape(fringes)[0]):
    fftfringes,_ = reconstruct_tomogram(fringes[i,:,:,:],z=2)
    normalized_volume_complex[i,:,:,:], min_vals, range_vals = normalize_volume_by_aline(abs(fftfringes))
    min_vals_list.append(min_vals)
    range_vals_list.append(range_vals)

normalized_volume_real = np.zeros(np.shape(fringes))
min_vals_list =[]
range_vals_list =[]
for i in range(np.shape(fringes)[0]):
    fftfringes,_ = reconstruct_tomogram(np.real(fringes[i,:,:,:]),z=2)
    normalized_volume_real[i,:,:,:], min_vals, range_vals = normalize_volume_by_aline(abs(fftfringes))
    min_vals_list.append(min_vals)
    range_vals_list.append(range_vals)

#%%
ntom=np.shape(normalized_volume_complex)[0]
zsize=1024
xsize=np.shape(normalized_volume_complex)[2]
ysize=np.shape(normalized_volume_complex)[3]
padded_tomogram = np.zeros((ntom,zsize,xsize,ysize))
padded_target = np.zeros((ntom,zsize,xsize,ysize))
for i in range(np.shape(padded_target)[0]):
    padded_tomogram[i,:,:,:], padded_target[i,:,:,:] = paired_random_zero_padding(normalized_volume_real[i,:,:,:], 
                                                                                  normalized_volume_complex[i,:,:,:], target_z_size=zsize)

#%%
import matplotlib.pyplot as plt
plt.figure()
plt.plot(padded_tomogram[6,:, 1, 1], label="Real signal fft")
plt.plot(padded_target[6,:,1,1], label="complex signal fft")
plt.legend()
plt.show()
#%%
t =6
fig,axs = plt.subplots(1,2)
axs[0].imshow(padded_tomogram[t,:, :, 1])
axs[1].imshow(padded_target[t,:, :, 1])
#%%
padded_tomogram_train = padded_tomogram[:-1]
padded_target_train = padded_target[:-1]

padded_tomogram_test = padded_tomogram[-1]
padded_target_test = padded_target[-1]


n, z, x, y = padded_tomogram_train.shape

X = padded_tomogram_train.reshape(n*x*y, z)
y = padded_target_train.reshape(n*x*y, z)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Funciones de la cGAN

def build_generator():
    input_data = Input(shape=(1124,))  # Concatenación de ruido (100) + target (1024)
    
    h = Dense(128)(input_data)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dense(256)(h)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dense(512)(h)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dense(1024, activation='tanh')(h)
    
    model = Model(input_data, h)
    
    return model


def build_discriminator(input_shape):
    input_data = Input(shape=input_shape)
    input_label = Input(shape=(1024,))
    
    merged = Concatenate()([input_data, input_label])
    
    x = Dense(512, activation='relu')(merged)
    x = Dense(256, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model([input_data, input_label], out)
    return model

def build_cgan(generator, discriminator):
    # El generador tomará ruido y la etiqueta como entrada y generará la aline correspondiente
    noise = Input(shape=(100,))
    label = Input(shape=(1024,))
    
    # Concatena ruido y label para pasar al generador
    combined_input = Concatenate()([noise, label])
    
    generated_data = generator(combined_input)
    
    # El discriminador tomará la aline generada y la etiqueta y determinará su autenticidad
    discriminator.trainable = False
    valid = discriminator([generated_data, label])
    
    model = Model([noise, label], valid)
    
    return model

def train(generator, discriminator, cgan, data, labels, epochs, batch_size=128, save_interval=50):
    # Dimensiones del ruido de entrada
    noise_dim = 100
    
    # Etiquetas para datos reales y generados
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        
        # ---------------------
        #  Entrenar Discriminador
        # ---------------------
        
        # Seleccionar un batch aleatorio de datos
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        real_labels = labels[idx]
        
        # Generar un batch de nuevos datos a través del generador
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        generated_data = generator.predict([noise, real_labels])
        
        # Entrenar el discriminador
        d_loss_real = discriminator.train_on_batch([real_data, real_labels], valid)
        d_loss_fake = discriminator.train_on_batch([generated_data, real_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Entrenar Generador
        # ---------------------
        
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = cgan.train_on_batch([noise, real_labels], valid)
        
        # Imprimir el progreso
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
        
        # Guardar imágenes generadas en intervalos regulares (puedes ajustar esto según tus necesidades)
        if epoch % save_interval == 0:
            save_generated_data(epoch, generator)

def save_generated_data(epoch, generator, num_samples=10):
    noise = np.random.normal(0, 1, (num_samples, 100))
    labels_sample = np.random.rand(num_samples, 1024)  # Puedes ajustar cómo seleccionas estas etiquetas
    generated_data = generator.predict([noise, labels_sample])
    
    # Aquí puedes guardar o visualizar las señales generadas
    # Por ejemplo, usando matplotlib:
    import matplotlib.pyplot as plt
    for i in range(num_samples):
        plt.figure(figsize=(10, 4))
        plt.plot(generated_data[i])
        plt.title(f"Generated signal at epoch {epoch}")
        plt.savefig(f"generated_{epoch}_{i}.png")
        plt.close()
#%%

aline_dim = 1024
# Construir el generador
generator = build_generator()

# Construir el discriminador
discriminator = build_discriminator(aline_dim)
# Compilar el discriminador
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Construir el modelo combinado cGAN
cgan = build_cgan(generator, discriminator)
# Compilar la cGAN
cgan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Mostrar un resumen de cada modelo
generator.summary()
discriminator.summary()
cgan.summary()


#%%

train(generator, discriminator, cgan, padded_tomogram, padded_target, epochs=10000, batch_size=32)

