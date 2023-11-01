#%%
import numpy as np
import scipy.io as sio
import os
from datetime import datetime
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

padded_tomogram_train = np.transpose(padded_tomogram_train,(0,2,3,1))
padded_target_train = np.transpose(padded_target_train,(0,2,3,1))

padded_tomogram_test = padded_tomogram[-1]
padded_target_test = padded_target[-1]


n, x, y, z = padded_tomogram_train.shape

X = np.reshape(padded_tomogram_train,(n*x*y,z))
Y = np.reshape(padded_target_train,(n*x*y,z))


X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#%% Funciones de la cGAN

def build_generator():
    input_data = Input(shape=(1024,))  # Concatenación de ruido (100) + target (1024)
    
    h = Dense(128)(input_data)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dense(256)(h)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dense(512)(h)
    h = LeakyReLU(0.2)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Dense(1024, activation='sigmoid')(h)
    
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
    # El generador tomará solo la etiqueta como entrada y generará la aline correspondiente
    label = Input(shape=(1024,))
    generated_data = generator(label)
    
    # El discriminador tomará la aline generada y la etiqueta y determinará su autenticidad
    discriminator.trainable = False
    valid = discriminator([generated_data, label])
    
    model = Model(label, valid)
    
    return model

def train(generator, discriminator, cgan, data, labels, epochs, batch_size=128, save_interval=100):    
    # Etiquetas para datos reales y generados
    start_time = datetime.now()
    folder_name = start_time.strftime('%d_%m_%y_%H_%M')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
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
        generated_data = generator.predict(real_labels)
        
        # Entrenar el discriminador
        d_loss_real = discriminator.train_on_batch([real_data, real_labels], valid)
        d_loss_fake = discriminator.train_on_batch([generated_data, real_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Entrenar Generador
        # ---------------------
        
        g_loss = cgan.train_on_batch(real_labels, valid)
        
        # Imprimir el progreso
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
        
        # Guardar imágenes generadas en intervalos regulares (puedes ajustar esto según tus necesidades)
        if epoch % save_interval == 0:
            save_generated_data(epoch, generator, labels, y_train, folder_name)
            generator.save(os.path.join(folder_name, f"generator_epoch_{epoch}.h5"))

def save_generated_data(epoch, generator, inputs, targets,folder_name, num_samples=1):
    # Tomamos muestras aleatorias del input y target proporcionado
    idx = np.random.randint(0, inputs.shape[0], num_samples)
    input_samples = inputs[idx]
    target_samples = targets[idx]
    
    # Usamos el generador para obtener la salida basada en el input
    generated_data = generator.predict(input_samples)
    
    import matplotlib.pyplot as plt
    for i in range(num_samples):
        fig, axs = plt.subplots(1, 3, figsize=(25, 5))
        
        # Plot del input
        axs[0].plot(input_samples[i])
        axs[0].set_title("Input signal (with mirror artifact)")
        
        # Plot de la señal generada
        axs[1].plot(generated_data[i])
        axs[1].set_title(f"Generated signal at epoch {epoch}")
        
        # Plot del target
        axs[2].plot(target_samples[i])
        axs[2].set_title("Target signal (without mirror artifact)")
        
        plt.tight_layout()
        plt.savefig(f"{folder_name}/epoch_{epoch}_sample_{i}.png")
        plt.close()



#%%

aline_dim = 1024

generator = build_generator()


discriminator = build_discriminator(aline_dim)

discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])


cgan = build_cgan(generator, discriminator)

cgan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')


generator.summary()
discriminator.summary()
cgan.summary()


#%%

train(generator, discriminator, cgan, X_train, y_train, epochs=2000, batch_size=32)

#%%
a = 5
fig,axs = plt.subplots(1,2)
axs[0].plot(padded_tomogram_train[a,:,100,1])
axs[1].plot(padded_target_train[a,:,100,1])

#%%

a = 11970
fig,axs = plt.subplots(1,2)
axs[0].plot(X[a,:])
axs[1].plot(Y[a,:])
# axs[2].plot(X[a,:]-y[a,:])