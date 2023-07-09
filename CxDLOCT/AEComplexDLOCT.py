#%%
import numpy as np 
import plotly.express as px
import os
import scipy.io as sio
from scipy.fft import fft, fftshift
from scipy.ndimage import shift
from numpy.random import randn
import plotly.graph_objs as go
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#%%
path = r'C:\Users\diego\Documents\Github\Simulated_Data_Complex'
os.chdir(path)
fringes = []
for filename in os.listdir(os.getcwd()):
   mat_contents = sio.loadmat(path+'\\'+filename)
   fringes1 = mat_contents['fringes1']
   fringes.append(fringes1)
   print(filename)
fringes = np.array(fringes)
# %%
testfringes = fringes[6,:,:,:]
trainfringes = fringes[0:6,:,:,:]
#%%
thisfringes = 5
# Assuming fringes1 is a 3D numpy array
fringes1 = fringes[thisfringes,:,:,:]
nK = fringes1.shape[0]  # the size along the first dimension
zeroPadding = 0 
noiseFloorDb = 0  # noise floor in dB
nZ, nX, nY = fringes1.shape  # fringes1 is 3D
zRef = nZ / 2  # zRef value
zSize = 256  # zSize value
# Apply hanning window along the first dimension
fringes1 = fringes1 * np.hanning(nK)[:, np.newaxis, np.newaxis]

fringes1_padded = np.pad(fringes1, ((zeroPadding, zeroPadding), (0, 0), (0, 0)), mode='constant')

# Fourier Transform
tom1True = fftshift(fft(fftshift(fringes1_padded, axes=0), axis=0), axes=0)
tom1 = tom1True + (((10 ** (noiseFloorDb / 20)) / 1) * (randn(nZ, nX, nY) + 1j * randn(nZ, nX, nY)))

refShift = int((2 * zRef + zSize) / zSize * nZ) // 2
tom1 = np.roll(tom1, refShift, axis=0)
tom1True = np.roll(tom1True, refShift, axis=0)

plot = 10*np.log10(abs(tom1[:,:,0])**2)
fig = px.imshow(plot,color_continuous_scale='gray',zmin=50,zmax=150)
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(fringes1[:,1,1]), mode='lines', name='Real Part'))
fig.add_trace(go.Scatter(y=np.imag(fringes1[:,1,1]), mode='lines', name='Imaginary Part'))
fig.show()
#%%
fringes_transpose = np.transpose(trainfringes,axes=[1,2,3,0])
# fringes_transpose = fringes1 * np.hanning(nK)[:, np.newaxis, np.newaxis,np.newaxis]
#%%
real_fringes = np.real(fringes_transpose)
hilbert_fringes = hilbert(real_fringes,axis=0)
#%%
thisfringes=5
fig = go.Figure()
fig.add_trace(go.Scatter(y=real_fringes[:,0,0,thisfringes], mode='lines', name='Real Part'))
fig.add_trace(go.Scatter(y=np.imag(fringes_transpose[:,0,0,thisfringes]), mode='lines', name='Imaginary Part'))
fig.add_trace(go.Scatter(y=np.imag(hilbert_fringes[:,0,0,thisfringes]), mode='lines', name='Hilbert transform'))
fig.show()
#%%
scaler = StandardScaler()
aline = 4096*5
xdata = np.stack((real_fringes, hilbert_fringes.imag), axis=-1)
xdata = xdata.reshape(256, -1, 2)
xdata = np.transpose(xdata,axes=[1,0,2])


# Reshape data to 2D
num_samples, num_alines, num_channels = xdata.shape
xdata_reshaped = xdata.reshape(num_samples * num_alines, num_channels)
# Apply StandardScaler
scaler = StandardScaler()
xdata_standardized = scaler.fit_transform(xdata_reshaped)
# Reshape back to original shape
xdata_standardized = xdata_standardized.reshape(num_samples, num_alines, num_channels)


ydata = np.stack((fringes_transpose.real, fringes_transpose.imag), axis=-1)
ydata = ydata.reshape(256, -1, 2)
ydata = np.transpose(ydata,axes=[1,0,2])

# Reshape data to 2D
num_samples, num_alines, num_channels = ydata.shape
ydata_reshaped = xdata.reshape(num_samples * num_alines, num_channels)
# Apply StandardScaler
ydata_standardized = scaler.fit_transform(ydata_reshaped)
# Reshape back to original shape
ydata_standardized = ydata_standardized.reshape(num_samples, num_alines, num_channels)

# ydata_standardized = scaler.fit_transform(ydata)
fig = go.Figure()
fig.add_trace(go.Scatter(y=ydata_standardized[aline,:,0], mode='lines', name='Real Part'))
fig.add_trace(go.Scatter(y=ydata_standardized[aline,:,1], mode='lines', name='Imaginary Part'))
fig.show()
#%%
X_train, X_test, y_train, y_test = train_test_split(xdata_standardized, ydata_standardized, test_size=0.2, random_state=42)
print("Training set shape for X:", X_train.shape)
print("Testing set shape for X:", X_test.shape)
print("Training set shape for y:", y_train.shape)
print("Testing set shape for y:", y_test.shape)
#%%
input_signal = Input(shape=(256, 2))
# Encoder
x = Conv1D(16, 3, activation="relu", padding="same")(input_signal) 
x = MaxPooling1D(2, padding="same")(x)
x = Conv1D(8, 3, activation="relu", padding="same")(x)
encoded = MaxPooling1D(2, padding="same")(x) 

# At this point the representation is compressed

# Decoder
x = Conv1D(8, 3, activation="relu", padding="same")(encoded)
x = UpSampling1D(2)(x)
x = Conv1D(16, 3, activation="relu", padding="same")(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(2, 3, activation="sigmoid", padding="same")(x)

autoencoder = Model(input_signal, decoded)
autoencoder.compile(optimizer='adam', loss='MeanSquaredError',metrics=['accuracy'])
autoencoder.summary()
#%%
# validation_data = [X_train, y_train]
autoencoder.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=[X_test, y_test])

#%%
real_fringes_test = testfringes.real
hilbert_fringes_test = hilbert(real_fringes_test)
data_test = np.stack((real_fringes_test, hilbert_fringes_test.imag), axis=-1)
data_test = data_test.reshape(256, -1, 2)
data_test = np.transpose(data_test,axes=[1,0,2])
data_test_standardized = scaler.transform(data_test.reshape(-1, data_test.shape[-1])).reshape(data_test.shape)
predictions = autoencoder.predict(data_test_standardized)
predictions_original_scale = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
predictions_original_scale = predictions_original_scale.transpose(1, 0, 2).reshape(256,256,16,2)
#%%
predicted = (predictions_original_scale[:,:,:,0]+1j*predictions_original_scale[:,:,:,1])
nK = predicted.shape[0]  # the size along the first dimension
zeroPadding = 0 
noiseFloorDb = 0  # noise floor in dB
nZ, nX, nY = predicted.shape  # fringes1 is 3D
zRef = nZ / 2  # zRef value
zSize = 256  # zSize value
# Apply hanning window along the first dimension
predicted = predicted * np.hanning(nK)[:, np.newaxis, np.newaxis]

predicted_padded = np.pad(predicted, ((zeroPadding, zeroPadding), (0, 0), (0, 0)), mode='constant')

# Fourier Transform
tom1True = fftshift(fft(fftshift(predicted_padded, axes=0), axis=0), axes=0)
tom1 = tom1True + (((10 ** (noiseFloorDb / 20)) / 1) * (randn(nZ, nX, nY) + 1j * randn(nZ, nX, nY)))

refShift = int((2 * zRef + zSize) / zSize * nZ) // 2
tom1 = np.roll(tom1, refShift, axis=0)
tom1True = np.roll(tom1True, refShift, axis=0)

plot = 10*np.log10(abs(tom1[:,:,0])**2)
fig = px.imshow(plot,color_continuous_scale='gray',zmin=70,zmax=150)
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(predicted[:,1,1]), mode='lines', name='Real Part'))
fig.add_trace(go.Scatter(y=np.imag(predicted[:,1,1]), mode='lines', name='Imaginary Part'))
fig.show()
