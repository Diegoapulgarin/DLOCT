#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from Deep_Utils import simple_sliding_window, simple_inv_sliding_window
from Utils import logScaleSlices, inverseLogScaleSlices, downSampleSlices
path = r'E:\DLOCT\ultimo experimento subsampling paper\Fovea'
model = 'model_125952.h5'
model = tf.keras.models.load_model(os.path.join(path,model))
#%%
filename = '[p.SHARP][s.Eye2a][10-09-2019_13-14-42]_TomInt_z=(295..880)_x=(65..960)_y=(1..960)'
real = '_real.bin'
imag = '_imag.bin'
nZbin = 586
nXbin = 896
nYbin = 960
npol = 2
tom = np.fromfile(path+'\\'+filename+real,'single')
# tom = np.reshape(tom,(100,928,960),order='F')
tom = np.reshape(tom,(nZbin,nXbin,nYbin,npol),order='F')
tom = np.sum(tom,axis=3)
tomi = np.fromfile(path+'\\'+filename+imag,'single')
# tomi = np.reshape(tomi,(100,928,960),order='F')
tomi = np.reshape(tomi,(nZbin,nXbin,nYbin,npol),order='F')
tomi = np.sum(tomi,axis=3)
tomOriginal = np.stack((tom, tomi), axis=3)
del tom, tomi
tomOriginal = tomOriginal[200:202,:,:,:]
#%%
num_zeros = 64
pad_width = ((0, 0), (0, 0), (0, num_zeros), (0, 0))
tomDatas = np.pad(tomOriginal, pad_width, mode='edge')
tomShape = np.shape(tomDatas)
print(tomShape)
slidingYSize = 128
slidingXSize = 128
strideY = 128
strideX = 128
slices = simple_sliding_window(tomDatas,tomShape,slidingYSize,slidingXSize,strideY,strideX)
logslices, slicesMax, slicesMin = logScaleSlices(slices)
logslicesUnder = downSampleSlices(logslices)
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
feature_map_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
#%%
input = logslicesUnder[10:11,:,:,:]
#%%
feature_maps = feature_map_model.predict(input)
layer_names = [layer.name for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
def visualize_feature_maps(feature_maps, layer_names):
    for fmap, layer_name in zip(feature_maps, layer_names):
        print(f"Visualizando mapas de características para la capa: {layer_name}")
        n_features = fmap.shape[-1]
        size = fmap.shape[1]
        display_grid = np.zeros((size, size * n_features))
        
        for i in range(n_features):
            x = fmap[0, :, :, i]
            x -= x.mean()
            if x.std() > 0:
                x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x
        
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
visualize_feature_maps(feature_maps, layer_names)
