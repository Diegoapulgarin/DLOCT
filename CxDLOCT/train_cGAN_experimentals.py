import numpy as np
import os
import matplotlib.pyplot as plt


def read_tomogram(file_path, dimensions):
    depth, height, width = dimensions
    with open(file_path, 'rb') as file:
        tomogram = np.fromfile(file, dtype='single')
        tomogram = tomogram.reshape((depth, height, width))
    return tomogram

def extract_dimensions(file_name):
    parts = file_name.split('_')
    dimensions = []
    for part in parts:
        if 'z=' in part or 'x=' in part or 'y=' in part:
            number = int(part.split('=')[-1])
            dimensions.append(number)
    return tuple(dimensions)

base_path = 'C:\\Users\\USER\\Documents\\GitHub\\Experimental_Data_complex'

tissues = ['depth_fovea']#, 'depth_fovea', 'depth_opticNerve']
tomograms = []
target = []
for tissue in tissues:

    artifact_path = os.path.join(base_path, 'tomogram_artifacts', tissue)
    no_artifact_path = os.path.join(base_path, 'tomogram_no_artifacts', tissue)
    artifact_files = os.listdir(artifact_path)
    no_artifact_files = os.listdir(no_artifact_path)

    for real_file, imag_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(artifact_path, real_file)
        imag_file_path = os.path.join(artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        print(real_file_path)
        print(imag_file_path)
        print(dimensions)
        tomReal =  read_tomogram(real_file_path,dimensions)
        tomImag = read_tomogram(imag_file_path,dimensions)
        tom = tomReal+1j*tomImag
        tomograms.append(tom)

    for real_file, imag_file in zip(no_artifact_files[::2], no_artifact_files[1::2]):
        real_file_path = os.path.join(no_artifact_path, real_file)
        imag_file_path = os.path.join(no_artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        print(real_file_path)
        print(imag_file_path)
        print(dimensions)
        tomReal =  read_tomogram(real_file_path,dimensions)
        tomImag = read_tomogram(imag_file_path,dimensions)
        tom = tomReal+1j*tomImag
        target.append(tom)

tomograms = np.array(tomograms)
target = np.array(target)
bscan = 0
plot1 = 10*np.log10(abs(tomograms[0,:,:,bscan])**2)
plot2 = 10*np.log10(abs(target[0,:,:,bscan])**2)
fig,ax = plt.subplots(1,2)
ax[0].imshow(plot1,cmap='gray',vmin=20,vmax=120)
ax[1].imshow(plot2,cmap='gray',vmin=20,vmax=120)
fig.show()