#%% import libraries
import numpy as np
import os
import matplotlib.pyplot as plt



#%% functions

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
        tomogram = tomogram.reshape((depth, height, width))
    return tomogram

#%% read tomograms

base_path = 'C:\\Users\\dpulgarin\\Documents\\Python Scripts\\CXDLOCT\\Experimental_Data_complex'

tissues = ['depth_nail']#, 'depth_fovea', 'depth_opticNerve','depth_chicken']
all_tomograms = []
all_targets = []

for tissue in tissues:
    artifact_path = os.path.join(base_path, 'tomogram_artifacts', tissue)
    no_artifact_path = os.path.join(base_path, 'tomogram_no_artifacts', tissue)

    artifact_files = os.listdir(artifact_path)
    no_artifact_files = os.listdir(no_artifact_path)


    for real_file, imag_file in zip(artifact_files[::2], artifact_files[1::2]):
        real_file_path = os.path.join(artifact_path, real_file)
        imag_file_path = os.path.join(artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        all_tomograms.extend(tom.reshape(-1, dimensions[1], dimensions[2]))


    for real_file, imag_file in zip(no_artifact_files[::2], no_artifact_files[1::2]):
        real_file_path = os.path.join(no_artifact_path, real_file)
        imag_file_path = os.path.join(no_artifact_path, imag_file)
        dimensions = extract_dimensions(real_file[:-4])
        tomReal = read_tomogram(real_file_path, dimensions)
        tomImag = read_tomogram(imag_file_path, dimensions)
        tom = tomReal + 1j * tomImag
        all_targets.extend(tom.reshape(-1, dimensions[1], dimensions[2]))
    del tom, tomImag, tomReal
    print(tissue, ' loaded')

all_tomograms = np.array(all_tomograms)
all_targets = np.array(all_targets)
all_tomograms = all_tomograms[:,:,0:10]
all_targets = all_targets[:,:,0:10]
#%%
target_size = 512
middle_bscan = int(dimensions[0]/2)
partitions = int(dimensions[1]/target_size)
all_tomograms_partioned = []
all_targets_partioned = []
for i in range(partitions):
    zini = middle_bscan-(int(target_size/2))
    zend = middle_bscan+(int(target_size/2))
    xini = i*target_size
    xend = (i+1)*target_size
    minitom = all_tomograms[zini:zend , xini:xend ,:]
    minitarget = all_targets[zini:zend,xini:xend,:]
    all_tomograms_partioned.extend(minitom.reshape(-1, target_size, target_size))
    all_targets_partioned.extend(minitarget.reshape(-1, target_size, target_size))
all_targets_partioned = np.transpose(np.array(all_targets_partioned),(1,2,0))
all_tomograms_partioned = np.transpose(np.array(all_tomograms_partioned),(1,2,0))
#%%
# plt.imshow(10*np.log10(abs(all_tomograms[:,:,0])**2))
plt.imshow(10*np.log10(abs(all_tomograms[zini:zend,xini:xend,0])**2))

plt.imshow(10*np.log10(abs(all_tomograms[:,:,0])**2))