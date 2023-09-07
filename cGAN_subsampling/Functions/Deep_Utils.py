import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import tifffile as tiff
import plotly.express as px
from skimage.metrics import structural_similarity as ssim


def tiff_3Dsave(array, filename):
    tiff.imwrite(filename, array)

def create_and_save_subplot(image1,
                            image2,
                            title1,
                            title2,
                            output_path,
                            dpi=300,
                            zmin=30,
                            zmax=100,
                            title_size=32,
                            title_color='black',
                            colorscale='gray',
                            file_name='file'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image1 = np.flipud(image1)
    image2 = np.flipud(image2)

    fig = make_subplots(rows=1, cols=2)

    fig.add_annotation(dict(text=title1, xref="x1", yref="paper",
                             x=int(image1.shape[0]/2),
                             y=1.07,
                             showarrow=False,
                             font=dict(size=title_size,color=title_color)))
    fig.add_annotation(dict(text=title2, xref="x2", yref="paper",
                            x=int(image1.shape[0]/2),
                            y=1.07,
                            showarrow=False,
                            font=dict(size=title_size,color=title_color)))

    fig.add_trace(go.Heatmap(z=image1, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=True,
                             colorbar=dict(y=0.5, len=(image1.shape[0]/(1*image1.shape[0])), yanchor="middle")), row=1, col=1)
    fig.add_trace(go.Heatmap(z=image2, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=False), row=1, col=2)

    fig.update_xaxes(scaleanchor = 'x',showticklabels=False, visible=False,showgrid=False)
    fig.update_yaxes(scaleanchor = 'x',showticklabels=False, visible=False,showgrid=False)
    
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=20),
            coloraxis_colorbar_x=0.83,
            font_family="Serif",
            font_size=24)
    html_name = '\\'+file_name+'.html'
    svg_name = file_name+'.svg'
    # fig.update_layout(title_text="Subplot of Two Images")
    # pio.write_image(fig, os.path.join(output_path, svg_name), format="svg", scale=dpi/72)
    fig.write_html(output_path + html_name)
    fig.show()

def sliding_window(image, window_size, step_size):
    """
    Create a sliding window over an image with a uniform size and stride.

    Args:
        image: A numpy array representing the input image.
        window_size: A tuple of two integers specifying the width and height of the window.
        stride: An integer specifying the stride of the sliding window.

    Returns:
        A list of numpy arrays, where each array represents a window from the image.
    """
    # Get the image height and width
    height, width = image.shape[:2]
    # Initialize the starting point of the sliding window
    x, y = 0, 0
    # Create an empty list to store the smaller images
    smaller_images = []
    # Iterate over the image
    while x <= width-window_size[0]:
        while y <= height-window_size[1]:
            # Get the current window
            window = image[y:y+window_size[1], x:x+window_size[0]]
            # Add the current window to the list of smaller images
            smaller_images.append(window)
            # overlap images Y axis
            ny = int(np.round((y + step_size[1])/2))
            if y != ny and ny + window_size[1] < height:
                window = image[ny:ny+window_size[1], x:x+window_size[0]]
                # Add the current window to the list of smaller images
                smaller_images.append(window)
            # overlap images X axis
            nx = int(np.round((x + step_size[0])/2))
            if x != nx and nx + window_size[0] < width:
                window = image[y:y+window_size[1], nx:nx+window_size[0]]
                # Add the current window to the list of smaller images
                smaller_images.append(window)
            # Move the sliding window to the bottom
            y += step_size[1]
        # Move the sliding window to the right
        x += step_size[0]
        y = 0
    # Return the list of smaller images
    smaller_images=np.array(smaller_images)
    return smaller_images

def inv_sliding_window(arr,window_size, original_size, step_size):
    """
    Recompose an image from a list of windows created by sliding_window.

    Args:
        windows: A list of numpy arrays representing the windows from the input image.
        output_shape: A tuple of two integers specifying the height and width of the output image.
        stride: An integer specifying the stride used to create the windows.

    Returns:
        A numpy array representing the reconstructed image.
    """
    # Create an empty image with original dimensions 
    imag_reconstruct = np.zeros(original_size)
    # individual original dimensions
    height = original_size[0]
    width  = original_size[1]
    # Initialize the starting point of the sliding window
    i, x, y, sc = 0,0,0,0
    # Iterate over the slices
    overlap_count = np.zeros(original_size)
    while i < len(arr):
        while x <= width-window_size[0]:
            while y <= height-window_size[1]:
                small_original =  imag_reconstruct[y:y+window_size[1],
                                                   x:x+window_size[0]]
                small_original = small_original + arr[i]
                overlap_count[y:y+window_size[1], x:x+window_size[0]] += 1
                imag_reconstruct[y:y+window_size[1],x:x+window_size[0]] = small_original   
                i+= 1    
                ny = int(np.round((y + step_size[1])/2))
                if y != ny and ny + window_size[1] < height:
                    small_original =  imag_reconstruct[ny:ny+window_size[1],
                                                       x:x+window_size[0]]
                    small_original = small_original + arr[i]
                    overlap_count[ny:ny+window_size[1], x:x+window_size[0]] += 1
                    imag_reconstruct[ny:ny+window_size[1],x:x+window_size[0]] = small_original  
                    i+=1
                nx = int(np.round((x + step_size[0])/2))
                if x != nx and nx + window_size[0] < width:
                    small_original =  imag_reconstruct[y:y+window_size[1], 
                                                       nx:nx+window_size[0]]
                    small_original = small_original + arr[i]
                    overlap_count[y:y+window_size[1], nx:nx+window_size[0]] += 1
                    imag_reconstruct[y:y+window_size[1],
                                     nx:nx+window_size[0]] = small_original  
                    i+=1
                y+=step_size[1]
            x+=step_size[0]
            y=0
    output = imag_reconstruct / overlap_count
    return output,overlap_count,imag_reconstruct

def calculate_oct_snr(tom):
    """
    Calculate the signal-to-noise ratio (SNR) for an OCT image.

    Args:
        oct_image: A 3D numpy array representing the OCT image, with dimensions (depth,height, width).

    Returns:
        The SNR of the OCT image.
    """

    oct_image = abs(tom[:,:,:,0]+1j*tom[:,:,:,1])**2
    # Determine the dimensions of the OCT image
    depth, height, width = oct_image.shape

    # Compute the signal as the mean of the intensity values across the entire image
    signal = np.mean(oct_image)

    # Compute the noise as the standard deviation of the intensity values within the background region
    background = oct_image[:,0:height//4, 0:width//4]
    noise = np.std(background)

    # Compute the SNR as the ratio of signal to noise
    snr = signal / noise

    return snr

def simple_sliding_window(tomData,tomShape,slidingYSize,slidingXSize,strideY,strideX):
    slices =[]
    for z in range(tomShape[0]):
            slidingYPos = 0
            # print(' z dimension :', z)
            while slidingYPos + slidingYSize <= tomShape[2]:
                slidingXPos = 0
                # print('\t sliding pos y :', slidingYPos)
                while slidingXPos + slidingXSize <= tomShape[1]:
                    # print('\t\t sliding pos x :', slidingXPos)
                    tomSlice = tomData[z, slidingXPos: slidingXPos + slidingXSize,
                                       slidingYPos:slidingYPos + slidingYSize, :]
                    slices.append(tomSlice)
                    slidingXPos = slidingXPos + strideX
                slidingYPos = slidingYPos + strideY
    slices = np.array(slices)
    return slices

def simple_inv_sliding_window(slices, tomShape, slidingYSize, slidingXSize, strideY, strideX):
    tomShapex = tomShape[1]
    tomShapey = tomShape[2]
    tomShapez = tomShape[0]
    tomShapec = tomShape[3]
    tomDataOver = np.zeros((tomShapez, tomShapex, tomShapey, tomShapec))
    slicesOver= slices
    sliceid = 0
    for z in range(tomShapez):
        slidingYPos = 0
        while slidingYPos + slidingYSize <= tomShapey:
            slidingXPos = 0
            while slidingXPos + slidingXSize <= tomShapex:
                tomSliceOver = slicesOver[sliceid]
                tomDataOver[z, slidingXPos: slidingXPos + slidingXSize,
                            slidingYPos:slidingYPos + slidingYSize, :] = tomSliceOver
                slidingXPos = slidingXPos + strideX
                sliceid = sliceid + 1
            slidingYPos = slidingYPos + strideY
    return tomDataOver

def MPS_single(plane,meandim):
    ftslice = np.fft.ifftshift(np.fft.fft2( np.fft.fftshift(plane)))
    powerSpectrumSlice = abs(ftslice)**2 / np.max(abs(ftslice)**2)
    meanPSslice = np.mean(powerSpectrumSlice, axis=meandim)
    meanPSslice = meanPSslice / np.max(meanPSslice)
    return meanPSslice

def Powerspectrum(plane):
    ftslice = np.fft.ifftshift(np.fft.fft2( np.fft.fftshift(plane)))
    powerSpectrum = abs(ftslice)**2
    powerSpectrumNorm = abs(ftslice)**2 / np.max(abs(ftslice)**2)
    return powerSpectrumNorm, powerSpectrum

def Correlation(slices, savename=None):
    slices = slices[:,:,0] + 1j*slices[:,:,1]
    
    correlationx = np.angle(slices[:,1:] * np.conjugate(slices[:,:-1]))
    correlationy = np.angle(slices[1:, :] * np.conjugate(slices[:-1, :]))
    # stdx = np.std(correlationx)
    # meanx = np.mean(correlationx)
    # stdy = np.std(correlationy)
    # meany = np.mean(correlationy)
    return correlationx, correlationy

def calculate_mse(image_original, image_reconstructed):
    """
    Calcula el Error Cuadrático Medio (MSE) entre dos imágenes.

    Parámetros:
    - image_original: numpy array 2D representando la imagen original en escala de grises.
    - image_reconstructed: numpy array 2D representando la imagen reconstruida por la cGAN.

    Devuelve:
    - MSE entre las dos imágenes.
    """
    # Asegúrate de que las imágenes tengan el mismo tamaño
    assert image_original.shape == image_reconstructed.shape, "Las imágenes deben tener el mismo tamaño."

    mse = np.mean((image_original - image_reconstructed) ** 2)
    return mse

def convert_to_dB(image):
    """
    Convierte una imagen OCT a escala de decibeles.

    Parámetros:
    - image: numpy array 2D representando la imagen original.

    Devuelve:
    - Imagen en escala de decibeles.
    """
    # Tomando la magnitud al cuadrado
    magnitude_squared = np.abs(image) ** 2
    
    # Convertir a escala de decibeles
    image_dB = 10 * np.log10(magnitude_squared + np.finfo(float).eps)  # Se añade un pequeño valor para evitar log(0)

    return image_dB

def calculate_ssim(image1, image2):
    """
    Calcula el Índice de Similitud Estructural (SSIM) entre dos imágenes.

    Parámetros:
    - image1, image2: numpy arrays 2D representando las imágenes a comparar.

    Devuelve:
    - SSIM entre las dos imágenes.
    """
    # Asegúrate de que las imágenes tengan el mismo tamaño y tipo de dato
    assert image1.shape == image2.shape, "Las imágenes deben tener el mismo tamaño."
    
    # El SSIM se calcula típicamente en imágenes de 8 bits (0-255)
    # Si tus imágenes no están en este rango, podrías considerar normalizarlas o adaptar el rango
    # Por simplicidad, aquí supondré que están en el rango [0, 1] (por ejemplo, tras una normalización)
    ssim_value, _ = ssim(image1, image2, full=True,data_range=1)
    return ssim_value

def calculate_psnr(image1, image2, max_val=1.0):
    """
    Calcula la Relación Señal-Ruido de Pico (PSNR) entre dos imágenes.

    Parámetros:
    - image1, image2: numpy arrays 2D representando las imágenes a comparar.
    - max_val: Valor máximo posible de la señal de la imagen. Por defecto es 1.0, para imágenes normalizadas.

    Devuelve:
    - PSNR entre las dos imágenes.
    """
    # Asegúrate de que las imágenes tengan el mismo tamaño y tipo de dato
    assert image1.shape == image2.shape, "Las imágenes deben tener el mismo tamaño."
    
    # Calcular el Error Cuadrático Medio (MSE)
    mse = np.mean((image1 - image2) ** 2)
    
    # Evitar un MSE de 0 (que daría un PSNR infinito)
    if mse == 0:
        return float('inf')

    # Calcular PSNR
    psnr = 10 * np.log10(max_val**2 / mse)
    
    return psnr