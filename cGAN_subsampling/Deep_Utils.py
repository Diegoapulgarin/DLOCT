
import numpy as np
import matplotlib.pyplot as plt

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
