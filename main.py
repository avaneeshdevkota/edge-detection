import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# Load and convert the input image to grayscale

input_image = cv2.imread('lenna.png')
input_image = cv2.cvtColor(src=input_image, code=cv2.COLOR_BGR2GRAY)

def displayImage(image):

    # Create a subplot and display the image in grayscale

    plt.imshow(image, 'gray', vmin=0, vmax=255)
    plt.show()

def convolve2d(image, kernel):

    # Flip the kernel both vertically and horizontally

    kernel = np.flipud(np.fliplr(kernel))
    k_sizeX, k_sizeY = kernel.shape
    stride = 1
    pad_x = k_sizeX // 2
    pad_y = k_sizeY // 2
    
    # Pad the input image so that border pixels are not deleted

    image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)))
    im_sizeX, im_sizeY = image.shape
    
    # Create an output image with zeros

    output = np.zeros((im_sizeX, im_sizeY))
    
    # Perform convolution over the image

    for y in range(0, im_sizeY, stride):
        if y > im_sizeY - k_sizeY:
            break

        for x in range(0, im_sizeX, stride):
            if x > im_sizeX - k_sizeX:
                break

            # Convolve the kernel with the image region and store the result in the output

            output[int(np.floor((2 * x + k_sizeX) / 2 * stride)), int(np.floor((2 * y + k_sizeY) / 2 * stride))] = (kernel * image[x:x+k_sizeX, y:y+k_sizeY]).sum()
    
    # Remove the padding to get the final output

    return output[pad_x:(im_sizeX - pad_x) // stride, pad_y:(im_sizeY - pad_y) // stride]

def normalize(image):

    return (image / np.max(image)) * 255

def gaussianBlur(input_image):

    # Define the Gaussian blur kernel

    G = (1 / 273) * np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    
    # Perform convolution with the Gaussian blur kernel and return the normalized result

    return normalize(convolve2d(input_image, G))

def sobelOperator(input_image):

    # Define the Sobel kernels for detecting gradients in x and y directions

    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    Sy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    
    # Convolve the input image with the Sobel kernels to get the x and y gradients

    Gx = normalize(convolve2d(input_image, Sx))
    Gy = normalize(convolve2d(input_image, Sy))
    
    # Compute the gradient magnitude and direction

    G = normalize(np.sqrt(Gx * Gx + Gy * Gy))
    gradient_direction = np.degrees(np.arctan2(Gy, Gx))
    
    return G, gradient_direction

# Function to map the gradient direction to one of four sectors

def return_direction(angle):

    if (angle >= -22.5 and angle <= 22.5) or (angle <= -157.5 and angle >= 157.5):
        return 0
    elif (angle >= 22.5 and angle <= 67.5) or (angle <= -112.5 and angle >= -157.5):
        return 1
    elif (angle >= 67.5 and angle <= 112.5) or (angle <= -67.5 and angle >= -112.5):
        return 2
    elif (angle >= 112.5 and angle <= 157.5) or (angle <= -22.5 and angle >= -67.5):
        return 3

# Suppress pixels that are near to maximum value pixels

def nonMaxSuppression(G, gradient_direction):

    for i in range(len(G)):
        for j in range(len(G[0])):
            sector = return_direction(gradient_direction[i][j])

            if (sector == 0):
                if (j > 0 and j < len(G[0]) - 1):
                    if G[i][j] < G[i][j-1] or G[i][j] < G[i][j+1]:
                        G[i][j] = 0

            elif (sector == 1):
                if (j > 0 and j < len(G[0]) - 1 and i > 0 and i < len(G) - 1):
                    if G[i][j] < G[i-1][j+1] or G[i][j] < G[i+1][j-1]:
                        G[i][j] = 0

            elif (sector == 2):
                if (i > 0 and i < len(G) - 1):
                    if G[i][j] < G[i-1][j] or G[i][j] < G[i+1][j]:
                        G[i][j] = 0
            else:
                if (j > 0 and j < len(G[0]) - 1 and i > 0 and i < len(G) - 1):
                    if G[i][j] < G[i-1][j-1] or G[i][j] < G[i+1][j+1]:
                        G[i][j] = 0
    return G

def applyThresholding(image, threshold):

    edge_count = 0

    for i in range(len(image)):
        for j in range(len(image[0])):

            if (image[i][j] < threshold):
                image[i][j] = 0

            else:
                image[i][j] = 255
                edge_count += 1

    print("The edge count is:", edge_count)
    return image

def pThresholding(image, percentile):

    pixel_array = []

    for i in range(len(image)):
        for j in range(len(image[0])):

            if (image[i][j] > 0):
                pixel_array.append(image[i][j])

    pixel_array.sort()
    threshold_index = int(percentile * len(pixel_array) - 1)
    threshold = pixel_array[threshold_index]

    print("The threshold value is:", threshold)
    return applyThresholding(image, threshold)

gradients, gradient_direction = sobelOperator(gaussianBlur(input_image))
suppressed_image = nonMaxSuppression(gradients, gradient_direction)
final_image = pThresholding(suppressed_image, 0.90)

displayImage(final_image)
