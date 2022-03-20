############################################################
# CIS 521: Individual Functions for CNN
############################################################

student_name = "Steven Fandozzi"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


############################################################
# Individual Functions
############################################################

def helper(img, ker, row, col):
    total = 0
    for i in range(0, ker.shape[1]):
        for j in range(0, ker.shape[0]):
            total += img[row-int((ker.shape[0]-1)/2)+i][col-int((ker.shape[1]-1)/2)+j] * ker[i][j]
    return total
        
def convolve_greyscale(image, kernel):
    final_img = np.zeros(image.shape)
    kernel = np.fliplr(kernel)
    kernel = np.flipud(kernel)
    padx = int((kernel.shape[0]-1)/2)
    pady = int((kernel.shape[1]-1)/2)
    Img = np.pad(image, ((padx,pady),(padx,pady)), 'constant')
    for col in range(pady, image.shape[1]+pady):
        for row in range(padx, image.shape[0]+padx):
            update = helper(Img, kernel, row, col)
            final_img[row-padx][col-pady] = update
    return final_img

"""
import numpy as np
image = np.array([
         [0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9], 
         [10, 11, 12, 13, 14], 
         [15, 16, 17, 18, 19], 
         [20, 21, 22, 23, 24]])
kernel = np.array([
         [0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]])
print(convolve_greyscale(image, kernel))

import numpy as np
image = np.array([
         [0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9], 
         [10, 11, 12, 13, 14], 
         [15, 16, 17, 18, 19], 
         [20, 21, 22, 23, 24]])
kernel = np.array([
         [1, 2, 3],
         [0, 0, 0],
         [-1, -2, -3]])
print(convolve_greyscale(image, kernel))


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('5.1.09.tiff'))
plt.imshow(image, cmap='gray')
plt.show()
kernel = np.array([
         [0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]])
output = convolve_greyscale(image, kernel)
plt.imshow(output, cmap='gray')
plt.show()
print(output)"""
 
def convolve_rgb(image, kernel):
    final_img = np.zeros(image.shape)
    for dep in range(0, image.shape[2]):
        final_img[:, :, dep] = convolve_greyscale(image[:, :, dep], kernel)
    return final_img

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('4.1.07.tiff'))
plt.imshow(image)
plt.show()
kernel = np.array([
         [0.11111111, 0.11111111, 0.11111111],
         [0.11111111, 0.11111111, 0.11111111],
         [0.11111111, 0.11111111, 0.11111111]])
output = convolve_rgb(image, kernel)
plt.imshow(output.astype('uint8'))
plt.show()
print(np.round(output[0:3, 0:3, 0:3], 2))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('4.1.07.tiff'))
plt.imshow(image)
plt.show()
kernel = np.ones((11, 11))
kernel /= np.sum(kernel)
output = convolve_rgb(image, kernel)
plt.imshow(output.astype('uint8'))
plt.show()
print(np.round(output[0:3, 0:3, 0:3], 2))"""
 
def max_pooling(image, kernel, stride):
    M, N = image.shape
    Kx, Ky = kernel
    Lx, Ly = stride
    
    final = [[0 for col in range(int(M/Ly)-int(Ky-Ly))] for row in range(int(N/Lx)-int(Kx-Lx))]
    
    for y, col in enumerate(range(0, M-int(Ky-Ly), Ly)):
        for x, row in enumerate(range(0, N-int(Kx-Lx), Lx)):
            max_ele = 0
            if M < row+Kx-1 or N < col+Ky-1:
                continue 
            for i in range(0, Ky):
                for j in range(0, Kx):
                    if image[row+j][col+i] > max_ele:
                        max_ele = image[row+j][col+i]
            final[x][y] = max_ele
    return np.array(final)

def average_pooling(image, kernel, stride):
    M, N = image.shape
    Kx, Ky = kernel
    Lx, Ly = stride
    final = [[0 for col in range(int(M/Ly)-int(Ky-Ly))] for row in range(int(N/Lx)-int(Kx-Lx))]
    for y, col in enumerate(range(0, M-int(Kx-Lx), Ly)):
        for x, row in enumerate(range(0, N-int(Kx-Lx), Lx)):
            avg = 0
            count = 0
            if M < row+Kx-1 or N < col+Ky-1:
                continue 
            for i in range(0, Ky):
                for j in range(0, Kx):
                    avg += image[row+j][col+i]
                    count += 1
            final[x][y] = avg/count
    return np.array(final)

"""
image = np.array([
         [1, 1, 2, 4],
         [5, 6, 7, 8],
         [3, 2, 1, 0],
         [1, 2, 3, 4]])
kernel_size = (2, 2)
stride = (2, 2)
print(max_pooling(image, kernel_size, stride))

image = np.array([
         [1, 1, 2, 4],
         [5, 6, 7, 8],
         [3, 2, 1, 0],
         [1, 2, 3, 4]])
kernel_size = (2, 2)
stride = (1, 1)
print(max_pooling(image, kernel_size, stride))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('5.1.09.tiff'))
plt.imshow(image, cmap='gray')
plt.show()
kernel_size = (2, 2)
stride = (2, 2)
output = max_pooling(image, kernel_size, stride)
plt.imshow(output, cmap='gray')
plt.show()
print(output)
print(output.shape)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('5.1.09.tiff'))
plt.imshow(image, cmap='gray')
plt.show()
kernel_size = (4, 4)
stride = (1, 1)
output = max_pooling(image, kernel_size, stride)
plt.imshow(output, cmap='gray')
plt.show()
print(output)
print(output.shape)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('5.1.09.tiff'))
plt.imshow(image, cmap='gray')
plt.show()
kernel_size = (3, 3)
stride = (1, 3)
output = max_pooling(image, kernel_size, stride)
plt.imshow(output, cmap='gray')
plt.show()
print(output)
print(output.shape)

image = np.array([
         [1, 1, 2, 4],
         [5, 6, 7, 8],
         [3, 2, 1, 0],
         [1, 2, 3, 4]])
kernel_size = (2, 2)
stride = (2, 2)
print(average_pooling(image, kernel_size, stride))

image = np.array([
         [1, 1, 2, 4],
         [5, 6, 7, 8],
         [3, 2, 1, 0],
         [1, 2, 3, 4]])
kernel_size = (2, 2)
stride = (1, 1)
print(average_pooling(image, kernel_size, stride))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('5.1.09.tiff'))
plt.imshow(image, cmap='gray')
plt.show()
kernel_size = (2, 2)
stride = (2, 2)
output = average_pooling(image, kernel_size, stride)
plt.imshow(output, cmap='gray')
plt.show() 
print(output)



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('5.1.09.tiff'))
plt.imshow(image, cmap='gray')
plt.show()
kernel_size = (4, 4)
stride = (1, 1)
output = average_pooling(image, kernel_size, stride)
plt.imshow(output, cmap='gray')
plt.show() 
print(output)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open('5.1.09.tiff'))
plt.imshow(image, cmap='gray')
plt.show()
kernel_size = (3, 3)
stride = (1, 3)
output = average_pooling(image, kernel_size, stride)
plt.imshow(output, cmap='gray')
plt.show() 
print(np.round(output, 5))
"""

def sigmoid(x):
    sig = []
    for ele in range(len(x)):
        sig.append(1/(1 + np.exp(-x[ele])))
    return sig








