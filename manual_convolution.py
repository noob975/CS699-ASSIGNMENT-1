# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 18:38:24 2022

@author: Aniruddha
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

cat = plt.imread("0000.jpg")
cat = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)

def normal_kernel(size=3, dev=0.3):
    normal = np.linspace(-(size - 1)/2., (size - 1)/2., size) 
    spread = np.exp(-0.5*np.square(normal)/np.square(dev)) #1D gaussian
    kernel = np.outer(spread, spread) #2D gaussian

    return kernel/np.sum(kernel)

def show_image(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    return 0
    
G = normal_kernel()
    
def convolve(img, kernel, stride=1, padding=0):
    
    M,N = img.shape

    def pad_image(img, padding_width):
        img_with_padding = np.zeros(shape=(
            M + padding_width*2,  
            N + padding_width*2
            ))
        
        img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img #only change inner elements of empty array
        
        return img_with_padding
    
    if padding:
        img = pad_image(img, padding)
        
    eps = 1e-12
    m,n = img.shape
    p,q = kernel.shape
    
    final = np.zeros((m,n))
    
    for i in range(0, m-p+1):
        
        for j in range(0, n-q+1, stride):
            
            sub = img[i:i+p, j:j+q] #part of image
            result = np.multiply(sub, kernel) #element-wise multiplication of kernel and part of image
            final[i,j] = np.sum(result) + eps #eps converts non-zero values for reshaping
    
    final = final[final != 0]
    s1 = int((M-p+2*padding)/stride + 1)
    s2 = int((N-q+2*padding)/stride + 1)
    final = final.reshape((s1,s2))
    
    return final

blurred_cat = convolve(cat, G)
show_image(blurred_cat)       