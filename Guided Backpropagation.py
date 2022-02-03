# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 22:30:59 2022

@author: Aniruddha
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
#%%

'''
Basic functios to plot images and ready input image for VGG19
'''
def prep_input(path):
    image = tf.image.decode_png(tf.io.read_file(path))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [224,224])
    return image

def normalize(img):
    grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm

def comparison_plot(img1, img2,vmin=0.3,vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(20,50))
    plt.subplot(1,2,1)
    plt.imshow(img1,vmin=vmin, vmax=vmax, cmap = "gray")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(img2, cmap = "gray")
    plt.axis("off")
#%%
'''
Testing vanilla VGG19 on sample image. Image taken from internet for clear legibility.
'''
img_path = "siamese_cat.jpg"

base_model = VGG19(weights='imagenet')
test_model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('block5_conv4').output)

input_image = prep_input(img_path)
input_image = preprocess_input(input_image)
result = base_model(input_image)
print(decode_predictions(result.numpy()))

#%%
'''
designing custom ReLU for guided backprop
'''
@tf.custom_gradient
def backpropRelu(x):
    def grad(dy):
        return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
    return tf.nn.relu(x), grad

layer_dict = [layer for layer in test_model.layers[1:] if hasattr(layer,'activation')]
for layer in layer_dict: #replacing vanilla ReLU with custom Relu
    if layer.activation == tf.keras.activations.relu:
        layer.activation = backpropRelu
#%%
def findMostActivated(tensor, top = 5): #finds top 5 most activated neurons from tensor size (14,14,512)
    t = np.array(tensor)
    t_flat = t.flatten()
    
    def access_hwd(position): #returns 3D position of neuron from 1D index
        height = (position//512)//14
        width = (position//512)%14
        depth = position - (512*14*height + 512*width)
        return height, width, depth
    
    ind = np.argpartition(t_flat, -top)[-top:] #gets indices of top 5 values from t_flat
    ind = ind[np.argsort(t_flat[ind])][::-1]
    
    return np.array([access_hwd(i) for i in ind])

def trackEffect(input_img, Top = 5):
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_img) #watching input_img. After chain rule, gradient will eventually be against input_img.
        conv = test_model(input_img)[0] #output of last convolutional layer
        most_activated = findMostActivated(conv, top = Top) #top 5 activated neurons
            
        for i in range(Top):
            x,y,z = most_activated[i]
            max_conv = conv[x,y,z]
            grads = tape.gradient(max_conv, input_img)
            comparison_plot(normalize(grads[0]), normalize(input_img[0]))
        
    return most_activated, conv
    
top5_neurons, output = trackEffect(input_image)

    
    
    




