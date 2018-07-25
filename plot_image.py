# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:33:35 2018

@author: Nick
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_image(im):
    pixels = int(np.sqrt(im.shape[0]))
    plt.imshow(np.reshape(im, (pixels, pixels)), cmap = 'plasma')
    plt.show()