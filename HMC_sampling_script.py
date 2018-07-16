# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:39:33 2018

@author: Nick
"""

import numpy as np
import scipy.io
import random
import fista


'''Load a dictionary of 300 basis functions, each with
dimensions of 16 x 16 pixels.  (~1.2 times overcomplete)'''

bases = np.load('basis1.npz')['basis']
pixels = 16

'''Load natural images'''

IMAGES = scipy.io.loadmat('IMAGES.mat')
IMAGES = IMAGES['IMAGES']
(imsize, imsize, num_images) = np.shape(IMAGES)

'''Randomly select image patch'''

def sample_images(sample_size, IMAGES):
    border = 4  #buffer around the edge of an entire image
    imi = np.ceil(num_images * random.uniform(0, 1))  #pick a random image
    I = np.zeros((pixels**2,sample_size))

    for i in range(sample_size):
        r = border + np.ceil((imsize-pixels-2*border) * random.uniform(0, 1))
        c = border + np.ceil((imsize-pixels-2*border) * random.uniform(0, 1))
        image = IMAGES[int(r):int(r+pixels), int(c):int(c+pixels), int(imi-1)]
        I[:,i] = np.reshape(image, pixels**2) 
        
    return I

I = sample_images(1, IMAGES)

ahat = fista.fista(I, bases, lambdav = 0.01, max_iterations=50)

'''HMC algorithm'''

def U(q, Phi = bases, lambdav = 0.01):
    
    return np.sum((I-Phi.dot(q))**2) + lambdav*np.sum(np.log(1 + q**2))

def grad_U(q, Phi = bases, lambdav = 0.01):
    
    grad = np.zeros(q.shape)
    reco_grad = np.sum((I - Phi.dot(ahat))*(-2*Phi), axis = 0)
    
    for i in range(300):
        grad[i,:] =  lambdav*2*q[i]/(1 + q[i]**2) + reco_grad[i]

    return(grad)
    
def HMC(epsilon, L, current_q):
    q = current_q
    p = 1e-2*np.random.normal(size = q.shape)
    current_p = p
    
    # Make a half step for momentum
    
    p += -epsilon*grad_U(q)/2
    
    # Alternate full steps for position and momentum
    
    for i in range(L):
        
        # Make a full step for the position
        
        q += epsilon*p
        
        # Make a full step for momentum, except at end of trajectory
        
        if i != L-1:
            
            p += -epsilon*grad_U(q)
            
    # Make a half step for momentum at the end
    
    p += -epsilon*grad_U(q)/2
    
    
    current_U = U(current_q)
    current_K = np.sum(current_p**2) / 2
    proposed_U = U(q)
    proposed_K = np.sum(p**2) / 2
    
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    #print(np.exp(current_U-proposed_U+current_K-proposed_K))

    if (np.random.uniform() < np.exp(current_U-proposed_U+current_K-proposed_K)):
        
        return q #accept
    
    else:
        return current_q # reject

sample=ahat
samplelist = np.zeros([100,300,1])

for i in range(100):
    
    if i % 100 ==0:
        print(i)
        
    print(U(sample, bases, 0.01))
    samplelist[i,:,:] = sample
    sample = HMC(0.1,25, sample)


