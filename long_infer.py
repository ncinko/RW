import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

samples = 100

def soft_update(net,nstim,u,y):
    # DE for internal variables
    #B = np.dot(net.Q, X)
    u = (1.-net.infrate)*u + net.infrate*(- 2*net.W.dot(y))
    u += np.random.normal(0.0, 0.05, (256,1))
    # external variables spike when internal variables cross thresholds
    y = np.array([u[:, ind] >= net.theta for ind in range(nstim)])
    y = y.T

    # reset the internal variables of the spiking units
    u = u*(1-y)
    
    return u,y


def get_noise_corr(act_list):
    noise_corr = [[[] for _ in range(256)] for _ in range(256)]
    for i in range(act_list.shape[2]):
        base_list = []
        for base in range(act_list.shape[1]):
            if np.average(act_list[:,base, i]) > 10:
                base_list.append(base)
        for neuron1, neuron2 in combinations(base_list,2):
            corr = np.corrcoef(act_list[:,neuron1,i], act_list[:,neuron2,i])[0,1]
            noise_corr[neuron1][neuron2].append(corr)
            
    noise_corr_avg = np.zeros((256,256))

    for i in range(256):
        for j in range(256):
            noise_corr_avg[i,j] = np.mean(noise_corr[i][j])
            
    return noise_corr, noise_corr_avg
            
    

def long_infer(net, X, act_list, infplot=False, savestr=None):
        """
        Simulate LIF neurons to get spike counts.
        Optionally plot mean square reconstruction error vs time.
        X:        input array
        Q:        feedforward weights
        W:        horizontal weights
        theta:    thresholds
        y:        outputs
        """

        nstim = X.shape[-1]
        new_act_list = np.zeros((samples,net.nunits,nstim))
        
        # projections of stimuli onto feedforward weights
        B = np.dot(net.Q, X)

        # initialize values. Note that I've renamed some variables compared to
        # Zylberberg's code. My variable names more closely follow the paper
        #u = np.zeros((net.nunits, nstim))  # internal unit variables
        u = np.random.normal(0.0, 0.5, size = (net.nunits, nstim))  # internal unit variables
        y = np.zeros((net.nunits, nstim))  # external unit variables
        #for t in range(100):
         #   u,y = soft_update(net,nstim,u,y)
        
        for i in range(samples):
            if i %100 == 0:
                print(i)
            acts = np.zeros((net.nunits, nstim))
            for t in range(1000):
                # DE for internal variables
                #B = np.dot(net.Q, X)
                u = (1.-net.infrate)*u + net.infrate*(B - 2*net.W.dot(y))
                u += np.random.normal(0.0, 0.05, (256,nstim))
                # external variables spike when internal variables cross thresholds
                y = np.array([u[:, ind] >= net.theta for ind in range(nstim)])
                y = y.T
    
                # add spikes to counts (after delay if applicable)
                if t >= net.delay:
                    acts = acts + y
    
                # reset the internal variables of the spiking units
                u = u*(1-y)
            new_act_list[i,:,:] = acts
        
        return np.concatenate((act_list, new_act_list), axis = 2)
        
        

