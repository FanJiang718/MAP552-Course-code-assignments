#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:10:30 2018

@author: FanJiang
"""

import numpy as np
import matplotlib.pyplot as plt

T =2.
N = 1000
n = 10
del_T = T/n

Z = np.random.randn(N, n)
W = np.cumsum(Z,axis = 1)
W = np.concatenate((np.zeros((N,1)), W), axis =1)

I = np.sum(W[:,0:-1]*(W[:,1:] - W[:,0:-1]), axis = 1)
J = np.sum(W[:,1:]*(W[:,1:] - W[:,0:-1]), axis = 1)
K = np.sum((W[:,1:]+W[:,0:-1])/2*(W[:,1:] - W[:,0:-1]), axis = 1)

WT_I = 0.5*W[:,-1]*W[:,-1] - I
WT_J = 0.5*W[:,-1]*W[:,-1] - J
WT_K = 0.5*W[:,-1]*W[:,-1] - K

mean_WT_I = np.mean(WT_I)
mean_WT_J = np.mean(WT_J)
mean_WT_K = np.mean(WT_K)

print("The sample mean of 1/2*W_T^2 - I_n: {}".format(mean_WT_I))
print("The sample mean of 1/2*W_T^2 - J_n: {}".format(mean_WT_J))
print("The sample mean of 1/2*W_T^2 - K_n: {}".format(mean_WT_K))

nn = np.arange(10,21)
for i in nn:
    
