#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 18:13:17 2018

@author: FanJiang
"""

import numpy as np
import matplotlib.pyplot as plt

T=1
n = 10
del_T = T/(2**n)
copy = 1000

"""
Z = np.random.randn(copy,2**n)
W = np.cumsum(np.sqrt(del_T)*Z, axis = 1)
mean_WT= np.mean(W[:,-1])
var_WT = np.var(W[:,-1])
print("Mean of W_T: "+ str(mean_WT))
print("Variance of W_T: "+ str(var_WT))

W_T = W[:,-1]-mean_WT
W_T2 = W[:,2**(n-1)-1] - np.mean(W[:,2**(n-1)-1])
cov = 1./copy*np.sum(W_T*W_T2)
print("The covariance of W_T and W_T/2: "+ str(cov))
"""
"""
W = np.zeros((copy, 1+2**n))
W[:,-1] = T*np.random.randn(copy)
for i in range(1,n+1):
    step = 2**(n-i)
    for j in range(2**(i-1)):
        index = step+2*j*step
        W[:,index] = np.random.normal((W[:,index+step]+W[:,index-step])/2, T/(2**(i+1)))

mean_WT= np.mean(W[:,-1])
var_WT = np.var(W[:,-1])

W_T = W[:,-1]-mean_WT
W_T2 = W[:,2**(n-1)] - np.mean(W[:,2**(n-1)])
cov = 1./copy*np.sum(W_T*W_T2)

print("Mean of W_T: "+ str(mean_WT))
print("Variance of W_T: "+ str(var_WT))
print("The covariance of W_T and W_T/2: "+ str(cov))
"""

def simu_forward(copy,n,T):
    del_T = T/(2**n)
    Z = np.random.randn(copy,2**n)
    W_Q1 = np.cumsum(np.sqrt(del_T)*Z, axis = 1)
    return np.concatenate((np.zeros((copy,1)),W_Q1),axis = 1)



def simu_backward(copy,n,T):
    del_T = T/(2**n)
    W = np.zeros((copy, 1+2**n))
    W[:,-1] = T*np.random.randn(copy)
    for i in range(1,n+1):
        step = 2**(n-i)
        for j in range(2**(i-1)):
            index = step+2*j*step
            W[:,index] = np.random.normal((W[:,index+step]+W[:,index-step])/2, T/(2**(i+1)))
    return W

W = simu_backward(copy,n,T)
mean_WT= np.mean(W[:,-1])
var_WT = np.var(W[:,-1])
W_T = W[:,-1]-mean_WT
W_T2 = W[:,2**(n-1)] - np.mean(W[:,2**(n-1)])
cov = 1./copy*np.sum(W_T*W_T2)


nn = np.arange(10,21)
QV_forward = []
QV_backward = []
for i in nn:
    print(i)
    W_f = simu_forward(1,i,T)
    QV_forward.append(np.sum((W_f[0,1:]-W_f[0,0:-1])*(W_f[0,1:]-W_f[0,0:-1])))
    W_b = simu_backward(1,i,T)
    QV_backward.append(np.sum((W_b[0,1:]-W_b[0,0:-1])*(W_b[0,1:]-W_b[0,0:-1])))
    
plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.plot(nn,QV_forward, 'r',label = 'QV of forward simulation')
plt.axhline(T)
plt.legend(loc = 'best')
plt.title('Quadratic variantion of forward simulation for different n')
plt.xlabel('n')
plt.ylabel('QV')
plt.subplot(1,2,2)
plt.plot(nn,QV_backward, 'r', label = 'QV of backward simulation')
plt.axhline(T)
plt.legend(loc = 'best')
plt.title('Quadratic variantion of backward simulation for different n')
plt.xlabel('n')
plt.ylabel('QV')
plt.show()