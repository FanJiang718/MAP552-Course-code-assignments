#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:57:42 2018

@author: FanJiang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.optimize as opt
from mpl_toolkits import mplot3d

Strikes=np.loadtxt('Strikes_CPS8.txt')
Maturities = np.loadtxt('Maturities_CPS8.txt')
Prices = np.loadtxt('Prices_CPS8_v2.txt')
n, m = Prices.shape

r = 0
K_min = 80
del_K = 0.1
del_T = 1./256
S0 = 100


def Call_BS(sigma, T,r,K,S0, C):
    T += 1e-15
    d1 = 1./(sigma*np.sqrt(T))*(np.log(S0/K)+(r+sigma*sigma/2.)*(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*sts.norm.cdf(d1) - K*np.exp(-r*T)*sts.norm.cdf(d2) - C

vol_implicit = np.zeros((n,m))
for i in range(n):
    if i%50 ==0:
        print(i)
    for j in range(m):
        vol_implicit[i,j] = opt.fsolve(Call_BS, x0 = 0.4, args=(Maturities[j], r, Strikes[i],S0,Prices[i,j]))[0]
        
        
data_x = np.zeros((2, n*m))
data_y = np.zeros(n*m)
for i in range(n):
    for j in range(m):
        data_x[0, i*m+j] = Strikes[i]
        data_x[1, i*m+j] = Maturities[j]
        data_y[i*m+j] = vol_implicit[i,j]
        

def f(x,a,b,c,d,e):
    return (a + b*x[0] + c*x[0]*x[0])*np.exp(-d*x[1]) + e
popt, pcov = opt.curve_fit(f, data_x, data_y, bounds=([-np.inf, -np.inf,0,0,-np.inf], np.inf))

y_pred = f(data_x,*popt)
y_pred_reshaped = y_pred.reshape((n,m))



def d_plus(sigma, T,r,K,S0):
    return 1./(sigma*np.sqrt(T))*(np.log(S0/K)+(r+sigma*sigma/2.)*T)

def d_minus(sigma, T,r,K,S0):
    return d_plus(sigma, T,r,K,S0) - sigma*np.sqrt(T)

I_T = (y_pred_reshaped[:,1:] - y_pred_reshaped[:,:-1])/del_T
I_T = I_T[1:-1,:]
I_K = (y_pred_reshaped[1:,:] - y_pred_reshaped[:-1,:])/del_K
I_K = I_K[:-1,1:]
#I_KK = (I_K[1:,:] - I_K[:-1,:])/del_K
I_KK = (y_pred_reshaped[2:,:] + y_pred_reshaped[0:-2,:] -2*y_pred_reshaped[1:-1,:])/(del_K**2)
I_KK = I_KK[:,1:]
I = y_pred_reshaped[1:-1,1:]
T = Maturities[1:].reshape((1,-1))
K = Strikes[1:-1].reshape((-1,1))
d1 = d_plus(I,T,r,K,S0)
d2 = d_minus(I,T,r,K,S0)
vol_imp_loc = I/T + 2*I_T
vol_imp_loc = vol_imp_loc /(1/I/T + K*K*(2*d1/(K*I*np.sqrt(T))*I_K+d1*d2/I*I_K*I_K + I_KK))



vol_imp_loc_new = np.sqrt(vol_imp_loc)

def interpol(x1,x2,y1,y2,s):
    return y1 + (s-x1)*(y2-y1)/(x2-x1)

def vol_interpo(S, i, vol, Strikes):
    result = np.zeros(S.shape)
    for j, s in enumerate(S):
        indices = np.where(Strikes>s)
        if len(indices[0]) <=1 :
            result[j] = vol[-1,i]
        elif len(indices[0]) ==len(Strikes):
            result[j] = vol[0,i]
        else:
            index = indices[0][0]
            result[j] = interpol(Strikes[index-1], Strikes[index], vol[index-2,i], vol[index-1,i],s)
    return result


N = 1000
S = np.zeros((N,m))
S[:,0] = S0 
B = np.random.randn(N,m)*np.sqrt(del_T)
C_simulated = np.zeros((n,m))
for i in range(0,m-1):
    S[:,i+1] = S[:,i] *(1 + B[:,i] * vol_interpo(S[:,i], i, vol_imp_loc_new, Strikes))

for i, k in enumerate(Strikes):
    for j in range(m):
        C_simulated[i,j] = np.mean((S[:,j]-k)*(S[:,j] > k))

difference = C_simulated - Prices