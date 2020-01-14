#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 21:34:59 2018

@author: FanJiang
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

#function of u and d
def f_u(b,T,n,sigma):
    h = T/n
    return np.exp(b*h+ sigma*np.sqrt(h))

def f_d(b,T,n,sigma):
    h = T/n
    return np.exp(b*h- sigma*np.sqrt(h))

def f_R(r,T,n):
    return np.exp(r*T/n)

def Sn(T,n,b,sigma,j,S0):
    u = f_u(b,T,n,sigma)
    d = f_d(b,T,n,sigma)
    S = [S0*(u**(j-i))*(d**(i)) for i in range(j+1)]
    S = np.array(S)
    return S

def Payoffn(T,n,b,sigma,K,S0):
    S  = Sn(T,n,b,sigma,n,S0)
    pay = S-K
    return pay*(pay>0)

def Calln(T,n,r,b,sigma,K,S0):
    R = f_R(r,T,n)
    u = f_u(b,T,n,sigma)
    d = f_d(b,T,n,sigma)
    q = (R-d)/(u-d)
    probs = np.array([sts.binom.pmf(k,n,1-q) for k in range(n+1)])
    price = np.exp(-r*T)* np.sum(Payoffn(T,n,b,sigma,K,S0)*probs)
    return price

def Deltan(T,n,r,b,sigma,K,j,S0):
    u = f_u(b,T,n,sigma)
    d = f_d(b,T,n,sigma)
    R = f_R(r,T,n)
    q = (R-d)/(u-d)
    S = Sn(T,n,b,sigma,j+1,S0)
    B = []
    probs = np.array([sts.binom.pmf(k,n-j-1,1-q) for k in range(n-j-1)])
    payoff_n = Payoffn(T,n,b,sigma,K,S0)
    t = T/n*(j+1)
    theta = []
    for i in range(len(S)):        
        B.append(np.exp(-r*(T-t))* np.sum(payoff_n[i:i+n-j-1]*probs))
    for i in range(len(B)-1):
        theta.append((B[i]-B[i+1])/(S[i]-S[i+1]))
    return theta

"""
Black-Scholes price and delta
"""
def d_plus(sigma,T,t,S,K,r):
    return 1./(sigma*np.sqrt(T-t))*(np.log(S/K)+(r+sigma*sigma/2.)*(T-t))
    
def d_minus(sigma,T,t,S,K,r):
    return d_plus(sigma,T,t,S,K,r) - sigma*np.sqrt(T-t)

def delta_BS(T,n,r,b,sigma,K,j,S0):
    S = Sn(T,n,b,sigma,j,S0)
    t = T/n*j
    d1 = d_plus(sigma,T,t,S,K,r)
    return sts.norm.cdf(d1)

def Call(T,r,sigma,K,S0):
    d1 = d_plus(sigma,T,0,S0,K,r)
    d2 = d_minus(sigma,T,0,S0,K,r)
    return S0*sts.norm.cdf(d1) - K*np.exp(-r*T)*sts.norm.cdf(d2)

if __name__ == "__main__":
    #main1()
    """
    Question 1
    """
    n= 50
    sigma = 0.3
    r = 0.05
    b = 0.05
    T =2
    S0 = 100
    K_list = [80+i for i in range(41)]
    price_list = [Calln(T,n,r,b,sigma,k,S0) for k in K_list]
    plt.figure()
    plt.plot(K_list,price_list)
    plt.xlabel('K')
    plt.ylabel('price')
    plt.title('The prices of call option for different K (Binomial model)')
    plt.show()

    j = 20
    delta = np.array([Deltan(T,n,r,b,sigma,k,j,S0) for k in K_list])
    delta_BS = np.array([delta_BS(T,n,r,b,sigma,k,j,S0) for k in K_list])
    
    fig, ax = plt.subplots()
    im = ax.imshow(delta)
    
    
    """
    Conclusion: 
    1.
    The price of the option decreases as the strike K increases, which means the value of 
    the call option is lower if the strike is high. Because for a given S0(100 here), it is 
    more difficult to have S_T>K if K is high, then the option is useless, thus, the value of
    this option decreases with the strike K.
    2.
    
    
    
    """
    
    
    
    """
    Question 2
    """
    """
    K = 105
    call_BS = Call(T,r,sigma,K,S0)
    nn = [10*i for i in range(1,101)]
    call_n = [Calln(T,n,r,b,sigma,K,S0) for n in nn]
    err = [call/call_BS-1 for call in call_n]
    plt.figure()
    plt.plot(nn,err, 'b', label = "err")
    plt.plot(nn,np.abs(err), 'r', label = "abs(err)")
    plt.xlabel('n')
    plt.ylabel('error')
    plt.title('The relative error between binomial model and BS model')
    plt.show()
    """