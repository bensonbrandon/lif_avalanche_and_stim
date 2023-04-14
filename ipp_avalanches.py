#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:57:52 2022

@author: bensonb
"""
import os
import sys
import numpy as np
import math
import decimal
decimal.getcontext().prec=1000

from mpmath import *
mp.prec=3000

import functools

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        # if not isinstance(args, collections.Hashable):
        #     return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

@memoized
def stirling_slow(n,k):
    # Stirling Algorithm
    # Cod3d by EXTR3ME
    # https://extr3metech.wordpress.com
    if n%1000==0:
        print(n,k)
    n1=n
    k1=k
    if n<=0:
        return 1
     
    elif k<=0:
        return 0
     
    elif (n==0 and k==0):
        return -1
     
    elif n!=0 and n==k:
        return 1
     
    elif n<k:
        return 0
 
    else:
        temp1=stirling_slow(n1-1,k1)
        temp1=k1*temp1
        return (k1*(stirling_slow(n1-1,k1)))+stirling_slow(n1-1,k1-1)
    
@memoized
def fact(x):
    if x==0:
        return 1
    return x*fact(x-1)

# memoize stirling numbers and factorials 0 to MAX*100
MAX = 65
[[stirling_slow(i*100,(j*100) +1) for j in range(i)] for i in range(1,MAX)]
[fact(i*100) for i in range(MAX)]

# @memoized
def Ps_tr(s,t,r,use_mp):
    '''
    Parameters
    ----------
    s : int
    t : int
    r : decimal.Decimal

    Returns
    -------
    decimal.Decimal

    '''
    
    if not use_mp:
        num = (r**s)*decimal.Decimal(fact(t)*stirling_slow(s,t))
        den = fact(s)*((r.exp() - 1)**t)
        
        try:
            return num/den
        except AttributeError:
            raise
        
    num = (r**s)*mpf(fact(t)*stirling_slow(s,t))
    den = fact(s)*((mp.exp(r)-1)**t)
    return num/den

# @memoized
def s_tr(t,r,use_mp):
    
    # accum = 0
    # for s in range(t,s_max):
    #     accum += s*Ps_tr(s,t,r,use_mp)
    # return accum
    if not use_mp:
        return (t*r)/(1-(-r).exp())
    return (t*r)/(1-mp.exp(-r))
    

# @memoized
def Pt_r(t,r,use_mp):
    if not use_mp:
        num = (r.exp() - 1)**(t-1)
        den = (r*t).exp()
        return num/den
    num = (mp.exp(r) - 1)**(t-1)
    den = mp.exp(r*t)
    return num/den

# @memoized
def Ps_r(s,r,use_mp):
    accum = 0
    for t in range(1,s+1):
        accum += Ps_tr(s,t,r,use_mp)*Pt_r(t,r,use_mp)
    return accum

# @memoized
def rho_r(r,use_mp):
    if not use_mp:
        return (1-((-r).exp()))*((-r).exp())
    return (1-mp.exp(-r))*mp.exp(-r)

# @memoized
def Pt_intarg(t,wr,r,use_mp):
    return wr*rho_r(r,use_mp)*Pt_r(t, r, use_mp)

# @memoized
def Ps_intarg(s,wr,r,use_mp):
    return wr*rho_r(r,use_mp)*Ps_r(s, r, use_mp)

# @memoized
def s_t_intarg(t,wr,r,use_mp):
    return wr*rho_r(r,use_mp)*s_tr(t, r, use_mp)

# @memoized
def stpt_intarg(t,wr,r,use_mp):
    return wr*rho_r(r,use_mp)*s_tr(t, r, use_mp)*Pt_r(t, r, use_mp)

# x: int
# wr, r
# if np.isscalar(wr):
#     wr = np.array([wr])
# if np.isscalar(r):
#     r = np.array([r])
# assert len(wr)==len(r)

def get_ps(x, wr, r):
    '''
    x: int
    wr: numpy array
    r: numpy array, the same length as wr
    '''
    farg = lambda i: Ps_intarg(x,mpf(wr[i]),mpf(r[i]),True)

    f_value = 0
    for i in range(len(r)):
        f_value = f_value + farg(i)
    f_logfloat = float(str(mp.log10(f_value)))
    return f_logfloat

def get_pt(x, wr, r):
    '''
    x: int
    wr: numpy array
    r: numpy array, the same length as wr
    '''
    farg = lambda i: Pt_intarg(x,mpf(wr[i]),mpf(r[i]),True)

    f_value = 0
    for i in range(len(r)):
        f_value = f_value + farg(i)
    f_logfloat = float(str(mp.log10(f_value)))
    return f_logfloat

def get_stpt(x, wr, r):
    '''
    x: int
    wr: numpy array
    r: numpy array, the same length as wr
    '''
    farg = lambda i: stpt_intarg(x,mpf(wr[i]),mpf(r[i]),True)

    f_value = 0
    for i in range(len(r)):
        f_value = f_value + farg(i)
    f_logfloat = float(str(mp.log10(f_value)))
    return f_logfloat

def get_norm(x, wr, r):
    '''
    x: int
    wr: numpy array
    r: numpy array, the same length as wr
    '''
    farg = lambda i: mpf(wr[i])*rho_r(r[i],True)

    f_value = 0
    for i in range(len(r)):
        f_value = f_value + farg(i)
    f_logfloat = float(str(mp.log10(f_value)))
    return f_logfloat
