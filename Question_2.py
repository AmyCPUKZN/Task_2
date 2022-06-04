# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:21:34 2022

@author: Amy
"""

import numpy as np

def Trap_rule(f,a,b,n):
    x = np.linspace(a,b,n)
    h = x[1]-x[0]
    I = h*(np.sum(f(x))-f(x[0])/2-f(x[-1])/2)
    return I

def Simpsons_rule(f,a,b,n):
    x = np.linspace(a,b,n)
    h = x[1]-x[0]
    I = h/3*(f(x[0])+4*np.sum(f(x[1:-2:2]))+2*np.sum(f(x[2:-1:2]))+f(x[-1]))
    return I

def f(x):
    return np.exp(-x**2)

def g(x):
    return np.sqrt(1+x**2)

print('Question 2 a)')
print('Trapezoidal:',Trap_rule(g,1,4,100),\
      '\nSimpsons:\t',Simpsons_rule(g,1,4,100),\
       '\nExact:\t\t',0.5*(-np.sqrt(2) + 4*np.sqrt(17) - np.arcsinh(1) + np.arcsinh(4)))
    
print('Trapezoidal error:',Trap_rule(g,1,4,100)-0.5*(-np.sqrt(2) + 4*np.sqrt(17) - np.arcsinh(1) + np.arcsinh(4)) ,\
      '\nSimpsons error:\t',Simpsons_rule(g,1,4,100)-0.5*(-np.sqrt(2) + 4*np.sqrt(17) - np.arcsinh(1) + np.arcsinh(4)))

print('\nQuestion 2 b)')
print('Trapezoidal:',Trap_rule(f,-1000,1000,10000),\
      '\nSimpsons:\t',Simpsons_rule(f,-1000,1000,10000),\
       '\nExact:\t\t', np.sqrt(np.pi))
print('Trapezoidal error:',Trap_rule(f,-1000,1000,10000)-np.sqrt(np.pi),\
      '\nSimpsons error:\t',Simpsons_rule(f,-1000,1000,10000)-np.sqrt(np.pi))