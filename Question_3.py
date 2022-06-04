# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:12:00 2022

@author: Amy
"""

import numpy as np
import matplotlib.pyplot as plt


# 1/2 Matrices correctly defined
# Select a matrix
#A = [[1,0],[0,-1]]
#A = [[0,1],[1,0]]
A = [[0,-1j],[1j,0]]
#A = [[4,1],[2,-1]]

def char_func(x):
    f = (A[0][0]-x)*(A[1][1]-x)-A[0][1]*A[1][0]
    return f.real

tol = 0.001

# Bisection method
def Bisection(f,x_min,x_max):
    x_A = (x_min+x_max)/2
    if f(x_min)*f(x_A) <= 0:
        return x_min, x_A
    else:
        return x_A, x_max
    
print('Bisection method:')

#First guess
x_min = -1.5
x_max = 0   
# Store center of interval
x_c_1 = [(x_min+x_max)/2]

if char_func(x_min)*char_func(x_max)>0:
    print('Interval chosen incorrectly, f(x_min)*f(x_max) < 0')
else:
    while x_max-x_min > tol:
        x_min, x_max = Bisection(char_func,x_min,x_max)
        x_c_1.append((x_min+x_max)/2)
    print('lambda =',(x_max + x_min)/2,'f(lambda) =',char_func((x_max + x_min)/2))

#Second guess
x_min = 0
x_max = 5
# Store center of interval
x_c_2 = [(x_min+x_max)/2]
if char_func(x_min)*char_func(x_max)>0:
    print('Interval chosen incorrectly, f(x_min)*f(x_max) < 0')
else:
    while x_max-x_min > tol:
        x_min, x_max = Bisection(char_func,x_min,x_max)
        x_c_2.append((x_min+x_max)/2)
    print('lambda =',(x_max + x_min)/2,'f(lambda) =',char_func((x_max + x_min)/2))
    


# Secant maethod
def Secant(f,x,h):
    x_new = (x*f(x-h)-(x-h)*f(x))/(f(x-h)-f(x))
    return x_new

print('\nSecant method:')
    
# First guess
x = -5*np.random.rand()
# Store estimations of root
x_n_1 = [x]
while abs(char_func(x)) > tol:
    x = Secant(char_func,x,0.01)
    x_n_1.append(x)
print('lambda =',x,'f(lambda) =',char_func(x))

# Second guess
x = 5*np.random.rand()
# Store estimations of root
x_n_2 = [x]
while abs(char_func(x)) > tol:
    x = Secant(char_func,x,0.01) 
    x_n_2.append(x)
print('lambda =',x,'f(lambda) =',char_func(x))

plt.figure()
plt.plot(x_c_1,'or',label='Bisection, first guess')
plt.plot(x_c_2,'ob',label='Bisection, second guess')

plt.plot(x_n_1,'^r',label='Secant, first guess')
plt.plot(x_n_2,'^b',label='Secant, second guess')

plt.title('Question 3 Convergence')
plt.ylabel('value')
plt.xlabel('n')
plt.legend()