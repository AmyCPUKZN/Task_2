# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:21:13 2022

@author: Amy
"""

import numpy as np
import matplotlib.pyplot as plt


def forward_diff(f,x,h):
    dfdx=(f(x+h)-f(x))/h
    return dfdx

def backward_diff(f,x,h):
    dfdx=(f(x)-f(x-h))/h
    return dfdx

def centered_diff(f,x,h):
    dfdx=(f(x+h)-f(x-h))/h/2
    return dfdx

# Question 1a
def f(x):
    return x**4/4-x**2-x

x = np.linspace(-2,3,100)
h = x[1]-x[0]
dfdx = []

for x_n in x:
    # Append results of all three method to dfdx
    dfdx.append([forward_diff(f,x_n,h), backward_diff(f,x_n,h), centered_diff(f,x_n,h)])
# Convert to numpy array (for easier indexing)
dfdx = np.asarray(dfdx)
 
fig = plt.figure()
names = ['forard','backward','centered']
plt.plot(x,x**3-2*x-1,'-k', label = 'Theoretical')
for i in range(3):
    plt.plot(x,dfdx[:,i],'.', label = names[i])

plt.title('Question 1a  f(x) = x^4/4-x^2 -x ')    
plt.ylabel('df/dx')
plt.xlabel('x')
plt.legend()
plt.tight_layout()

fig = plt.figure()
names = ['forard','backward','centered']
for i in range(3):
    plt.plot(x,(dfdx[:,i]- (x**3-2*x-1)),'.', label = names[i])
 
plt.title('Question 1a  error: estimate - dfdx')
plt.ylabel('error')
plt.xlabel('x')
plt.legend()
plt.tight_layout()

    
#Question 1b
def g(x):
    return np.tanh(x)

x = np.linspace(-5,5,100)
h = x[1]-x[0]
dgdx = []

for x_n in x:
    dgdx.append([forward_diff(g,x_n,h), backward_diff(g,x_n,h), centered_diff(g,x_n,h)])
dgdx = np.asarray(dgdx)

fig = plt.figure()
names = ['forard','backward','centered']
plt.plot(x,1/np.cosh(x)**2,'-k', label = 'Theoretical')
for i in range(3):
    plt.plot(x,dgdx[:,i],'.', label = names[i])
 
plt.title('Question 1b  g(x) = tanh(x)')
plt.ylabel('dg/dx')
plt.xlabel('x')
plt.legend()
plt.tight_layout()

fig = plt.figure()
names = ['forard','backward','centered']
for i in range(3):
    plt.plot(x,(dgdx[:,i]- 1/np.cosh(x)**2),'.', label = names[i])

plt.title('Question 1b  error: estimate - dgdx')
plt.ylabel('error')
plt.xlabel('x')
plt.legend()
plt.tight_layout()

print('The errors are smallest for the centered difference in both parts a and b.')