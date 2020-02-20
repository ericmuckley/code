# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:45:32 2019
@author: ericmuckley@gmail.com


This script explores multivariate (multi-dimensional) data interpolation.


"""

from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

# create z values across grid
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

# number of points in each dimension of the grid
length = 500

# create a grid of possible (x, y) values
grid_x, grid_y = np.mgrid[0:1:length*1j, 0:1:length*1j]

# create (x,y) ordered pairs to sample
points = np.random.rand(length, 2)
# calculate z values for the (x,y) ordered pairs
values = func(points[:,0], points[:,1])


# create interpolated grids
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')


# plot results
plt.subplot(221)
plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(9, 9)
plt.show()
