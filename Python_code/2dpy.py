# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hetero=False
inputfile1="spec.csv"

hetero=True
inputfile1=r"C:\Users\a6q\2Dpy\spec1.csv"
inputfile2=r"C:\Users\a6q\2Dpy\spec2.csv"

left_large=False
dynamic=True
num_contour=16


def contourplot(spec):
    x = spec.columns[0:].astype(float)
    y = spec.index[0:].astype(float)
    z = spec.values
    zmax = np.absolute(z).max()
    plt.figure(figsize=(4,4))
    plt.contour(x,y,z,num_contour,cmap='bwr',vmin=-1*zmax,vmax=zmax)
    # pyplot.pcolormesh(x,y,z,cmap='jet',vmin=-1*zmax,vmax=zmax)
    if left_large==True:
        plt.xlim(max(x),min(x))
        plt.ylim(max(y),min(y))


def hilbert_noda_transform(spec):
    # Returns the Hilber-Noda transformation matrix of a spectrum.
    noda = np.zeros((len(spec), len(spec)))
    for i in range(len(spec)):
        for j in range(len(spec)):
            if i != j:
                noda[i, j] = 1/math.pi/(j - i)
    return noda


#%% read files

spec1 = pd.read_csv(inputfile1,header=0, index_col=0).T
if hetero == False:
    inputfile2 = inputfile1
    
spec2 = pd.read_csv(inputfile2, header=0, index_col=0).T
if len(spec1)!=len(spec2):
    raise Exception('data mismatching')

spec1.T.plot(legend=None)
if left_large == True:
    plt.xlim(max(spec1.columns), min(spec1.columns))

if hetero == True:
    spec2.T.plot(legend=None)
    if left_large == True:
        plt.xlim(max(spec2.columns), min(spec2.columns))
if dynamic:
 spec1 = spec1 - spec1.mean()
 spec2 = spec2 - spec2.mean()
 




#%% perform synchronous correlation
sync = pd.DataFrame(spec1.values.T@spec2.values/(len(spec1) - 1))
sync.index = spec1.columns
sync.columns = spec2.columns
sync = sync.T
contourplot(sync)
#sync.to_csv(inputfile1[:len(inputfile1)-4]+'_sync.csv')


#%% perform asynchronous correlatoin

# get Hilbert-Noda transformation matrix
noda = hilbert_noda_transform(spec1)

# asynchronouse correlation
asyn = pd.DataFrame(spec1.values.T@noda@spec2.values/(len(spec1) - 1))
asyn.index = spec1.columns
asyn.columns = spec2.columns
asyn = asyn.T
contourplot(asyn)
#asyn.to_csv(inputfile1[:len(inputfile1)-4]+'_async.csv')



#%% 



data = sync
x = data.columns[0:].astype(float)
y = data.index[0:].astype(float)
z = data.values
zmax = np.absolute(z).max()


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.contour(x, y, z, num_contour, cmap='bwr')
ax.set_title('Synchronous')
ax.set_xlabel(r"$\nu$${_{1}}$ / cm${^{-1}}$")
ax.set_ylabel(r"$\nu$${_{2}}$ / cm${^{-1}}$")
plt.show()



fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.contour(x,y,z,num_contour,colors='black',linewidths=0.5,linestyles='solid',vmin=-1*zmax,vmax=zmax)
ax.pcolormesh(x,y,z,cmap='jet',vmin=-1*zmax,vmax=zmax)
