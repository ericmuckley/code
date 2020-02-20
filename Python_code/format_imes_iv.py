# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:38:39 2020

@author: a6q
"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = r'C:\Users\a6q\exp_data\2020-01-16_09-03__iv.csv' 
df = pd.read_csv(filename).dropna()

df2 = pd.DataFrame({'bias': df.iloc[:, 0]})

# get array of open-circuit voltages
voc = []

for i in range(1, len(df.columns), 2):
    
    voc0 = df.iloc[np.argmin(np.abs(df.iloc[:, i])), 0]
    voc.append(voc0)
    
    curr0 = float(df.iloc[50, i])
    
    
    df2[str(i)] = df.iloc[:, i] * 1e6
    plt.plot(df2.iloc[:, 0], df2[str(i)])
plt.show()

'''
for i in range(1, len(df2.columns)):
    plt.semilogy(df2.iloc[:, 0], np.abs(df2.iloc[:, i]))
plt.show()

'''


plt.plot(voc)
plt.show()
