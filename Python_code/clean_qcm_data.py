# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:29:07 2019

@author: a6q
"""

import pandas as pd
import matplotlib.pyplot as plt

filename = r'C:\Users\a6q\Desktop\PBI water uptake\PBI data\2019-06-27_17-15__qcm_params_pbi.csv'

df = pd.read_csv(filename).dropna(how='any')

d_cols = [col for col in df.columns if 'd_' in col]
ddf = df[d_cols]


# loop over each column
for col in d_cols:
    
    plt.plot(ddf[col], label='raw')
    
    
    # loop over each row
    for i in range(1, len(ddf[col])):
        
        if ddf[col].iloc[i] <= 0:
            
            ddf[col].iloc[i] = ddf[col].iloc[i-1] 
        
    
    plt.plot(ddf[col], label='clean')
    plt.title(col)
    plt.legend()
    plt.show()