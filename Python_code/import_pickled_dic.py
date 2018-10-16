import numpy as np
import pickle
import matplotlib.pyplot as plt


#%%


with open(r'exp_data\saved_surfaces.pkl',
          'rb') as handle:
    dic = pickle.load(handle)
