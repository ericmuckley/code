# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:54:27 2019

@author: a6q
"""

import pickle
#import pandas as pd


def open_pickle(filename):
    """
    Opens serialized Python pickle file as a dictionary.
    Filename should be something like 'saved_data.pkl'.
    """
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic





filename = r'C:\Users\a6q\exp_data\pp_multimode.pkl'


# open the file as a dictionary
dic = open_pickle(filename)