# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np


def h5store(filename, key, df, **kwargs):
    '''Store pandas dataframes into an HDF5 file using a key and metadata'''
    store = pd.HDFStore(filename)
    store.put(key, df)
    store.get_storer(key).attrs.metadata = kwargs
    store.close()


def h5load(filename, key):
    '''Retrieve data and metadata from an HDF5 file using a key'''
    with pd.HDFStore(filename) as store:
        data = store[key]
        metadata = store.get_storer(key).attrs.metadata
    return data, metadata




aoddcols = ['a1', 'a3b', 'a5']
boddcols = ['b1', 'b3', 'b5']
aevencols = ['a2', 'a4', 'a6']
bevencols = ['b2', 'b4', 'b6']


aodd = pd.DataFrame(
        data=np.random.random((1000, 3)),
        columns=aoddcols)

bodd = pd.DataFrame(
        data=np.random.random((1000, 3)),
        columns=boddcols)

aeven = pd.DataFrame(
        data=np.random.random((1000, 3)),
        columns=aevencols)

beven = pd.DataFrame(
        data=np.random.random((1000, 3)),
        columns=bevencols)





filename = 'hdf_test.h5'

s = pd.HDFStore(filename)

s['a/odd'] = aodd
s['b/odd'] = bodd
s['a/even'] = aeven
s['b/even'] = beven













