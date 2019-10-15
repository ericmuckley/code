# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:36:30 2019

@author: a6q
"""


#import keras

import h5py
import numpy as np

filename = r'exp_data\voinovaFull.h5'

wdict = {}

def read_hdf5(path):
    # read hdf5 file and extract data from it
    weights = {}
    keys = []
    # open file
    with h5py.File(path, 'r') as f:
        # append all keys to list
        f.visit(keys.append) 
        for key in keys:
            #print(key)
            # contains data if ':' in key
            #print(f[key].name)
            #weights[f[key].name] = f[key].value
            
            group = f[key]
            print(group)
            print(type(group))
            if not isinstance(group, h5py._hl.group.Group):
                print('yes')
                print(np.shape(f[key]))
                print(f[key][()])
                wdict[key] = f[key][()]
            #for i in group:
            #    print(type(i))
            

            #print(np.shape(f[key]))
            
            '''
            if ':' in key: 
                print(f[key].name)
                weights[f[key].name] = f[key].value
                print(np.shape(weights))
            '''
            
    return weights, keys



data, keys = read_hdf5(filename)

'''
def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for path, _ in h5py_dataset_iterator(f):
            yield path



path2 = traverse_datasets(filename)

for i in path2:
    print(i)

    print(type(i))


'''