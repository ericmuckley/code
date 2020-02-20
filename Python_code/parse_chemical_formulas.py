# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:07:12 2019

@author: a6q
"""

import pandas as pd
import re



# import data from citrination
filename = r'C:\Users\a6q\exp_data\citrination-export.csv'
data = pd.read_csv(filename)


data = data[['Name', 'formula']]
data = data.dropna()


# convert series of chemical formulas to a list of strings
formulas = list(data['formula'])#.values


# def ():
"""Convert a list of string chemical formulas in LaTeX math subscript
notation into hash    .""" 


form_dict = {}

# loop over each chemical formula string
for f in formulas:
    
    form_dict[f] = {}
    
    # convert to list of strings split by uppercase letters 
    formula = re.findall('[A-Z][^A-Z]*', f)
    
    # loop over each atom in the chemical formula
    for atom in formula:
        
        # for single atoms
        if '$_{' not in atom:
            atom_num = 1
        #else:
        #    atom_num = atom.split('$_{')[0]#.split('}$')[0]
        
        form_dict[f][atom] = atom_num
        
# for mutiple atom

















