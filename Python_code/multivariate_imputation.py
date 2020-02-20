# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:40:14 2019
@author: ericmuckley@gmail.com

This script explored multivariate imputation for incomplete
machine learning data input.


"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# incomplete input data
input_data = [[1,2, 3], [3,4, np.nan], [5,6,7], [8,9, np.nan]]

# create imputer object
imp = IterativeImputer(max_iter=100, random_state=0)
# fit imputer to the input data
imp.fit(input_data)  


#X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]


full_input_data = imp.transform(input_data)
print(full_input_data)
