# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:58:15 2018

@author: a6q
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_table(r'exp_data\alex.txt')


fit = np.polyfit(data.x, data.y, 4)




