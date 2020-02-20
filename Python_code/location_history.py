# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:11:36 2019
@author: ericmuckley@gmail.com

This script reads a Location History.json file from Google Takeout
locatoin history and plots the latitude and longitude.

"""

#%%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

#%%

# json file containing location history data from Google Takeout
filename = r'C:\Users\a6q\exp_data\Location History.json'

# read the json file
data = pd.read_json(filename)
# keep all the entries as an array of dictionaries
data = data['locations'].values

# loop over each entry (location) and ofrmat the data
dates = np.empty(len(data)).astype(str)
cords = np.empty((len(data), 2))
for e, entry in enumerate(data):
    
    # format the timestamp of the entry
    ts_raw = int(entry['timestampMs']) / 1e3
    ts = datetime.fromtimestamp(int(ts_raw)).strftime('%Y-%m-%d %H:%M:%S')
    dates[e] = ts
    
    # format longitude and latitude
    long = entry['longitudeE7']
    lat = entry['latitudeE7']
    long = (long - 4294967296) / 1e7 if long > 9e8 else long / 1e7
    lat = (lat - 4294967296)  / 1e7 if lat > 9e8 else lat / 1e7
    cords[e] = [long, lat]

# create dataframe to hold all the data
df = pd.DataFrame(columns=['long', 'lat'], data=cords)
df['date'] = dates


#%%

plt.scatter(df['long'], df['lat'], s=5, c='k', marker='o')
plt.show()

















