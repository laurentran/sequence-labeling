# this file is written to parse a specific dataset to construct an LSTM input layer and deserialize the data
# your resulting data structure should be a 3D matrix of [samples, time steps, features]

import pandas as pd
import numpy as np
import sys

# read data
prepivot = pd.read_csv('data/prepivot_weekly.csv')

# read category lookup into dictionary
categories = {}
with open("data/categories.txt") as f:
    for line in f:
       (key, val) = line.split()
       categories[key] = val

# construct input layer data
data = np.zeros((199, 153, 37))
labels = np.empty((199, 1))

sample = 0
previousID = ''
for index, row in prepivot.iterrows():
    if index == 0:
        previousID = row['LinkID']
        labels[0] = row['Goal']
    memberID = row['LinkID']
    if memberID != previousID:
        sample = sample + 1
        labels[sample] = row['Goal']
    timeStep = int(row['TimeStep']) - 1
    feature = int(categories[row['Category']])
    
    if timeStep <= 153:
        data[sample, timeStep, feature] = int(row['Amount'])
    
    previousID = memberID

# save input as .npy
np.save('./data/sequence_data.npy', data)
np.save('./data/sequence_labels.npy', labels)
