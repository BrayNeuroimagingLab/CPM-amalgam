"""
Convert 3D Functional Connectivity object into 2D flattened matrix

No constraint on parcellation size

EDIT: The path to the .mat file and its ['key']

:return: FC_flat.npy,  n by p  predictor matrix
         target.npy    n by 1  phenotype target vector
         both saved to disk, place in root directory before running

@author Rylan Marianchuk
rylan.marianchuk@ucalgary.ca
July 2021
"""

import numpy as np
import scipy.io as sio




# Update path HERE
FC = sio.loadmat(r"../../CPM-Pipeline/in/empirical/268_108.mat")['all_mats']
Y = sio.loadmat(r"../../CPM-Pipeline/in/empirical/unStdAge.mat")['age']

FC = np.swapaxes(FC, 0, 2)

# Creating new FC_flat
FC_flat = []
for x in FC:
    sbj = []
    for i in range(len(x)):
        for j in range(i):
            sbj.append(x[i,j])
    FC_flat.append(sbj)

with open('../FC_flat.npy', 'wb') as f:
    np.save(f, np.array(FC_flat, dtype=np.float64))

with open('../target.npy', 'wb') as f:
    np.save(f, np.array(Y, dtype=np.float64))

print("Save complete")
