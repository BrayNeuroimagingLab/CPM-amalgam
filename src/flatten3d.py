"""
Preprocessing file for missing data and .mat files
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

FC_flat = np.array(FC_flat, dtype=np.float64)

# Remove columns containing nan
bad_inds = set(np.argwhere(np.isnan(FC_flat))[:,1])
print("Number of edges to remove because of nan: " + str(len(bad_inds)))
bad_inds = sorted(bad_inds)
print("FC_flat shape before deletion: " + str(FC_flat.shape))
FC_flat = np.delete(FC_flat, np.s_[bad_inds], axis=1)
print("FC_flat shape after deletion: " + str(FC_flat.shape))
print("Checking for nan: ")
print(set(np.argwhere(np.isnan(FC_flat))))

with open('../FC_flat.npy', 'wb') as f:
    np.save(f, np.array(FC_flat))

with open('../target.npy', 'wb') as f:
    np.save(f, np.array(Y, dtype=np.float64))

print("Save complete")
