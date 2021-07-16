import sys
import os
import numpy as np

# Iterations requested
iter = sys.argv[1]
print("Iterations requested: " + iter)
# Parcellation dimension of the FC matrix
parcel = sys.argv[2]
print("Parcellation dimension of the FC matrix: " + parcel)
parcel = int(parcel)
# Generate 1d-2d-map-268-35778
p = int((parcel ** 2 - parcel) / 2)

print(sys.argv[3])

# Preprocessing needed for all jobs
oneD = []
for i in range(parcel):
    for j in range(i):
        oneD.append((i,j))

with open("src/1d-2d-map-" + str(parcel) + "-" + str(p) + ".npy", 'wb') as f:
    np.save(f, np.array(oneD))

os.system("sbatch --output=$wdir/multivariate.out ./src/execute-multivariate.sh -export=iter,parcel,wdir")
#os.system("sbatch --output=$wdir/rf.out ./src/execute-randomforest.sh -export=iter,parcel,wdir")
#os.system("sbatch --output=$wdir/univariate.out ./src/execute-univariate.sh -export=iter,parcel,wdir")

