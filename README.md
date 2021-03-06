# CPM-amalgam
A collection of Connectome Predictive Modelling edge selection methods with univariate, multivariate, and random forest **regression** models

*Author: Rylan Marianchuk*

*July 2021*

Python packages required:
```
glmnet-python
cython
pandas
plotly
scikit-learn
scipy
numpy
```

To run an amalgam of models on ARC (High Performance Computing cluster), activate the conda environment within `bray_bulk`:
```
export PATH=/bulk/bray_bulk/software/miniconda3/bin:$PATH
source activate CPM-amalgam
```
Set the working directory outside of `bray_bulk` where you have `FC_flat.npy`, `control.csv`, and `target.npy`,
```
export wdir=/home/rylan.marianchuk/name1
```
Set the iteration count.
```
export iter=100
```
Set the parcellation dimension of the FC data,
```
export parcel=268
```
Then naviage to `/bulk/bray_bulk/CPM-amalgam`, then call the main sbatch on the shell script,
```
sbatch --output=$wdir/output.out run_all.sh -export=iter,parcel,wdir
```
Wait for all jobs to finish, may take up to 48 hours depending on data size.

To run the figure generation, set environment variables again, since a new session likely wiped them,
```
export PATH=/bulk/bray_bulk/software/miniconda3/envs/CPM-amalgam:$PATH
source activate CPM-amalgam
export wdir=/home/rylan.marianchuk/name1
```
Run the summary.py file,
```
sbatch --output=$wdir/summary.out summary.sh -export=wdir
```
See `summary.out` for top performing models, and `/home/rylan.marianchuk/name1/results/figures` for html displays of the results.

`/home/rylan.marianchuk/name1/results/npy` holds all results on the disk in .npy format if figures need to be modified for a publication.

