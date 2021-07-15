# CPM-amalgam
A collection of Connectome Predictive Modelling edge selection methods with univariate, multivariate, and random forest models

```
export PATH=/bulk/bray_bulk/software/miniconda3/bin:$PATH
source activate CPM-amalgam
```
Set the working directory where you have `FC_flat.npy`, 'control.csv`, and `target.npy`,

```
export wdir=/home/rylan.marianchuk/name1
```
Set the iteration count. 100 Iterations ~= 24 hours. 1 iteration ~= 15 minutes.
```
export iter=100
```
Set the parcellation dimension of the FC data,
```
export parcel=268
```
