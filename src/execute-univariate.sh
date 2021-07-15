#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=cpu2019
#SBATCH --mem=100000MB

python ./src/CPM_univariate.py $iter $parcel $wdir


