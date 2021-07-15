#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu2019
#SBATCH --mem=100MB

mkdir ${wdir}/results
mkdir ${wdir}/results/npy
mkdir ${wdir}/results/figures

python src/run_all.py $iter $parcel $wdir


