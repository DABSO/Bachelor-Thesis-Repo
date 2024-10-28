#!/usr/bin/env bash

module purge

module load anaconda
conda activate ba
echo "Loaded Anaconda"
cd /hpc/gpfs2/home/u/nedaniel/BachelorArbeit/ba-daniel-neu/
ls .
python3  