#!/bin/bash
#SBATCH -J RM2_code_exercise_job
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 02:00:00


# Load modules 
module load apps/binapps/anaconda3/2024.10
module load cuda/12.6.2


# Download data if not already present
if [ ! -f Compiled_Reviews.txt ]; then
    wget https://raw.githubusercontent.com/cbannard/lela60331_24-25/refs/heads/main/coursework/Compiled_Reviews.txt
fi

# Run
cd ~
python RM2_code_exercise.py $SLURM_JOB_ID
