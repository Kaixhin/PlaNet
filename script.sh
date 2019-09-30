#!/bin/bash
#SBATCH --qos=low                             # Ask for job
#SBATCH --cpus-per-task=4                     # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=50G                             # Ask for 10 GB of RAM
#SBATCH --time=96:00:00                       # The job will run for 3 hours
#SBATCH -o /network/home/rajrohan/slurm-%j.out  # Write the log on tmp1
#SBATCH --error=/network/home/rajrohan/slurm-%j.err  # Write the log on tmp1

module load cuda/10.0
module load anaconda/3
module load mujoco/2.0

# for anaconda
source $CONDA_ACTIVATE
source activate p35

xvfb-run -s "-screen 0 1400x900x24" bash

# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.conda/envs/p35/lib"
python main.py --env 'Pusher3DOFDefault-v0' > /network/home/rajrohan/myoutput.txt
