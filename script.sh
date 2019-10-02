#!/bin/bash
#SBATCH --qos=low                             # Ask for job
#SBATCH --gres=gpu:2                          # Ask for 1 GPU
#SBATCH --mem=200G                             # Ask for 10 GB of RAM
#SBATCH --time=96:00:00                       # The job will run for 3 hours
#SBATCH -o /network/home/rajrohan/output/slurm-%A-%a.out  # Write the log on tmp1
#SBATCH --error=/network/home/rajrohan/error/slurm-%A-%a.err  # Write the log on tmp1
#SBATCH --array=0-5

module load cuda/10.0
module load anaconda/3
module load mujoco/2.0

# for anaconda
source $CONDA_ACTIVATE
source activate p35

xvfb-run -s "-screen 0 1400x900x24" bash

# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.conda/envs/p35/lib"

python main.py --env 'Pusher3DOFDefault-v0' --id 'October-1-Pusher3DOFDefault-NewReward'
