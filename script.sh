#!/bin/bash
#SBATCH --qos=low                             # Ask for job
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs for 1 task
#SBATCH --gres=gpu:1                          # Ask for 2 GPU
#SBATCH --mem=35G                             # Ask for 35 GB of RAM
#SBATCH --time=96:00:00                       # The job will run for 3 hours
#SBATCH -o /network/home/rajrohan/output/slurm-%A-%a.out  # Write the log on tmp1
#SBATCH --error=/network/home/rajrohan/error/slurm-%A-%a.err  # Write the log on tmp1
#SBATCH --array=0-5

module load cuda/10.0
module load anaconda/3
module load mujoco


# export PATH=/network/home/rajrohan/anaconda3/bin:$PATH

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/network/home/rajrohan/.mujoco/mjpro150/bin"

# for anaconda
source $CONDA_ACTIVATE
# conda activate p35
# alias python="~/.conda/envs/p35/bin/python"
# source /network/home/rajrohan/planet/bin/activate
conda activate planetConda
# Bhairav Suggestion
xvfb-run :$SLURM_JOB_ID -screen 0 84x84x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:$SLURM_JOB_ID

# xvfb-run :$SLURM_JOB_ID -screen 0 1400x900x24 -ac +extension GLX +render -noreset &> xvfb.log &
# xvfb-run -s "-screen $SLURM_ARRAY_TASK_ID 1400x900x24" bash

# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.conda/envs/p35/lib"

python main.py --env 'Pusher3DOFDefault-v0' --id 'October-2-Pusher3DOFDefault'
