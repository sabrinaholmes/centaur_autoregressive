#!/bin/bash
#SBATCH --job-name=rl_centaur
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100:2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# Read the token from token.txt
TOKEN=$(cat token.txt)
#echo "Read token
echo "Read token: $TOKEN"
# Export the token as an environment variable
export HF_TOKEN="$TOKEN"
# Debugging: Print the current working directory and environment
echo "Current working directory: $(pwd)"
# 1. Load the tool from Spack
spack load miniconda3

# 2. Source the conda profile so the 'activate' command works
# Note: The path below is the standard one for Spack-installed Miniconda
source $(spack location -i miniconda3)/etc/profile.d/conda.sh

# 3. Now you can activate and run
conda activate unsloth_env
echo "Conda environment 'unsloth_env' activated."

srun python predictive_rl_centaur.py