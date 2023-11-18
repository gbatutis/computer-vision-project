#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=cv_train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mcs9843@nyu.edu
#SBATCH --output=/scratch/mcs9834/llm_env/cv_final_project/model_training/train_model.out
module purge
if [ -e /dev/nvidia0 ]; then nv="--nv"; fi
singularity exec $nv \
  --overlay /scratch/mcs9834/llm_env/my_pytorch.ext3:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "source /ext3/env.sh; sh /scratch/mcs9834/llm_env/model_training/train_model.sh"
