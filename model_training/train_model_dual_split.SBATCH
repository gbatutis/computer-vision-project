#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=cv_train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mcs9834@nyu.edu
#SBATCH --output=/scratch/mcs9834/llm_env/cv-final-project/model_training/train_model_dual_split.out
module purge

singularity exec \
  --overlay /scratch/mcs9834/llm_env/my_pytorch.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "source /ext3/env.sh; sh /scratch/mcs9834/llm_env/cv-final-project/model_training/train_model_dual_split.sh"
