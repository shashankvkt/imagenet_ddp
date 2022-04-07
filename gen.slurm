#!/bin/bash
#SBATCH --job-name=imagenet              		   # name of job
#SBATCH --account=ofq@gpu
#SBATCH -C v100-32g 							   # reserving 32 GB GPUs only
#SBATCH --ntasks=8					 			   # total number of processes (= number of GPUs here)
#SBATCH --ntasks-per-node=4 
#SBATCH --nodes=2                                  # reserving 1 node          
#SBATCH --gres=gpu:4                 			   # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10           			   # number of cores per task (1/4 of the 4-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         			   # hyperthreading is deactivated
#SBATCH --time=03:00:00             			   # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logfiles/log.out    # name of output file
#SBATCH --error=logfiles/log.error   # name of error file (here, in common with the output file)
#SBATCH --qos=qos_gpu-t3

# cleans out the modules loaded in interactive and inherited by default 
module purge

# loading of modules
module load pytorch-gpu/py3/1.10.1

# echo of launched commands
set -x

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
		--data_dir /gpfsdswork/dataset/imagenet/RawImages --output_dir /gpfsstore/rech/ofq/uco38ei/imagenet_ddp/ \
		--batch_size_per_gpu 128 --epochs 300 --lr 0.1 --optimizer sgd \
		--weight_decay 1e-4