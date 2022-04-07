#!/bin/bash
#SBATCH --job-name=imagenet              		   # name of job
#SBATCH --account=ofq@gpu
#SBATCH -C v100-32g 	                                   # reserving 32 GB GPUs only
#SBATCH --ntasks=16					   # total number of GPUs
#SBATCH --ntasks-per-node=4 				   # GPUs per node
#SBATCH --nodes=4                                  	   # reserving n node          
#SBATCH --gres=gpu:4                 			   # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10           			   # number of cores per task (1/4 of the 4-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         			   # hyperthreading is deactivated
#SBATCH --time=20:00:00             			   # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logfiles/log_%j.out                       # name of output file
#SBATCH --error=logfiles/log_%j.error                      # name of error file (here, in common with the output file)
#SBATCH --qos=qos_gpu-t3


cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.10.1

srun python ./main.py --data_dir /gpfsdswork/dataset/imagenet/RawImages --save_dir /gpfsstore/rech/ofq/uco38ei/imagenet/ \
					  --batch-size 128 --log-interval 100 --epochs 300 --lr 0.1
