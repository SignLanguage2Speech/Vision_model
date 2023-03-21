#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Vision_model_CTCLoss_training
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: gpus in exclusive process mode --
#BSUB -gpu "num=4:mode=exclusive_process:mps=yes"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 24GB of system-memory
#BSUB -R "rusage[mem=24GB]"
#BSUB -R "select[gpu32gb]"

### -- set the email address --
#BSUB -u s204503@student.dtu.dk

### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Visual_Backbone_Pretraining-%J.out
#BSUB -e Visual_Backbone_Pretraining-%J.err
# -- end of LSF options --

nvidia-smi
### Load the cuda module

module load python3/3.10.7
module load cuda/12.0
module load cudnn/v8.3.2.44-prod-cuda-11.X
module load ffmpeg

source /zhome/2b/d/156632/Desktop/bachelor/Vision_model/bach/bin/activate
python3 /zhome/2b/d/156632/Desktop/bachelor/Vision_model/lightning_PHOENIX_main.py
# source /zhome/d6/f/156047/BachelorProject/Vision_model/bach/bin/activate
# python3 /zhome/d6/f/156047/BachelorProject/Vision_model/main.py
