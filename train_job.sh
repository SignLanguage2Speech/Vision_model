#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J wlasl_s2g_bs6
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=60GB]"
#BSUB -R "select[gpu80gb]"

### -- set the email address --
#BSUB -u s204138@student.dtu.dk

### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Visual_Encoder_Pretraining-%J.out
#BSUB -e Visual_Encoder_Pretraining-%J.err
# -- end of LSF options --

module load python3/3.10.7
module load cuda/12.0
module load cudnn/v8.3.2.44-prod-cuda-11.X


### source /zhome/6b/b/151617/env2/bin/activate ### Bigom venv
### python3 /zhome/6b/b/151617/Vision_model/PHOENIX_main.py

source /zhome/d6/f/156047/BachelorProject/Vision_model/bach/bin/activate ### Michael venv
python3 /zhome/d6/f/156047/BachelorProject/Vision_model/PHOENIX_main.py ###

