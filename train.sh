#!/bin/ksh 
#$ -q gpu
#$ -l hostname=webern57
#$ -o result.out
#$ -j y
#$ -N unet_gang
cd $WORKDIR
source /beegfs/data/work/c-2iia/in156281/venv/bin/activate
module load python
python /beegfs/data/work/c-2iia/in156281/unet_gang/training.py
