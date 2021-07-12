#!/bin/bash [could also be /bin/tcsh]
#$ -S /bin/bash
#$ -N COVID19-variants
#$ -pe mvapich2-sam 32
#$ -cwd
#$ -o $HOME/logs/output-soccer
#$ -e $HOME/logs/errors-soccer
#$ -t 1:6:1

# avoid multithreading in numpy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# >>>  conda initialize >>>
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate soccer
# >>>  conda initialize >>>

cd $HOME/Repositories/covid19_variants_chile/run_on_cluster/

python -u ./cluster_runs.py -i $SGE_TASK_ID
