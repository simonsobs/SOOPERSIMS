#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=cov-sims

set -e

# Log file
log="./log_cov_sims"

export OMP_NUM_THREADS=1

module use --append /scratch/gpfs/SIMONSOBS/modules
module load soconda


basedir=/home/kw6905/bbdev/SOOPERSIMS/scripts  ## YOUR RUNNING DIRECTORY
cd $basedir

paramfile='../paramfiles/paramfile_cov.yaml'
paramfile_fit='../paramfiles/paramfile_cov_fit.yaml'
echo "Running pipeline with paramfile: ${paramfile}"

# # Run to estimate Gaussian foreground parameters from non-Gaussian sims
# python templates_fit.py --globals ${paramfile} --plots

# # Run to generate Gaussian covariance simulations
# python cov_sims.py --globals $paramfile_fit --plots"

com="srun -n 100 -c 4 --cpu_bind=cores \
     python -u \
     cov_sims.py --globals $paramfile_fit"

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1
echo "Ending batch script at $(date)"
