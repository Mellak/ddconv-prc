#!/bin/bash
#SBATCH --job-name=gkernel
#SBATCH --output=out_err_data/gate_%a.out
#SBATCH --error=out_err_data/gate_%a.err
#SBATCH --array=1-1000%300
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU

# Access the simu_number variable
resolution=2
number=$((SLURM_ARRAY_TASK_ID + 0000)) # to generate 1000 per 1000!
# Define the directory path
directory="/homes/ymellak/PR_Correction/NewData/Phantoms_Ga68_${resolution}mm/Phantom_${number}"

# Check if the directory exists, if not create it
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

srun singularity run /homes/ymellak/test_dir/gate_latest.sif '-a [resolution,'$resolution'][number,'$number'] /homes/ymellak/PR_Correction/GateSimulations/main_PR.mac'
