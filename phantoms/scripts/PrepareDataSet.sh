#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --output=output_copy_rename/phantom_copy_%a.out
#SBATCH --error=output_copy_rename/phantom_copy_%a.err
#SBATCH --array=1-1000%100
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU

# Get the phantom number from the job array
phantom_number=$((SLURM_ARRAY_TASK_ID + 8000))
resolution=2
# Define the directory path for each phantom
phantom_dir="/homes/ymellak/PR_Correction/NewData/Phantoms_Ga68_${resolution}mm/Phantom_${phantom_number}"

# Check if the phantom directory exists
if [ -d "$phantom_dir" ]; then
  # Run the Python script inside the Singularity container
  srun singularity exec /homes/ymellak/python/SIF/pytorch.sif python /homes/ymellak/PR_Correction/Python_Code/NewCode/PrepareDataSet.py \
    $phantom_number $resolution \
    --dtype float32 \
    --base_resolution_mm 2.0 \
    --base_sizes 31 21 11 \
    --data_root "/homes/ymellak/PR_Correction/NewData"
else
  echo "Directory $phantom_dir does not exist"
fi

