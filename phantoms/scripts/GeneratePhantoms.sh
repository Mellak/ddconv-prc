#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --output=output_phantoms/gphantoms_array_%a.out
#SBATCH --error=output_phantoms/gphantoms_array_%a.err
#SBATCH --array=1-1000%100
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU

# Access the simu_number variable
resolution=2
number=$((SLURM_ARRAY_TASK_ID + 0000))


# Define the directory path
directory="/homes/ymellak/PR_Correction/NewData/Phantoms_Ga68_${resolution}mm/Phantom_${number}"

# Check if the directory exists, if not create it
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

# max_dist_of_radius = 30.0 and base_resolution 2.0 are specific for Ga68, edit them for other isotopes

# Run the simulation using srun and singularity
srun singularity run --nv /homes/ymellak/python/SIF/pytorch.sif python '/homes/ymellak/PR_Correction/Python_Code/NewCode/GeneratePhantoms.py' \
  $resolution $number \
  --max_dist_of_radius 30.0 \
  --base_resolution 2.0 \
  --num_materials 10 \
  --min_zones 5 \
  --max_zones 10 \
  --min_zone_frac 0.05 \
  --max_zone_frac 0.3 \
  --num_shapes 12 \
  --shape_size_min 2 \
  --shape_size_max 10 \
  --offset_base 5 \
  --random_voxel_fraction 0.10 \
  --output_root "/homes/ymellak/PR_Correction/NewData"

