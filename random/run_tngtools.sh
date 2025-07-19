#!/bin/bash
#SBATCH --job-name=tngtools       # job name
#SBATCH --partition=cpu  # e.g. general, debug, etc.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00          # hh:mm:ss walltime
#SBATCH --mem=4G                 # adjust as needed
#SBATCH --account=bemi-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Loading modules"

# ——— load your environment ———
# (Adjust to how you activate conda / module load on Delta)
module load anaconda3_cpu
source activate hyrax      # or `conda activate hyrax`

echo "Loaded modules"

# ——— set your API key ———
# Option A: if you have it in ~/.env
# export $(grep -v '^#' ~/.env | xargs)
# Option B: hardcode (not recommended)
# export TNG50_API_KEY="3849226db021fa31e0fd58651b7c943f"

# ——— run TNG-Tools steps ———
echo "Generating URL list…"
tng-gen-urls \
  --api-key 3849226db021fa31e0fd58651b7c943f \

echo "Splitting FITS files…"

tng-split split \
  --url-list all_file_urls.txt \
  --batch-size 1000 \
  --split-output-dir split_images \
  --remove-parent \
  --catalog-path catalog.fits --api-key 3849226db021fa31e0fd58651b7c943f

echo "Done!"
