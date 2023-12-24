#!/bin/bash

R=$1
B=$2
model_type=$3
encoder_dim=$4

temp_script=$(mktemp /tmp/train_script.XXXXXX)
cat <<EOT > $temp_script
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40GB
#SBATCH --job-name=run_$R
#SBATCH --output=run_$R.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pr2257@nyu.edu
singularity exec --nv \
--overlay /scratch/pr2257/arpl/keypoints_ws/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "source /ext3/env.sh; conda activate fw_env; cd ../; python train.py -r $R -b $B --model_type $model_type --encoder_dim $encoder_dim"
EOT

sbatch $temp_script
rm $temp_script