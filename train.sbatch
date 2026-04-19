#!/bin/bash
#SBATCH --job-name=uav-rl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/m.dehghani/uav-rl/train_%j.out
#SBATCH --error=/home/m.dehghani/uav-rl/train_%j.err

echo "=== Job started: $(date) on $(hostname) ==="
module load anaconda3/2024.06
cd /home/m.dehghani/uav-rl

echo "=== GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Training start: $(date) ==="
python -u -m experiments.run_enhanced_n25
echo "=== Training finished: $(date) ==="
