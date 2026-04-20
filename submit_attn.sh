#!/bin/bash
# Attention-based PPO scaling experiment — 10 jobs (5 N values × 2 seeds).
# Each job requests 1 GPU, ~1M env steps, bounded 4h.

set -u
REPO=${REPO:-/home/m.dehghani/uav-rl}
LOGDIR=$REPO/logs_attn
mkdir -p "$LOGDIR"

for N in 5 10 15 20 25; do
  for SEED in 0 1; do
    TAG="attn_N${N}_s${SEED}"
    SCRIPT="$LOGDIR/${TAG}.sbatch"
    cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${TAG}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=${LOGDIR}/${TAG}_%j.out
#SBATCH --error=${LOGDIR}/${TAG}_%j.err

echo "=== ${TAG} start \$(date) on \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
module load anaconda3/2024.06
cd ${REPO}
export OMP_NUM_THREADS=4
python -u -m experiments.exp_attn_scaling --N ${N} --seed ${SEED}
echo "=== ${TAG} done \$(date) ==="
EOF
  done
done

echo "Generated $(ls ${LOGDIR}/*.sbatch | wc -l) attention-PPO sbatch scripts in ${LOGDIR}/"
echo "Use the drip-feeder to submit them safely under the QoS cap."
