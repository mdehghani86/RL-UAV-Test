#!/bin/bash
# Multi-agent UAV paper — submits all N x K x algo cells as parallel SLURM jobs.
#
# Sweeps:
#   N = 30, 50, 70, 100
#   K = 2, 3, 5
#   Heuristics (1 seed each, fast — 4 per cell)
#   MARL algos: ippo, mappo, ia3c (2 seeds each — 6 per cell)
#
# Total: 4 * 3 * (4 + 6) = 120 jobs (each short, bounded by --time)
# Heuristics are CPU-only; MARL jobs request 1 GPU each.
#
# Usage on Explorer:
#   cd ~/uav-rl && git pull origin feature/multi-agent-paper && bash submit_ma.sh

set -u
REPO=${REPO:-/home/m.dehghani/uav-rl}
LOGDIR=$REPO/logs_ma
mkdir -p "$LOGDIR"

N_LIST=(30 50 70 100)
K_LIST=(2 3 5)
HEURISTICS=("parallel_greedy" "cluster_greedy" "cluster_maxreward")
ALGOS=("ippo" "mappo" "ia3c")
SEEDS=(0 1)

declare -a JOB_IDS

# --- MARL jobs (GPU) ---
for N in "${N_LIST[@]}"; do
  for K in "${K_LIST[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        TAG="N${N}_K${K}_${ALGO}_s${SEED}"
        SCRIPT="$LOGDIR/${TAG}.sbatch"
        cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=ma_${TAG}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=${LOGDIR}/${TAG}_%j.out
#SBATCH --error=${LOGDIR}/${TAG}_%j.err

echo "=== ${TAG} start \$(date) on \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
module load anaconda3/2024.06
cd ${REPO}
export OMP_NUM_THREADS=4
python -u -m experiments.ma_exp_scaling --N ${N} --K ${K} --algo ${ALGO} --seed ${SEED}
echo "=== ${TAG} done \$(date) ==="
EOF
        JID=$(sbatch --parsable "$SCRIPT")
        echo "submitted ${TAG}: job ${JID}"
        JOB_IDS+=("$JID")
      done
    done
  done
done

# --- Heuristic jobs (short CPU) ---
for N in "${N_LIST[@]}"; do
  for K in "${K_LIST[@]}"; do
    for HEUR in "${HEURISTICS[@]}"; do
      TAG="N${N}_K${K}_${HEUR}_s0"
      SCRIPT="$LOGDIR/${TAG}.sbatch"
      cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=ma_${TAG}
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=${LOGDIR}/${TAG}_%j.out
#SBATCH --error=${LOGDIR}/${TAG}_%j.err

echo "=== ${TAG} start \$(date) on \$(hostname) ==="
module load anaconda3/2024.06
cd ${REPO}
python -u -m experiments.ma_exp_scaling --N ${N} --K ${K} --algo ${HEUR} --seed 0
echo "=== ${TAG} done \$(date) ==="
EOF
      JID=$(sbatch --parsable "$SCRIPT")
      echo "submitted ${TAG}: job ${JID}"
      JOB_IDS+=("$JID")
    done
  done
done

echo
echo "Submitted ${#JOB_IDS[@]} jobs total."
echo "Monitor:   squeue -u \$USER --format='%.10i %.20j %.10T %.10M'"
echo "Results:   ${REPO}/results_ma/"
