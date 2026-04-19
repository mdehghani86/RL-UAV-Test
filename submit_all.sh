#!/bin/bash
# Submits all 5 experiments as parallel SLURM jobs on Explorer's SHARING
# partition (better availability + bigger nodes), plus a final figure-builder
# that depends on them.
#
# Strategy:
#   - partition=sharing (A100/A5000/A6000/L40 nodes, much more idle capacity)
#   - --gres=gpu:a100:1 preference; SLURM will pick next-best if A100 unavailable
#   - 8 CPUs + 64 GB per job (our PPO + heuristic eval likes parallel cores)
#   - --time=08:00:00 (max safe)
#
# Usage (from ~/uav-rl on Explorer):
#   bash submit_all.sh

set -u
REPO=${REPO:-/home/m.dehghani/uav-rl}
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"

# Priority-ordered GPU types — SLURM tries each; first matching node gets the job.
# A100 > A6000 > A5000 > L40s > any gpu.
# We submit the same script; SLURM does the matching via --gres.
GPU_PREF='gpu:a100:1'   # override per-job below if needed

declare -a JOB_IDS
for E in 1 2 3 4 5; do
    case $E in
        1) MOD=experiments.exp1_reward_ablation;          NAME=uav-exp1-reward ;;
        2) MOD=experiments.exp2_representation_ablation;  NAME=uav-exp2-repr  ;;
        3) MOD=experiments.exp3_scaling;                  NAME=uav-exp3-scale ;;
        4) MOD=experiments.exp4_algorithm_comparison;     NAME=uav-exp4-algo  ;;
        5) MOD=experiments.exp5_curriculum;               NAME=uav-exp5-curr  ;;
    esac
    SCRIPT="$LOGDIR/exp${E}.sbatch"
    cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --partition=sharing
#SBATCH --gres=${GPU_PREF}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=${LOGDIR}/exp${E}_%j.out
#SBATCH --error=${LOGDIR}/exp${E}_%j.err

echo "=== exp${E} start \$(date) on \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
module load anaconda3/2024.06
cd ${REPO}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -u -m ${MOD}
echo "=== exp${E} done \$(date) ==="
EOF
    JID=$(sbatch --parsable "$SCRIPT")
    echo "submitted exp${E} (${NAME}): job ${JID}"
    JOB_IDS+=("$JID")
done

# Figure-builder job — waits for all 5 to finish
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
FIG_SCRIPT="$LOGDIR/build_figs.sbatch"
cat > "$FIG_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=uav-figs
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterany:${DEP}
#SBATCH --output=${LOGDIR}/figs_%j.out

echo "=== build figures \$(date) ==="
module load anaconda3/2024.06
cd ${REPO}
python -u -m experiments.build_paper_figures
echo "=== done \$(date) ==="
EOF
FID=$(sbatch --parsable "$FIG_SCRIPT")
echo "submitted figure-builder: job ${FID} (runs after all 5 exps finish)"

echo
echo "Submitted jobs: ${JOB_IDS[*]} + figs=${FID}"
echo "Monitor:   squeue -u \$USER"
echo "Tail any:  tail -f ${LOGDIR}/exp1_<jobid>.out"
echo "Results:   ${REPO}/results/paper_figures/"
