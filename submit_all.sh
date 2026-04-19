#!/bin/bash
# Submits all 5 experiments as separate SLURM jobs on Explorer, plus a final
# figure-builder job that waits for the others to finish.
#
# Usage (from ~/uav-rl on Explorer):
#   bash submit_all.sh
#
# Produces one SLURM job per experiment. Log files at ~/uav-rl/logs/exp{N}_<jobid>.out
# Final report: ~/uav-rl/results/paper_figures/

set -u
REPO=${REPO:-/home/m.dehghani/uav-rl}
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"

# Common SBATCH header (GPU node, 8 hours, 32 GB).
COMMON='#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

module load anaconda3/2024.06
cd REPO_PATH
python -c "import torch; print(\"CUDA:\", torch.cuda.is_available())"
'

# Build + submit one sbatch per experiment.
declare -a JOB_IDS
for E in 1 2 3 4 5; do
    case $E in
        1) MOD=experiments.exp1_reward_ablation ;;
        2) MOD=experiments.exp2_representation_ablation ;;
        3) MOD=experiments.exp3_scaling ;;
        4) MOD=experiments.exp4_algorithm_comparison ;;
        5) MOD=experiments.exp5_curriculum ;;
    esac
    SCRIPT="$LOGDIR/exp${E}.sbatch"
    cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=uav-exp${E}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=${LOGDIR}/exp${E}_%j.out
#SBATCH --error=${LOGDIR}/exp${E}_%j.err

echo "=== exp${E} start \$(date) on \$(hostname) ==="
module load anaconda3/2024.06
cd ${REPO}
python -u -m ${MOD}
echo "=== exp${E} done \$(date) ==="
EOF
    JID=$(sbatch --parsable "$SCRIPT")
    echo "submitted exp${E}: job ${JID}"
    JOB_IDS+=("$JID")
done

# Final figure-builder job: depends on all 5 finishing.
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
FIG_SCRIPT="$LOGDIR/build_figs.sbatch"
cat > "$FIG_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=uav-figs
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
echo "All submitted. Monitor with:  squeue -u \$USER"
echo "Tail any exp:                  tail -f ${LOGDIR}/exp1_<jobid>.out"
echo "Figures when done:             ls ${REPO}/results/paper_figures/"
