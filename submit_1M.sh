#!/bin/bash
# 1M-step re-run of exp2 (representation ablation) and exp3 (scaling sweep).
# Results tagged with _1M suffix so they don't overwrite tonight's runs.
# Each per-cell sbatch job; drip-feeder-safe (per-job, not a big loop).

set -u
REPO=${REPO:-/home/m.dehghani/uav-rl}
LOGDIR=$REPO/logs_1M
mkdir -p "$LOGDIR"

# --- exp2 cells: 6 variants × 3 seeds = 18 jobs ---
VARIANTS=(V0_baseline V1_relative_frame V2_time_features V3_tight_potential V4_cosine_entropy V5_all)
for V in "${VARIANTS[@]}"; do
  for SEED in 0 1 2; do
    TAG="exp2_1M_${V}_s${SEED}"
    SCRIPT="$LOGDIR/${TAG}.sbatch"
    cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${TAG}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=${LOGDIR}/${TAG}_%j.out
#SBATCH --error=${LOGDIR}/${TAG}_%j.err
echo "=== ${TAG} start \$(date) on \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
module load anaconda3/2024.06
cd ${REPO}
export OMP_NUM_THREADS=4
python -u -c "
import sys
sys.path.insert(0, '${REPO}')
from experiments.exp2_representation_1M import VARIANTS, ITERS, STEPS_PER_ITER, EPISODES_EVAL
from experiments.exp_common import env_factory, train_ppo, save_summary, eval_heuristic
from src.heuristic import evaluate_policy
from src.run_logger import RunLogger
from pathlib import Path
N = 25
variant = '${V}'
seed = ${SEED}
out_dir = Path('${REPO}') / 'results' / 'exp2_representation_1M'
out_dir.mkdir(parents=True, exist_ok=True)
opts = dict(VARIANTS[variant])
ent_schedule = opts.pop('ent_schedule')
env_fn = env_factory(N=N, **opts)
logger = RunLogger(str(out_dir / f'run_{variant}__s{seed}'),
                   {'exp':'representation_1M','variant':variant,'N':N,'seed':seed,'iters':ITERS})
trainer = train_ppo(env_fn, iters=ITERS, steps_per_iter=STEPS_PER_ITER,
                    ent_schedule=ent_schedule, seed=seed, logger=logger)
summary, _ = evaluate_policy(env_fn, lambda e, o, i: trainer.act(o, i),
                              episodes=EPISODES_EVAL, seed=seed+777)
save_summary(out_dir, f'{variant}__s{seed}',
             {'exp':'representation_1M','policy':'PPO','variant':variant,'N':N,'seed':seed,
              'iters':ITERS,'steps_per_iter':STEPS_PER_ITER,**opts,'ent_schedule':ent_schedule,**summary})
"
echo "=== ${TAG} done \$(date) ==="
EOF
  done
done

# --- exp3 cells: 5 Ns × 2 seeds = 10 jobs ---
for N in 5 10 15 20 25; do
  for SEED in 0 1; do
    TAG="exp3_1M_N${N}_s${SEED}"
    SCRIPT="$LOGDIR/${TAG}.sbatch"
    cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${TAG}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=${LOGDIR}/${TAG}_%j.out
#SBATCH --error=${LOGDIR}/${TAG}_%j.err
echo "=== ${TAG} start \$(date) on \$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
module load anaconda3/2024.06
cd ${REPO}
export OMP_NUM_THREADS=4
python -u -c "
import sys
sys.path.insert(0, '${REPO}')
from experiments.exp3_scaling_1M import _iters_for, EPISODES_EVAL, PAD_N, PAD_M
from experiments.exp_common import env_factory, train_ppo, save_summary, eval_heuristic
from src.heuristic import evaluate_policy
from src.run_logger import RunLogger
from pathlib import Path
N = ${N}; seed = ${SEED}
out_dir = Path('${REPO}') / 'results' / 'exp3_scaling_1M'
out_dir.mkdir(parents=True, exist_ok=True)
iters = _iters_for(N)
env_fn = env_factory(N=N, pad_num_customers=PAD_N, pad_num_chargers=PAD_M,
                     use_relative_frame=True, include_time_features=True,
                     potential_mode='tight')
logger = RunLogger(str(out_dir / f'run_ppo_enhanced__N{N}__s{seed}'),
                   {'exp':'scaling_1M','policy':'PPO-enhanced','N':N,'seed':seed,'iters':iters})
trainer = train_ppo(env_fn, iters=iters, steps_per_iter=4096,
                    ent_schedule='cosine', seed=seed, logger=logger)
summary, _ = evaluate_policy(env_fn, lambda e, o, i: trainer.act(o, i),
                              episodes=EPISODES_EVAL, seed=seed+777)
save_summary(out_dir, f'ppo_enhanced__N{N}__s{seed}',
             {'exp':'scaling_1M','policy':'PPO-enhanced','N':N,'seed':seed,'iters':iters,**summary})
# Also eval heuristics for this N
for h in ['greedy_db','nearest_deadline','nearest_neighbour','random']:
    s = eval_heuristic(h, env_fn, episodes=EPISODES_EVAL, seed=999)
    save_summary(out_dir, f'{h}__N{N}',
                 {'exp':'scaling_1M','N':N,'policy':h,'seed':0,**s})
"
echo "=== ${TAG} done \$(date) ==="
EOF
  done
done

echo "Scripts generated in ${LOGDIR}/. Queue them with the drip-feeder so they respect the QoS cap."
echo "Submit all now (no drip) — will likely hit cap if 28 jobs queued in one burst:"
echo "  for s in ${LOGDIR}/*.sbatch; do sbatch \$s; done"
