#!/bin/bash
#SBATCH --job-name=policy-compare
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out

# Matched PPO/direct-backprop benchmark on ITER hybrid flat-top GyroBohm.
#
# Default equality contract per training seed:
#   * 7,040,000 environment transitions
#   * 4,400 actor optimizer updates
#   * 1,600 transitions in each actor-gradient minibatch
#   * 100-step PPO/TBPTT temporal horizon
#   * identical 133 -> 64 -> 64 -> 4 swish residual-mean policy
#   * deterministic frozen-policy evaluation on the same 64 reset keys every
#     704,000 training transitions
#
# METHOD must be `ppo` or `direct_backprop`. All transition counts exclude
# evaluation rollouts.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$HOME/plasmax}"
cd "$REPO"
mkdir -p outputs/logs

METHOD="${METHOD:?set METHOD=ppo or METHOD=direct_backprop}"
ENV_SETUP="${ENV_SETUP:-iter/hybrid/flattop}"
BACKEND="${BACKEND:-bohm_gyrobohm}"
REWARD="${REWARD:-P_diff}"
VARIANT="${VARIANT:-realistic}"
DISRUPTION_PENALTY="${DISRUPTION_PENALTY:--5.0}"
SEED="${SEED:-0}"
# Tuning jobs explicitly pass EVAL_SEED=10000. The default is the disjoint
# final-curve bank so an accidentally omitted override cannot tune on test data.
EVAL_SEED="${EVAL_SEED:-20000}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-64}"
TRAINING_STEPS="${TRAINING_STEPS:-7040000}"
EVAL_FREQ="${EVAL_FREQ:-704000}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-policy-compare-final-iter-hybrid-flattop-bgb}"
WANDB_ENTITY="${WANDB_ENTITY:-flair}"
DRY_RUN="${DRY_RUN:-0}"

# Shared actor shape. Defaults match both implementations without extra CLI
# flags; keep these explicit in the logged launcher contract.
HIDDEN_1="${HIDDEN_1:-64}"
HIDDEN_2="${HIDDEN_2:-64}"
ACTIVATION="${ACTIVATION:-swish}"

# Direct-backprop geometry.
ROLLOUT_STEPS="${ROLLOUT_STEPS:-4400}"
TRUNCATION_STEPS="${TRUNCATION_STEPS:-100}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-16}"
DIRECT_BATCH_STEPS=$((NUM_ROLLOUTS * ROLLOUT_STEPS))
DIRECT_PASSES="${DIRECT_PASSES:-$((TRAINING_STEPS / DIRECT_BATCH_STEPS))}"
DIRECT_EVAL_EVERY="${DIRECT_EVAL_EVERY:-$((EVAL_FREQ / DIRECT_BATCH_STEPS))}"
DIRECT_LR="${DIRECT_LR:-0.0001}"
DIRECT_GRAD_CLIP="${DIRECT_GRAD_CLIP:-1.0}"

# PPO geometry. One epoch means each collected transition is used in exactly
# one actor update; four minibatches make its 1,600-sample gradient batch match
# direct backprop's 16 rollouts x 100 truncated steps.
PPO_NUM_ENVS="${PPO_NUM_ENVS:-64}"
PPO_NUM_STEPS="${PPO_NUM_STEPS:-100}"
PPO_NUM_EPOCHS="${PPO_NUM_EPOCHS:-1}"
PPO_NUM_MINIBATCHES="${PPO_NUM_MINIBATCHES:-4}"
PPO_LR="${PPO_LR:-0.0003}"
PPO_INITIAL_LOG_STD="${PPO_INITIAL_LOG_STD:--1.2039728043259361}"
PPO_ENT_COEF="${PPO_ENT_COEF:-0.01}"
PPO_GAMMA="${PPO_GAMMA:-0.99}"
PPO_GAE_LAMBDA="${PPO_GAE_LAMBDA:-0.95}"
# The environment already applies explicit observation scales. Keeping RMS off
# makes PPO and direct backprop deploy the same residual-mean mapping.
PPO_NORMALIZE_OBSERVATIONS="${PPO_NORMALIZE_OBSERVATIONS:-0}"
PPO_NORMALIZE_REWARDS="${PPO_NORMALIZE_REWARDS:-1}"
REQUIRE_MATCHED_OPTIMIZATION="${REQUIRE_MATCHED_OPTIMIZATION:-1}"

PPO_ITERATION_STEPS=$((PPO_NUM_ENVS * PPO_NUM_STEPS))
DIRECT_GRADIENT_STEPS=$((NUM_ROLLOUTS * TRUNCATION_STEPS))
if [ "$PPO_NUM_MINIBATCHES" -le 0 ]; then
    echo "PPO_NUM_MINIBATCHES must be positive" >&2
    exit 2
fi
if [ $((PPO_ITERATION_STEPS % PPO_NUM_MINIBATCHES)) -ne 0 ]; then
    echo "PPO_NUM_ENVS * PPO_NUM_STEPS must be divisible by PPO_NUM_MINIBATCHES" >&2
    exit 2
fi
PPO_GRADIENT_STEPS=$((PPO_ITERATION_STEPS / PPO_NUM_MINIBATCHES))
DIRECT_UPDATES=$((DIRECT_PASSES * ROLLOUT_STEPS / TRUNCATION_STEPS))
PPO_UPDATES=$((TRAINING_STEPS / PPO_ITERATION_STEPS * PPO_NUM_EPOCHS * PPO_NUM_MINIBATCHES))

if [ "$METHOD" != "ppo" ] && [ "$METHOD" != "direct_backprop" ]; then
    echo "METHOD must be ppo or direct_backprop, got: $METHOD" >&2
    exit 2
fi
if [ $((DIRECT_PASSES * DIRECT_BATCH_STEPS)) -ne "$TRAINING_STEPS" ]; then
    echo "direct-backprop geometry does not equal TRAINING_STEPS" >&2
    exit 2
fi
if [ $((TRAINING_STEPS % PPO_ITERATION_STEPS)) -ne 0 ]; then
    echo "TRAINING_STEPS must be divisible by PPO_NUM_ENVS * PPO_NUM_STEPS" >&2
    exit 2
fi
if [ $((EVAL_FREQ % DIRECT_BATCH_STEPS)) -ne 0 ] || [ "$DIRECT_EVAL_EVERY" -lt 1 ]; then
    echo "EVAL_FREQ must be a positive multiple of the direct pass size" >&2
    exit 2
fi
if [ $((EVAL_FREQ % PPO_ITERATION_STEPS)) -ne 0 ]; then
    echo "EVAL_FREQ must be divisible by the PPO collection size" >&2
    exit 2
fi
if [ $((TRAINING_STEPS % EVAL_FREQ)) -ne 0 ]; then
    echo "TRAINING_STEPS must be divisible by EVAL_FREQ" >&2
    exit 2
fi
if [ $((ROLLOUT_STEPS % TRUNCATION_STEPS)) -ne 0 ]; then
    echo "ROLLOUT_STEPS must be divisible by TRUNCATION_STEPS" >&2
    exit 2
fi
if [ "$REQUIRE_MATCHED_OPTIMIZATION" -eq 1 ] && \
    { [ "$DIRECT_UPDATES" -ne "$PPO_UPDATES" ] || \
      [ "$DIRECT_GRADIENT_STEPS" -ne "$PPO_GRADIENT_STEPS" ] || \
      [ "$TRUNCATION_STEPS" -ne "$PPO_NUM_STEPS" ]; }; then
    echo "optimizer updates, gradient batch, and temporal horizon must match" >&2
    exit 2
fi

echo "method=$METHOD seed=$SEED training_steps=$TRAINING_STEPS eval_freq=$EVAL_FREQ"
echo "direct: updates=$DIRECT_UPDATES gradient_steps=$DIRECT_GRADIENT_STEPS"
echo "ppo: updates=$PPO_UPDATES gradient_steps=$PPO_GRADIENT_STEPS"
if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

module purge 2>/dev/null || true
source "${VENV:-.venv}/bin/activate"

# Concurrent jobs compile byte-identical graphs, so avoid shared NFS cache
# write contention by assigning one cache directory per job.
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax-policy-compare-${SLURM_JOB_ID:-solo}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True ${XLA_EXTRA:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi -L || true
uv run --no-sync python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

if [ "$METHOD" = "direct_backprop" ]; then
    srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
        uv run --no-sync python experiments/studies/train_backprop_policy.py \
        --env "$ENV_SETUP" \
        --backend "$BACKEND" \
        --reward "$REWARD" \
        --variant "$VARIANT" \
        --disruption-penalty "$DISRUPTION_PENALTY" \
        --rollout-steps "$ROLLOUT_STEPS" \
        --truncation-steps "$TRUNCATION_STEPS" \
        --num-rollouts "$NUM_ROLLOUTS" \
        --eval-rollouts "$EVAL_ROLLOUTS" \
        --eval-seed "$EVAL_SEED" \
        --eval-every-passes "$DIRECT_EVAL_EVERY" \
        --iters "$DIRECT_PASSES" \
        --learning-rate "$DIRECT_LR" \
        --grad-clip "$DIRECT_GRAD_CLIP" \
        --hidden-sizes "$HIDDEN_1" "$HIDDEN_2" \
        --seed "$SEED" \
        --out "outputs/policy_compare_direct_${WANDB_GROUP}_seed${SEED}.npz" \
        --wandb \
        --wandb-mode "$WANDB_MODE" \
        --wandb-group "$WANDB_GROUP" \
        --wandb-entity "$WANDB_ENTITY"
else
    PPO_NORMALIZATION_ARGS=()
    if [ "$PPO_NORMALIZE_OBSERVATIONS" -eq 0 ]; then
        PPO_NORMALIZATION_ARGS+=(--ppo.no-normalize-observations)
    fi
    if [ "$PPO_NORMALIZE_REWARDS" -eq 0 ]; then
        PPO_NORMALIZATION_ARGS+=(--ppo.no-normalize-rewards)
    fi
    srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
        uv run --no-sync python scripts/train_ppo.py \
        --env.env-setup "$ENV_SETUP" \
        --env.backend "$BACKEND" \
        --env.reward "$REWARD" \
        --env.variant "$VARIANT" \
        --env.disruption-penalty "$DISRUPTION_PENALTY" \
        --env.eval-n-envs "$EVAL_ROLLOUTS" \
        --env.eval-seed "$EVAL_SEED" \
        --env.deterministic-eval \
        --env.force-minimal-callback \
        --seed "$SEED" \
        --num-seeds 1 \
        --ppo.total-timesteps "$TRAINING_STEPS" \
        --ppo.num-envs "$PPO_NUM_ENVS" \
        --ppo.num-steps "$PPO_NUM_STEPS" \
        --ppo.num-epochs "$PPO_NUM_EPOCHS" \
        --ppo.num-minibatches "$PPO_NUM_MINIBATCHES" \
        --ppo.eval-freq "$EVAL_FREQ" \
        --ppo.learning-rate "$PPO_LR" \
        --ppo.gamma "$PPO_GAMMA" \
        --ppo.gae-lambda "$PPO_GAE_LAMBDA" \
        --ppo.ent-coef "$PPO_ENT_COEF" \
        --ppo.hidden-sizes "$HIDDEN_1" "$HIDDEN_2" \
        --ppo.activation "$ACTIVATION" \
        --ppo.residual-policy \
        --ppo.initial-log-std "$PPO_INITIAL_LOG_STD" \
        "${PPO_NORMALIZATION_ARGS[@]}" \
        --wandb.mode "$WANDB_MODE" \
        --wandb.group "$WANDB_GROUP" \
        --wandb.entity "$WANDB_ENTITY"
fi
