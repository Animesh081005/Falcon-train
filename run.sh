#!/usr/bin/env bash
# Run Falcon word-box OCR in either end-to-end sanity or complete training mode.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run.sh --sanity    Train for 20 epochs on 20 train + 5 validation samples
  ./run.sh --train     Train the complete dataset with production defaults

Environment overrides:
  DATASET_DIR       Dataset root (default: synthetic_ocr_data_for_falcon)
  MODEL_ID          Hugging Face model id (default: tiiuae/Falcon-OCR)
  MODEL_DIR         Optional local Falcon-OCR checkpoint directory
  OUTPUT_ROOT       Checkpoint parent directory (default: runs)
  BATCH_SIZE        Per-GPU batch size (default: 1)
  GRAD_ACCUM        Production gradient accumulation (default: 16)
  TRAIN_EPOCHS      Production epochs (default: 5)
  LEARNING_RATE     Production LR (default: 2e-5)
  MAX_DIMENSION     Production image dimension cap (default: 1024)
  NUM_WORKERS       DataLoader processes (default: 8)
  SAVE_EVERY_STEPS  Periodic last-checkpoint interval (default: 1000)
  BBOX_LOSS_WEIGHT  Bounding-box token loss multiplier (default: 4.0)
  GRADIENT_CHECKPOINTING  1 saves VRAM; 0 is faster (default: 1)
  MIXED_PRECISION   bf16 or no (default: bf16)
  ACCELERATE_ARGS   Extra arguments before `-m wordbox_ocr.train`

Examples:
  ./run.sh --sanity
  NUM_WORKERS=16 GRAD_ACCUM=32 ./run.sh --train
  ACCELERATE_ARGS="--multi_gpu --num_processes 4" ./run.sh --train
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  --sanity|sanity) MODE="sanity" ;;
  --train|train) MODE="train" ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown mode: $1" >&2; usage >&2; exit 2 ;;
esac

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DATASET_DIR="${DATASET_DIR:-synthetic_ocr_data_for_falcon}"
MODEL_ID="${MODEL_ID:-tiiuae/Falcon-OCR}"
MODEL_DIR="${MODEL_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_DIMENSION="${MAX_DIMENSION:-1024}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
BBOX_LOSS_WEIGHT="${BBOX_LOSS_WEIGHT:-4.0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
TRAIN_MANIFEST="${DATASET_DIR}/manifests/train.jsonl"
VALIDATION_MANIFEST="${DATASET_DIR}/manifests/validation.jsonl"

command -v python >/dev/null 2>&1 || { echo "python is not available" >&2; exit 1; }
command -v accelerate >/dev/null 2>&1 || {
  echo "accelerate is not installed. Run: pip install -r requirements.txt" >&2
  exit 1
}
[[ -s "$TRAIN_MANIFEST" ]] || {
  echo "Missing training manifest: $TRAIN_MANIFEST" >&2
  echo "Generate it with: python scripts/generate_synthetic_dataset.py --output '$DATASET_DIR' --num-images 100000" >&2
  exit 1
}
[[ -s "$VALIDATION_MANIFEST" ]] || {
  echo "Missing validation manifest: $VALIDATION_MANIFEST" >&2
  exit 1
}

MIXED_PRECISION="$MIXED_PRECISION" python - <<'PY'
import importlib.util
import os
import sys
try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed")
if not torch.cuda.is_available():
    sys.exit("CUDA is unavailable; Falcon training requires an NVIDIA GPU")
if os.environ["MIXED_PRECISION"] == "bf16" and not torch.cuda.is_bf16_supported():
    sys.exit("GPU does not support BF16; rerun with MIXED_PRECISION=no")
if importlib.util.find_spec("falcon_perception") is None:
    sys.exit("falcon_perception is not installed; run: pip install -e third_party/falcon-perception")
print(f"CUDA OK: {torch.cuda.get_device_name(0)}; GPUs={torch.cuda.device_count()}")
PY

train_rows=$(wc -l < "$TRAIN_MANIFEST")
validation_rows=$(wc -l < "$VALIDATION_MANIFEST")

if [[ "$MODE" == "sanity" ]]; then
  if (( train_rows < 20 || validation_rows < 5 )); then
    echo "Sanity mode needs at least 20 train and 5 validation rows; found ${train_rows}/${validation_rows}." >&2
    exit 1
  fi
  EPOCHS=20
  MODE_GRAD_ACCUM=4
  MODE_LR="2e-5"
  MODE_MAX_DIM=768
  OUTPUT_DIR="${OUTPUT_ROOT}/sanity"
  SAMPLE_ARGS=(--max-train-samples 20 --max-validation-samples 5)
  echo "SANITY: 20 train samples, 5 validation samples, 20 epochs"
else
  EPOCHS="$TRAIN_EPOCHS"
  MODE_GRAD_ACCUM="$GRAD_ACCUM"
  MODE_LR="$LEARNING_RATE"
  MODE_MAX_DIM="$MAX_DIMENSION"
  OUTPUT_DIR="${OUTPUT_ROOT}/falcon-wordbox"
  SAMPLE_ARGS=()
  echo "TRAIN: ${train_rows} train samples, ${validation_rows} validation samples, ${EPOCHS} epochs"
fi

PRECISION_ARGS=()
if [[ "$MIXED_PRECISION" == "bf16" ]]; then
  PRECISION_ARGS+=(--bf16)
elif [[ "$MIXED_PRECISION" != "no" ]]; then
  echo "MIXED_PRECISION must be 'bf16' or 'no'" >&2
  exit 2
fi

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  CHECKPOINTING_ARGS=(--gradient-checkpointing)
elif [[ "$GRADIENT_CHECKPOINTING" == "0" ]]; then
  CHECKPOINTING_ARGS=(--no-gradient-checkpointing)
else
  echo "GRADIENT_CHECKPOINTING must be 1 or 0" >&2
  exit 2
fi

# Word splitting is intentional: ACCELERATE_ARGS is an advanced shell override.
# shellcheck disable=SC2206
EXTRA_ACCELERATE_ARGS=( ${ACCELERATE_ARGS:-} )

if [[ -n "$MODEL_DIR" ]]; then
  [[ -s "$MODEL_DIR/model.safetensors" ]] || {
    echo "MODEL_DIR does not contain model.safetensors: $MODEL_DIR" >&2
    exit 1
  }
  MODEL_ARGS=(--resume "$MODEL_DIR")
else
  MODEL_ARGS=(--model-id "$MODEL_ID")
fi

accelerate launch "${EXTRA_ACCELERATE_ARGS[@]}" -m wordbox_ocr.train \
  "${MODEL_ARGS[@]}" \
  --train "$TRAIN_MANIFEST" \
  --validation "$VALIDATION_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$MODE_GRAD_ACCUM" \
  --learning-rate "$MODE_LR" \
  --weight-decay 0.1 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --max-seq-len 8192 \
  --min-dimension 256 \
  --max-dimension "$MODE_MAX_DIM" \
  --num-workers "$NUM_WORKERS" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --bbox-loss-weight "$BBOX_LOSS_WEIGHT" \
  "${CHECKPOINTING_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}"

echo "Completed ${MODE} run. Best checkpoint: ${OUTPUT_DIR}/best"
