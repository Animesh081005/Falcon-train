# Falcon Word-Box OCR

Fine-tunes `tiiuae/Falcon-OCR` (300M) to generate OCR words and word-level
bounding boxes from one model:

```text
<word>Invoice</word><box>0138,0060,0123,0025</box>
```

The box fields are normalized `center_x,center_y,width,height`. Inference
returns pixel `xyxy` boxes.

## 1. Linux GPU setup

Requirements: NVIDIA GPU, Python 3.11+, and a compatible CUDA/PyTorch build.

```bash
git clone <OFFICE_REPOSITORY_URL> Falcon-train
cd Falcon-train

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Change cu128 if the server uses a different supported CUDA build.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e third_party/falcon-perception
pip install -e .
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
```

The vendored `falcon-perception` package contains the shared implementation
needed to load Falcon-OCR. Installing it does not train Falcon-Perception.

## 2. Generate and validate synthetic data

```bash
sudo apt-get update
sudo apt-get install -y fonts-dejavu-core fonts-liberation fonts-freefont-ttf

python scripts/generate_synthetic_dataset.py \
  --output synthetic_ocr_data_for_falcon \
  --num-images 100000 \
  --workers 32

python -m wordbox_ocr.validate_data \
  --manifest synthetic_ocr_data_for_falcon/manifests/train.jsonl

python scripts/analyze_dataset.py \
  --manifest synthetic_ocr_data_for_falcon/manifests/train.jsonl
```

This creates `train.jsonl`, `validation.jsonl`, and `test.jsonl` under
`synthetic_ocr_data_for_falcon/manifests/`.

For a quick generator test, use `--num-images 30 --workers 2`.

## 3. Mandatory sanity training

Sanity mode trains 20 samples for 20 epochs and validates on 5 samples.

```bash
unset MODEL_DIR

MIXED_PRECISION=bf16 \
BATCH_SIZE=2 \
NUM_WORKERS=8 \
GRADIENT_CHECKPOINTING=0 \
SAVE_EVERY_STEPS=20 \
./run.sh --sanity
```

If memory is insufficient, use `BATCH_SIZE=1` and
`GRADIENT_CHECKPOINTING=1`. If BF16 is unsupported, use
`MIXED_PRECISION=no`.

Checkpoints:

```text
runs/sanity/best
runs/sanity/last
```

Inspect the selected checkpoint:

```bash
readlink -f runs/sanity/best
cat runs/sanity/best/trainer_state.json
```

## 4. Inference

```bash
python -m wordbox_ocr.infer \
  --checkpoint runs/sanity/best \
  --image /path/to/test.jpg \
  --output prediction.json \
  --max-words 64 \
  --max-new-tokens 2048 \
  --max-dimension 1024

python -m json.tool prediction.json
```

Verify that `parse_valid` is `true`, the word count is reasonable, and the
boxes align with the image.

## 5. Full training

Example with effective batch size 32:

```bash
unset MODEL_DIR

MIXED_PRECISION=bf16 \
BATCH_SIZE=8 \
GRAD_ACCUM=4 \
TRAIN_EPOCHS=5 \
MAX_DIMENSION=1024 \
NUM_WORKERS=16 \
GRADIENT_CHECKPOINTING=0 \
SAVE_EVERY_STEPS=500 \
./run.sh --train 2>&1 | tee full-training.log
```

For a lower-memory GPU, use `BATCH_SIZE=2`, `GRAD_ACCUM=16`, and
`GRADIENT_CHECKPOINTING=1`.

Production checkpoints:

```text
runs/falcon-wordbox/best
runs/falcon-wordbox/last
```

`best` uses a localization-and-stop-aware validation score. `last` is the most
recent recoverable checkpoint. Both remain safe if training is interrupted.

Initialize another run from a compatible Falcon-OCR checkpoint with:

```bash
MODEL_DIR=/absolute/path/to/checkpoint ./run.sh --train
```

Do not resume a v2 `center_x,center_y,width,height` run from an older v1
`x0,y0,x1,y1` fine-tuned checkpoint.

## 6. Real-data quick reference

Canonical JSONL row:

```json
{"image":"/data/page.png","width":1240,"height":1754,"words":[{"text":"Invoice","bbox":[95,83,247,126]}]}
```

Available preparation commands:

```bash
# TextOCR
python scripts/convert_textocr.py \
  --annotations /data/textocr/train.json \
  --images-root /data/textocr/images \
  --output /data/wordbox/textocr_train.jsonl

# ICDAR/SROIE quadrilateral annotations
python scripts/convert_icdar_dataset.py \
  --images-root /data/sroie/images \
  --annotations-root /data/sroie/annotations \
  --source sroie \
  --output /data/wordbox/sroie_train.jsonl

# Mix manifests by sampling weight
python scripts/mix_manifests.py \
  --source synthetic_ocr_data_for_falcon/manifests/train.jsonl:0.30 \
  --source /data/wordbox/textocr_train.jsonl:0.30 \
  --source /data/wordbox/documents_train.jsonl:0.40 \
  --num-samples 200000 \
  --output mixed_ocr_data/manifests/train.jsonl

DATASET_DIR=mixed_ocr_data ./run.sh --train
```

Add `mixed_ocr_data/manifests/validation.jsonl` before training.

## More information

- [PLAN.md](PLAN.md): model and architecture decision
- [REAL_DATA_PLAN.md](REAL_DATA_PLAN.md): real-data curriculum and evaluation
- `./run.sh --help`: all training overrides

The current trainer supports Falcon-OCR. Falcon-Perception 300M/600M requires a
separate objective for its coordinate and size heads and is intentionally
rejected by this trainer.
