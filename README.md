# Falcon Word-Box OCR

End-to-end supervised fine-tuning of `tiiuae/Falcon-OCR` to emit OCR words and
word-level bounding boxes from a **single model**.

## Decision

Use **Falcon-OCR (300M)** as the initialization. It has the same small backbone
size as Falcon-Perception-300M, but is already specialized for document text.
The Perception models' coordinate heads are useful for prompted object
detection, not dense ordered word transcription. This project instead teaches
Falcon-OCR a constrained autoregressive target:

```text
<word>hello</word><box>31,92,174,131</box>
<word>world</word><box>190,92,331,131</box>
```

Coordinates are integers in `[0, 1000]`, normalized relative to the original
image. At inference they are converted back to pixel coordinates. The model,
not a second detector, generates both fields.

The official repository is described as inference-only, but training is
possible: the Apache-2.0 checkpoints contain ordinary safetensors, the model is
a PyTorch `nn.Module`, and its full-sequence forward pass returns logits. This
repository supplies the missing teacher-forcing data path, loss, optimizer,
checkpointing, evaluation, and structured inference. It is a new fine-tuning
recipe, not an official TII training recipe.

## Data format

JSONL, one page/image per line. Boxes are pixel `xyxy` coordinates and words
must be in desired reading order:

```json
{"image":"images/page_001.png","width":1240,"height":1754,"words":[{"text":"Invoice","bbox":[95,83,247,126]},{"text":"No.","bbox":[260,83,311,126]}]}
```

`width` and `height` are optional and validated against the image when present.
Empty/whitespace words are discarded. Invalid boxes fail early.

## Generate 100k synthetic pages

The repository includes a deterministic, multiprocessing Linux generator. It
creates JPEG pages, the quadrilateral `.txt` annotations, and train-ready JSONL
manifests in one pass. The dataset directory is ignored by Git.

```bash
# Ubuntu/Debian: make sure several Latin fonts are present
sudo apt-get update
sudo apt-get install -y fonts-dejavu-core fonts-liberation fonts-freefont-ttf

python scripts/generate_synthetic_dataset.py \
  --output synthetic_ocr_data_for_falcon \
  --num-images 100000 \
  --vocabulary synthetic_words.txt \
  --workers 32
```

The defaults reproduce the requested 9–10 words per image. For a more capable
full-page model, create an additional dense subset with
`--min-words 24 --max-words 80` and mix its manifest into training.
Sparse pages use vertically stratified lines covering the full image; this is
important because a top-heavy synthetic layout causes coordinate collapse even
when transcription loss is low. `dataset_info.json` records layout version
`spatially-balanced-v2`.

If fonts are stored with the repository or on a mounted volume, pass
`--font-dir /path/to/fonts`. Use a small run to verify the environment:

```bash
python scripts/generate_synthetic_dataset.py \
  --output /tmp/falcon_synthetic_smoke --num-images 20 --workers 2
python -m wordbox_ocr.validate_data \
  --manifest /tmp/falcon_synthetic_smoke/manifests/train.jsonl
```

Train directly from the generated manifests:

```bash
accelerate launch -m wordbox_ocr.train \
  --train synthetic_ocr_data_for_falcon/manifests/train.jsonl \
  --validation synthetic_ocr_data_for_falcon/manifests/validation.jsonl \
  --output-dir runs/falcon-wordbox --bf16 \
  --batch-size 1 --gradient-accumulation 16
```

Or use the checked launcher. Sanity mode runs the complete training and
checkpoint path for 20 epochs on exactly 20 training and 5 validation samples;
training mode uses the complete manifests and recommended defaults:

```bash
./run.sh --sanity
./run.sh --train
```

The production defaults are five epochs, BF16, per-GPU batch size 1, effective
batch size 16 through gradient accumulation, learning rate `2e-5`, weight decay
`0.1`, 3% warmup, gradient clipping at 1.0, gradient checkpointing, and a 1024
pixel image cap. Override hardware-sensitive values through environment
variables, for example `NUM_WORKERS=16 GRAD_ACCUM=32 ./run.sh --train`.

Bounding-box target tokens receive a default 4x loss weight so easy OCR text
cannot hide poor localization in the aggregate loss. For sparse 9–10-word pages,
`MAX_DIMENSION=640` is a useful speed/accuracy starting point. If VRAM permits,
`GRADIENT_CHECKPOINTING=0` avoids recomputation and is materially faster.

Before optimization, training evaluates and saves the initialized model. Every
completed epoch may atomically advance the `best` pointer based on token-weighted
validation loss, while `last` is updated every epoch and every 1,000 optimizer
steps. Checkpoints are written to immutable directories before either symlink is
changed, so interruption cannot partially overwrite the previous best model.

## GPU quick start

Requirements: Linux, NVIDIA GPU, Python 3.11+, PyTorch 2.5+, and a Hugging Face
token if the checkpoint ever becomes gated.

```bash
git clone <this-repository> falcon-wordbox-ocr
cd falcon-wordbox-ocr
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e third_party/falcon-perception

accelerate launch -m wordbox_ocr.train \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --output-dir runs/falcon-wordbox \
  --bf16 --batch-size 1 --gradient-accumulation 16
```

Smoke-test the complete data/model path before a long run:

```bash
python -m wordbox_ocr.validate_data --manifest data/train.jsonl
accelerate launch -m wordbox_ocr.train --train data/train.jsonl \
  --validation data/validation.jsonl --output-dir runs/smoke \
  --max-steps 2 --max-samples 4 --bf16
```

Inference:

```bash
python -m wordbox_ocr.infer \
  --checkpoint runs/falcon-wordbox/best \
  --image test.png --output prediction.json
```

## Recommended training stages

1. **Pipeline proof:** 100–1,000 clean synthetic pages, overfit a tiny subset,
   and require near-zero training loss plus valid parsing.
2. **Warm-up:** freeze the image projector and bottom transformer blocks for
   1–3k steps (`--freeze-layers 11`), then resume with all layers trainable and
   a lower learning rate.
3. **Full fine-tune:** mix real annotated pages with synthetic data. Preserve
   reading-order diversity, small text, rotations, scans, receipts, and tables.
4. **Validate:** report word recognition F1, box AP at IoU 0.5, end-to-end word
   F1 at IoU 0.5, normalized edit distance, and parse-valid rate. Select on
   end-to-end F1—not loss alone.

For a 24 GB GPU, start with batch 1, BF16, gradient checkpointing, and image
`--max-dimension 1024`. A 40–80 GB GPU can raise resolution/batch size. Tiny
text needs resolution, so tune `--max-dimension` before increasing sequence
length. Full-page pages with thousands of words may exceed the 8192-token
context; tile such pages during dataset preparation or reduce the grammar.

## Model comparison

| Model | Size / heads | Existing strength | Word-box OCR fit |
|---|---:|---|---|
| Falcon-OCR | 300M, no perception heads | OCR text, tables, formulas | **Best starting point**; teach serialized word boxes |
| Falcon-Perception-300M | 300M, coord/size heads, no segmentation | open-vocabulary detection | Alternative research path, but dense word ordering and transcription need new joint-head training |
| Falcon-Perception | larger, coord/size + segmentation heads | detection and masks | Excess compute and task mismatch |

The published `generate_with_layout` output is region-level and uses an
external PP-DocLayoutV3 detector. It does not meet the single-model word-box
objective.

## Repository layout

- `wordbox_ocr/`: dataset, grammar, training, evaluation, and inference
- `configs/`: reproducible example arguments
- `tests/`: CPU unit tests for serialization, parsing, and metrics
- `third_party/falcon-perception/`: vendored official inference implementation
  (Apache-2.0), with a state-dict-compatible training-only option that computes
  vocabulary logits only at supervised target positions; replace/update
  deliberately when upstream changes

## Important limitations

- The official tokenizer has no dedicated 1,000 coordinate tokens. Coordinates
  are represented by existing text tokens. This is immediately trainable and
  checkpoint-compatible, but a future tokenizer expansion may improve accuracy.
- Generated structured text is not mathematically guaranteed valid. Constrained
  decoding is a useful production follow-up; training reports parse-valid rate.
- Reading order comes from annotations. A model cannot learn a stable order if
  the manifest order is inconsistent.
