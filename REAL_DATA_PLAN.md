# Real-data and localization plan

## What the observed checkpoint is telling us

The text path has adapted sooner than localization. The earlier target encoded
four variable-width `xyxy` decimal strings and trained one ordinary EOS token at
the end of a much longer sequence. Easy markup/text tokens dominated the scalar
loss, and even the reported bounding-box loss included the easy `<box>` tags.
Consequently, a loss such as `0.48` could coexist with weak geometry and a weak
probability of stopping.

Version 2 of this repository makes the following coordinated changes:

- emits fixed-width normalized `center_x,center_y,width,height` fields;
- weights only coordinate payload tokens as box tokens;
- trains Falcon's native `<|end_of_query|>` terminal token at 8x weight;
- reports separate text, coordinate, and stop losses;
- selects `best` with `bbox + 0.25 * text + 0.25 * stop`, rather than aggregate
  token loss alone;
- stops inference on EOS or end-of-query, after a configurable record cap, or
  when a repeated one-word/phrase cycle is detected;
- reads `wordbox_format` from each checkpoint so v1 checkpoints remain usable.

This format change requires a new v2 training run. Do not resume a v1 checkpoint
and expect its already-learned output grammar to switch instantly; initialize
from Falcon-OCR for the cleanest experiment. Old v1 checkpoints can still be
tested and receive the new inference stopping guards.

## Canonical data contract

Every source is converted to one JSONL row per image:

```json
{"image":"/data/page.png","width":1240,"height":1754,"source":"textocr","source_id":"123","words":[{"text":"Invoice","bbox":[95,83,247,126]}]}
```

Boxes are tight pixel `xyxy` rectangles. `words` must be in the desired reading
order. Split by original document, capture session, vendor/template, or source
ID before creating pages/crops; never randomly split near-duplicate pages.

Available adapters:

```bash
# Scene text, signs, packaging, and many title-like images
python scripts/convert_textocr.py \
  --annotations /data/textocr/TextOCR_0.1_train.json \
  --images-root /data/textocr/images \
  --output /data/wordbox/textocr_train.jsonl

# SROIE/ICDAR and the 8-point text-file format used by this repository
python scripts/convert_icdar_dataset.py \
  --images-root /data/sroie/train/img \
  --annotations-root /data/sroie/train/box \
  --source sroie --output /data/wordbox/sroie_train.jsonl
```

For CORD, invoices, boarding passes, or internal annotations, export the same
canonical schema. Preserve authoritative annotation order when supplied. If it
is not supplied, the adapters infer top-to-bottom/left-to-right raster order;
multi-column documents should receive an explicit order during annotation or a
domain-specific converter.

## Recommended corpus

Use the source licenses only after they have been approved for the intended
product. A sensible initial mix is:

- 35% real documents: internal invoices, forms, boarding passes, scans, and
  digital PDFs with verified word boxes;
- 25% real scene/title text: TextOCR, plus your licensed book-cover/title data;
- 15% receipts: CORD and/or SROIE if their terms fit the project;
- 25% synthetic: the existing clean pages plus dense pages, perspective,
  rotation, shadows, blur, compression, folds, and background clutter.

Public data cannot replace a representative, human-checked validation/test set.
Create at least 500–1,000 held-out images per important domain. Include empty or
very-low-text images in a separate presence/calibration suite, even if the
current trainer filters empty targets.

Create a deterministic weighted mixture:

```bash
python scripts/mix_manifests.py \
  --source synthetic_ocr_data_for_falcon/manifests/train.jsonl:0.30 \
  --source /data/wordbox/textocr_train.jsonl:0.30 \
  --source /data/wordbox/documents_train.jsonl:0.25 \
  --source /data/wordbox/receipts_train.jsonl:0.15 \
  --num-samples 200000 --seed 17 \
  --output mixed_ocr_data/manifests/train.jsonl
```

Validation must be a direct concatenation of held-out sources, not oversampled.
Keep per-domain manifests as well as the combined manifest so regressions are
visible.

## Training curriculum

1. **V2 pipeline proof:** train 20 samples until every training image has exact
   text, valid structure, correct count, and high box IoU. A low CE is not the
   acceptance criterion.
2. **Synthetic warm-up:** 20k–50k spatially diverse pages, one epoch, LR
   `2e-5`, coordinate weight 4, stop weight 8. This teaches the new grammar.
3. **Mixed adaptation:** 30% synthetic / 70% real, 3–5 epochs, LR `1e-5` to
   `2e-5`, resolution 1024–1536 when VRAM permits. Oversample minority domains,
   not individual validation examples.
4. **Real refinement:** real-only or 10% synthetic, 1–2 epochs, LR `2e-6` to
   `5e-6`, cosine decay. Stop if a domain's end-to-end F1 degrades.
5. **Hard-example pass:** oversample errors involving tiny words, tall/narrow
   glyphs, extreme aspect ratios, perspective, and repeated tokens. Audit labels
   before treating them as hard examples.

For speed, disable gradient checkpointing when the batch already fits, reduce
the image cap for sparse pages, and use multi-GPU data parallelism. A nominal
batch of 64 does not remove the cost of processing 64 full image-token grids;
pixels and sequence length, not only optimizer-step count, dominate runtime.

## Acceptance metrics

Report these for every domain and for size/aspect-ratio buckets:

- word recognition precision/recall/F1;
- detection precision/recall/AP at IoU 0.5 and mean IoU of matched words;
- end-to-end word F1 requiring correct text and IoU >= 0.5;
- exact word-count accuracy, parse-valid rate, and terminal-token success rate;
- width error, height error, center error, and predicted/true width and height
  ratios. These expose the “too wide, too short” failure directly.

Use a small generation-based validation subset every epoch and the complete
held-out suite after training. Token CE is a diagnostic, not the final model
selection metric.

## Falcon-Perception decision

The same early-fusion family makes transfer feasible, but the current trainer is
**not** a correct Falcon-Perception trainer. Falcon-Perception emits structured
`<coord> -> <size> -> <seg>` steps and learns continuous coordinate/size heads;
Falcon-OCR emits vocabulary logits and was trained specifically for glyph-level
OCR. Loading the 600M checkpoint into this vocabulary-only objective would leave
its localization heads unsupervised and sacrifice the stronger OCR
initialization.

Recommended order:

1. Finish the corrected Falcon-OCR v2 baseline on real data.
2. If geometry plateaus, build a hybrid single model: initialize the 300M shared
   backbone/text head from Falcon-OCR, attach the coordinate and size heads from
   Falcon-Perception-300M, and train ordered records containing word text plus
   native `<coord>/<size>` targets. Optimize text CE, coordinate loss, size loss,
   and stop loss jointly. No segmentation head is needed.
3. Only then test the 600M Falcon-Perception backbone. Its 28-layer, 1024-wide
   weights are not shape-compatible with the 22-layer, 768-wide OCR backbone, so
   it needs a deliberate OCR+geometry training recipe and materially more
   compute. Larger capacity may help after sufficient diverse real data, but it
   does not inherently fix stopping or tight word boundaries.

The 300M hybrid is the highest-value next architecture experiment because it
retains the OCR-sized backbone and uses the family’s purpose-built continuous
geometry interface. Compare it against v2 on exactly the same held-out splits.
