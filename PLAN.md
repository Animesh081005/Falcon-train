# Implementation plan and feasibility conclusion

## Feasibility

Training is technically and legally possible. All three checkpoints are
Apache-2.0 and distributed as PyTorch-compatible safetensors. The published
repository is labelled inference-only because it omits the optimizer, training
loop, dataset mixture, and losses—not because the model is non-differentiable.
Its standard forward path returns vocabulary logits, and the authors state that
FlexAttention is used for inference and training.

What cannot currently be reproduced from public artifacts is TII's original
pretraining: the exact corpus, sampling weights, augmentations, schedule, and
training infrastructure are not published. This project therefore supports
supervised fine-tuning, not claimed reproduction.

## Architecture decision

1. **Falcon-OCR (selected):** 22 layers, 768 hidden size, about 300M parameters,
   early-fusion image and text tokens, pretrained specifically for OCR. It has
   no coordinate/segmentation heads. Fine-tune its language head to emit both
   word text and quantized boxes.
2. **Falcon-Perception-300M:** same lightweight backbone scale with coordinate
   and size heads, trained for open-vocabulary object detection. Using those
   heads would require defining dense word queries, matching predictions to
   ordered transcripts, adding recognition supervision, and modifying
   generation. Useful for a later multi-head experiment, but higher risk.
3. **Falcon-Perception:** 28 layers/1024 hidden plus coordinate, size, and mask
   heads. Strongest spatial machinery, but needlessly expensive and still lacks
   a published dense OCR-word joint objective.

## Output and loss

- Input: one document image, an explicit word-box instruction, and Falcon-OCR's
  pretrained `<|OCR_PLAIN|>` task token.
- Output: repeated `<word>…</word><box>x0,y0,x1,y1</box>` records.
- Coordinates: original-pixel boxes quantized to `[0,1000]` before training and
  dequantized after generation.
- Loss: next-token cross entropy only on the target response. Image tokens and
  prompt tokens have label `-100`.
- This is genuinely one-model inference: box localization is learned as token
  generation. No layout detector is used.

## Execution phases and gates

1. Validate manifests and reading order; create train/dev/test splits by source
   document, never by page, to prevent leakage.
2. Overfit 4–16 pages. Gate: valid grammar and near-perfect recovery on those
   pages. Failure here indicates a pipeline bug, not insufficient data.
3. Warm up with lower layers frozen, then unfreeze and full-fine-tune.
4. Evaluate exact/case-folded word accuracy, recognition edit similarity, box
   IoU, and joint correct-word-at-IoU-0.5 F1. Gate production on joint F1 and
   parse-valid rate.
5. Stress-test tiny text, scans, rotations, multilingual text, long pages, and
   unseen templates. Adjust image resolution and data mixture based on slices.
6. Export the best safetensors checkpoint with the unchanged tokenizer and run
   deterministic greedy decoding. Add constrained decoding only after baseline
   measurement if malformed records remain material.

## Data guidance

Use real word-box datasets licensed for the deployment domain plus synthetic
rendered pages. Normalize polygons to enclosing `xyxy` boxes unless polygon
output is a requirement. Preserve punctuation consistently. Define whether
spaces, ligatures, hyphenated line breaks, and rotated words are individual
tokens before annotation; inconsistent policy creates an artificial ceiling.

Suggested starting scale is 100k–1M diverse page images, but domain similarity
matters more than a universal count. Always begin with the overfit gate and
learning curves rather than committing to a fixed data number.

## Known risks and mitigations

- Long structured responses consume context: tile pages or group into regions
  during training while retaining a single model at inference.
- Numeric coordinates use the existing tokenizer: consider adding dedicated
  coordinate tokens only after establishing this checkpoint-compatible baseline.
- Autoregressive hallucination/duplication: increase hard negatives, use stable
  reading order, and optionally constrain the grammar during decode.
- Tiny text: raise effective input resolution and include tiny-text examples;
  sequence length alone does not restore lost pixels.
- Multi-GPU behavior: launch through Accelerate; the code retains DDP forward
  calls and accesses backbone metadata through the wrapped module.
