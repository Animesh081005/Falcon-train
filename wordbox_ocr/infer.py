from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from falcon_perception import load_from_hf_export
from falcon_perception.batch_inference import BatchInferenceEngine

from .format import parse, parse_validity
from .modeling import prepare_inference_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--min-dimension", type=int, default=256)
    p.add_argument("--max-dimension", type=int, default=1024)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required by the upstream FlexAttention implementation")
    model, tokenizer, model_args = load_from_hf_export(hf_local_dir=args.checkpoint)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    with Image.open(args.image) as source_image:
        source_image = source_image.convert("RGB")
        width, height = source_image.size
        batch = prepare_inference_batch(
            tokenizer, model_args, source_image,
            min_dimension=args.min_dimension, max_dimension=args.max_dimension,
        )
    batch = {k: v.cuda() for k, v in batch.items()}
    engine = BatchInferenceEngine(model, tokenizer)
    prompt_length = batch["tokens"].shape[1]
    output_ids, _ = engine.generate(
        **batch, max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id], task="ocr_plain",
    )
    # The upstream batch engine returns prompt + generated tokens in a padded row.
    generated = output_ids[0, prompt_length:].detach().cpu().tolist()
    generated = [x for x in generated if x != tokenizer.pad_token_id]
    if tokenizer.eos_token_id in generated:
        generated = generated[:generated.index(tokenizer.eos_token_id)]
    raw = tokenizer.decode(generated)
    words = parse(raw, width, height)
    result = {"image": args.image, "width": width, "height": height,
              "parse_valid": bool(parse_validity(raw)), "raw": raw,
              "words": [{"text": x.text, "bbox": list(x.bbox)} for x in words]}
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
