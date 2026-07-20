from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from falcon_perception import load_from_hf_export
from falcon_perception.batch_inference import BatchInferenceEngine

from .format import FORMAT_V1, FORMAT_V2, parse, parse_validity
from .modeling import prepare_inference_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-words", type=int, default=64,
                   help="Hard record cap; raise for dense pages or use page tiling")
    p.add_argument("--format-version", choices=(FORMAT_V1, FORMAT_V2),
                   help="Override checkpoint output format (normally auto-detected)")
    p.add_argument("--min-dimension", type=int, default=256)
    p.add_argument("--max-dimension", type=int, default=1024)
    args = p.parse_args()
    if args.max_new_tokens < 1 or args.max_words < 1:
        raise SystemExit("--max-new-tokens and --max-words must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required by the upstream FlexAttention implementation")
    model, tokenizer, model_args = load_from_hf_export(hf_local_dir=args.checkpoint)
    config_path = Path(args.checkpoint).resolve() / "config.json"
    checkpoint_config = (json.loads(config_path.read_text(encoding="utf-8"))
                         if config_path.exists() else {})
    # Older checkpoints used xyxy. Missing metadata therefore means v1, not the
    # newest training format, so existing checkpoints stay testable.
    format_version = args.format_version or checkpoint_config.get("wordbox_format", FORMAT_V1)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    with Image.open(args.image) as source_image:
        source_image = source_image.convert("RGB")
        width, height = source_image.size
        batch = prepare_inference_batch(
            tokenizer, model_args, source_image,
            min_dimension=args.min_dimension, max_dimension=args.max_dimension,
            format_version=format_version,
        )
    batch = {k: v.cuda() for k, v in batch.items()}
    engine = BatchInferenceEngine(model, tokenizer)
    prompt_length = batch["tokens"].shape[1]
    stop_ids = list(dict.fromkeys(
        token_id for token_id in (
            tokenizer.eos_token_id,
            getattr(tokenizer, "end_of_query_token_id", None),
        ) if token_id is not None
    ))
    record_end_ids = tokenizer.encode("</box>")
    output_ids, _ = engine.generate(
        **batch, max_new_tokens=args.max_new_tokens,
        stop_token_ids=stop_ids, task="ocr_plain",
        record_end_token_ids=record_end_ids, max_records=args.max_words,
        stop_on_repeated_record_cycle=True,
    )
    # The upstream batch engine returns prompt + generated tokens in a padded row.
    generated = output_ids[0, prompt_length:].detach().cpu().tolist()
    generated = [x for x in generated if x != tokenizer.pad_token_id]
    stop_positions = [generated.index(token_id) for token_id in stop_ids if token_id in generated]
    if stop_positions:
        generated = generated[:min(stop_positions)]
    raw = tokenizer.decode(generated)
    words = parse(raw, width, height, format_version)
    result = {"image": args.image, "width": width, "height": height,
              "format_version": format_version,
              "parse_valid": bool(parse_validity(raw, format_version)), "raw": raw,
              "words": [{"text": x.text, "bbox": list(x.bbox)} for x in words]}
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
