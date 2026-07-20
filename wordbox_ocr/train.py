from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from falcon_perception import load_from_hf_export

from .dataset import WordBoxDataset
from .modeling import Collator, forward_loss


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="tiiuae/Falcon-OCR")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-samples", type=int,
                   help="Deprecated shared cap for both splits")
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-validation-samples", type=int)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--min-dimension", type=int, default=256)
    p.add_argument("--max-dimension", type=int, default=1024)
    p.add_argument("--bbox-loss-weight", type=float, default=4.0)
    p.add_argument("--freeze-layers", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", help="Checkpoint directory/symlink used as model initialization")
    p.add_argument("--save-every-steps", type=int, default=1000,
                   help="Write a recoverable last checkpoint every N optimizer steps; 0 disables")
    return p.parse_args()


def _pointer_target(pointer: Path) -> Path | None:
    return pointer.resolve() if pointer.is_symlink() else None


def _atomic_pointer(pointer: Path, target: Path):
    """Atomically update a relative symlink while preserving legacy directories."""
    pointer.parent.mkdir(parents=True, exist_ok=True)
    if pointer.exists() and not pointer.is_symlink():
        legacy = pointer.parent / f"legacy-{pointer.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        pointer.rename(legacy)
    temporary = pointer.parent / f".{pointer.name}.next-{os.getpid()}"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(target, pointer.parent), target_is_directory=True)
    os.replace(temporary, pointer)


def _cleanup_checkpoints(output: Path):
    """Keep only checkpoints referenced by best and last."""
    retained = {_pointer_target(output / name) for name in ("best", "last")}
    retained.discard(None)
    checkpoint_root = output / "checkpoints"
    if not checkpoint_root.exists():
        return
    for model_file in checkpoint_root.glob("*/*/model.safetensors"):
        directory = model_file.parent.resolve()
        if directory not in retained:
            shutil.rmtree(directory)
    for run_dir in checkpoint_root.iterdir():
        if run_dir.is_dir() and not any(run_dir.iterdir()):
            run_dir.rmdir()


def save_checkpoint(accelerator, model, tokenizer, output: Path, run_id: str,
                    tag: str, state: dict, pointers: tuple[str, ...]):
    """Save to an immutable directory, then atomically move best/last pointers."""
    accelerator.wait_for_everyone()
    directory = output / "checkpoints" / run_id / tag
    if accelerator.is_main_process:
        directory.mkdir(parents=True, exist_ok=False)
        unwrapped = accelerator.unwrap_model(model)
        tensors = {k: v.detach().contiguous().cpu() for k, v in unwrapped.state_dict().items()}
        save_file(tensors, str(directory / "model.safetensors"))
        tokenizer.save_pretrained(directory)
        config = {
            "architectures": ["FalconOCRForCausalLM"],
            "model_type": "falcon_ocr",
            "wordbox_format": "wordbox-v1-normalized-1000",
            "base_model": state["model_id"],
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (directory / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
        for pointer in pointers:
            _atomic_pointer(output / pointer, directory)
        _cleanup_checkpoints(output)
    accelerator.wait_for_everyone()
    return directory


def evaluate(accelerator, model, tokenizer, validation_loader) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(6, dtype=torch.float64)
    with torch.no_grad():
        for batch in validation_loader:
            _, sample_stats = forward_loss(
                model, tokenizer, batch, return_sample_stats=True)
            gathered = accelerator.gather_for_metrics(sample_stats)
            totals += gathered.double().sum(dim=0).cpu()
    if totals[1] == 0 or totals[3] == 0 or totals[5] == 0:
        raise RuntimeError("Validation set contains no supervised target tokens")
    return {
        "weighted": (totals[0] / totals[1]).item(),
        "text": (totals[2] / totals[3]).item(),
        "bbox": (totals[4] / totals[5]).item(),
    }


def main():
    args = arguments()
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.epochs < 1:
        raise SystemExit("batch size, gradient accumulation, and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if args.save_every_steps < 0 or args.num_workers < 0:
        raise SystemExit("save interval and worker count cannot be negative")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise SystemExit("--warmup-ratio must be in [0, 1)")
    if not (0 < args.min_dimension <= args.max_dimension):
        raise SystemExit("image dimensions must satisfy 0 < min <= max")
    if args.bbox_loss_weight < 1.0:
        raise SystemExit("--bbox-loss-weight must be at least 1.0")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision="bf16" if args.bf16 else "no",
    )
    run_id_holder = [(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
                     if accelerator.is_main_process else None]
    broadcast_object_list(run_id_holder, from_process=0)
    run_id = run_id_holder[0]
    set_seed(args.seed)
    model, tokenizer, model_args = load_from_hf_export(
        hf_model_id=None if args.resume else args.model_id,
        hf_revision=args.model_revision,
        hf_local_dir=args.resume,
    )
    if not model_args.perception_heads and args.model_id != "tiiuae/Falcon-OCR":
        accelerator.print("Warning: checkpoint auto-detected as OCR despite a different model id")
    model.train()
    if args.gradient_checkpointing:
        # Upstream blocks are plain nn.Modules; non-reentrant checkpointing is injected per block.
        from torch.utils.checkpoint import checkpoint
        for block in model.layers.values():
            original = block.forward
            block.forward = (lambda *a, _f=original, **kw:
                             checkpoint(_f, *a, use_reentrant=False, **kw)
                             if torch.is_grad_enabled() else _f(*a, **kw))
    for i, block in enumerate(model.layers.values()):
        if i < args.freeze_layers:
            block.requires_grad_(False)

    train_limit = args.max_train_samples if args.max_train_samples is not None else args.max_samples
    validation_limit = (args.max_validation_samples
                        if args.max_validation_samples is not None else args.max_samples)
    train_ds = WordBoxDataset(args.train, train_limit)
    val_ds = WordBoxDataset(args.validation, validation_limit)
    accelerator.print(f"dataset: train={len(train_ds)} validation={len(val_ds)}")
    collator = Collator(tokenizer, model_args, args.max_seq_len,
                        args.min_dimension, args.max_dimension,
                        args.bbox_loss_weight)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collator,
                          pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collator,
                        pin_memory=True)
    # Prepare dataloaders before calculating steps: their lengths change when
    # Accelerate shards them across multiple GPUs.
    train_dl, val_dl = accelerator.prepare(train_dl, val_dl)
    updates_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation)
    total_steps = args.max_steps or args.epochs * updates_per_epoch
    warmup_steps = round(total_steps * args.warmup_ratio)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                      lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / max(warmup_steps, 1),
                                    max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    global_step = 0
    # Establish a valid best checkpoint before validation or the first optimizer
    # step. Even an early validation/data failure leaves a complete testable model.
    initial_state = {
        "epoch": 0, "global_step": 0, "val_loss": None,
        "model_id": args.model_id, "model_revision": args.model_revision,
        "run_id": run_id, "status": "initialized",
    }
    initial_directory = save_checkpoint(
        accelerator, model, tokenizer, output, run_id,
        "epoch-0000-step-00000000", initial_state, ("best", "last"),
    )
    initial_metrics = evaluate(accelerator, model, tokenizer, val_dl)
    best_loss = initial_metrics["weighted"]
    if accelerator.is_main_process:
        initial_state.update(val_loss=best_loss, best_val_loss=best_loss,
                             val_text_loss=initial_metrics["text"],
                             val_bbox_loss=initial_metrics["bbox"],
                             status="initial_validation_complete")
        temporary_state = initial_directory / ".trainer_state.next"
        temporary_state.write_text(json.dumps(initial_state, indent=2) + "\n")
        os.replace(temporary_state, initial_directory / "trainer_state.json")
    accelerator.wait_for_everyone()
    accelerator.print(
        f"initial validation weighted={best_loss:.5f} "
        f"text={initial_metrics['text']:.5f} bbox={initial_metrics['bbox']:.5f}; "
        "best checkpoint saved")
    progress = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)
    try:
        for epoch in range(args.epochs):
            model.train()
            for batch in train_dl:
                with accelerator.accumulate(model):
                    loss = forward_loss(model, tokenizer, batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    global_step += 1
                    progress.update(1)
                    progress.set_postfix(loss=f"{loss.item():.4f}")
                    if args.save_every_steps and global_step % args.save_every_steps == 0:
                        state = {
                            "epoch": epoch + 1, "global_step": global_step, "val_loss": None,
                            "model_id": args.model_id, "model_revision": args.model_revision,
                            "run_id": run_id, "status": "in_progress",
                        }
                        save_checkpoint(
                            accelerator, model, tokenizer, output, run_id,
                            f"periodic-step-{global_step:08d}", state, ("last",),
                        )
                if global_step >= total_steps:
                    break

            val_metrics = evaluate(accelerator, model, tokenizer, val_dl)
            val_loss = val_metrics["weighted"]
            accelerator.print(
                f"epoch={epoch + 1} step={global_step} weighted={val_loss:.5f} "
                f"text={val_metrics['text']:.5f} bbox={val_metrics['bbox']:.5f}")
            improved = val_loss < best_loss
            if improved:
                best_loss = val_loss
            state = {
                "epoch": epoch + 1, "global_step": global_step, "val_loss": val_loss,
                "val_text_loss": val_metrics["text"], "val_bbox_loss": val_metrics["bbox"],
                "best_val_loss": best_loss, "model_id": args.model_id,
                "model_revision": args.model_revision, "run_id": run_id,
                "status": "epoch_complete",
            }
            pointers = ("last", "best") if improved else ("last",)
            save_checkpoint(
                accelerator, model, tokenizer, output, run_id,
                f"epoch-{epoch + 1:04d}-step-{global_step:08d}", state, pointers,
            )
            model.train()
            if global_step >= total_steps:
                break
    except KeyboardInterrupt:
        state = {
            "epoch": epoch + 1, "global_step": global_step, "val_loss": None,
            "best_val_loss": best_loss, "model_id": args.model_id,
            "model_revision": args.model_revision, "run_id": run_id,
            "status": "interrupted",
        }
        save_checkpoint(accelerator, model, tokenizer, output, run_id,
                        f"interrupted-step-{global_step:08d}", state, ("last",))
        accelerator.print("Training interrupted; last and best checkpoints are preserved")
        raise
    progress.close()


if __name__ == "__main__":
    main()
