from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from falcon_perception.attention import create_batch_attention_mask
from falcon_perception.data import ImageProcessor, get_pos_thw, tokenize_inputs
from falcon_perception.model import ImgScatterEntry

from .format import PROMPT


class TrainingKVCache:
    """No-cache adapter: full teacher-forced attention remains differentiable."""
    def __init__(self):
        self.pos = 0
        self.pos_t = None
    def get_pos(self): return 0
    def insert_kv(self, layer_id, k, v, **kwargs): return k, v


@dataclass
class Collator:
    tokenizer: object
    model_args: object
    max_seq_len: int = 8192
    min_dimension: int = 256
    max_dimension: int = 1024

    def __post_init__(self):
        self.processor = ImageProcessor(
            self.model_args.spatial_patch_size, 1,
            min_pixels=self.min_dimension ** 2,
            max_pixels=self.max_dimension ** 2,
        )

    def __call__(self, samples):
        sequences, prompt_lengths, images = [], [], []
        for sample in samples:
            processed = self.processor.preprocess(images=[sample["image"]])
            prompt_ids, selected = tokenize_inputs(
                PROMPT, processed, self.tokenizer,
                self.model_args.spatial_patch_size, 1, self.max_seq_len,
            )
            target_ids = self.tokenizer.encode(sample["target"])
            ids = list(prompt_ids) + target_ids + [self.tokenizer.eos_token_id]
            if len(ids) > self.max_seq_len:
                raise ValueError(
                    f"Sample {sample['path']} has {len(ids)} tokens; max is {self.max_seq_len}. "
                    "Tile the page or reduce words per sample."
                )
            sequences.append(np.asarray(ids, dtype=np.int64))
            prompt_lengths.append(len(prompt_ids))
            images.extend(selected)

        length = max(map(len, sequences))
        tokens = np.full((len(samples), length), self.tokenizer.pad_token_id, np.int64)
        labels = np.full((len(samples), length), -100, np.int64)
        for i, (ids, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
            tokens[i, -len(ids):] = ids
            start = length - len(ids) + prompt_len
            labels[i, start:] = ids[prompt_len:]

        if len(images) != len(samples):
            raise ValueError("Each training sample must contain exactly one usable image")
        # smart_resize caps area, not each side. Size the canvas from the actual
        # processed images so unusually tall/wide documents cannot overflow it.
        canvas_h = max(image.shape[1] for image in images)
        canvas_w = max(image.shape[2] for image in images)
        processed = self.processor.batch_images_with_mask(images, canvas_h, canvas_w)
        pos_t, pos_hw = get_pos_thw(
            tokens, processed["padding_mask"], self.tokenizer,
            self.model_args.spatial_patch_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return {"tokens": torch.from_numpy(tokens), "labels": torch.from_numpy(labels),
                "pixel_values": torch.from_numpy(processed["pixel_values"]),
                "pixel_mask": torch.from_numpy(processed["padding_mask"]),
                "pos_t": torch.from_numpy(pos_t), "pos_hw": torch.from_numpy(pos_hw)}


def forward_loss(model, tokenizer, batch, *, return_sample_stats: bool = False):
    core = model.module if hasattr(model, "module") else model
    tokens = batch["tokens"]
    labels = batch["labels"]
    B, S = tokens.shape
    attention = create_batch_attention_mask(
        tokens, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        soi_token_id=tokenizer.image_cls_token_id,
        eoi_token_id=tokenizer.end_of_image_token_id, max_len=S,
    )
    scatter = []
    ps = core.args.spatial_patch_size
    tokens_cpu = tokens.detach().cpu()
    masks_cpu = batch["pixel_mask"].detach().cpu()
    for b in range(B):
        pos = (tokens_cpu[b] == core.args.img_id).nonzero(as_tuple=True)[0]
        if len(pos):
            mask = masks_cpu[b]
            hv = int(mask.sum(dim=-2).max()) // ps
            wv = int(mask.sum(dim=-1).max()) // ps
            scatter.append(ImgScatterEntry(b, int(pos[0]), len(pos), hv, wv))
    valid_targets = labels[:, 1:] != -100
    logit_positions = torch.zeros_like(tokens, dtype=torch.bool)
    logit_positions[:, :-1] = valid_targets
    logits, _ = model(
        tokens=tokens, attention_mask=attention, kv_cache=TrainingKVCache(),
        rope_pos_t=batch["pos_t"], rope_pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"], img_scatter_info=scatter,
        logit_positions=logit_positions,
    )
    packed_labels = labels[:, 1:][valid_targets]
    token_losses = torch.nn.functional.cross_entropy(
        logits.float(), packed_labels, reduction="none")
    loss = token_losses.mean()
    if not return_sample_stats:
        return loss
    sample_ids = torch.arange(B, device=tokens.device).unsqueeze(1).expand(B, S - 1)[valid_targets]
    sample_sums = token_losses.new_zeros(B).scatter_add_(0, sample_ids, token_losses.detach())
    sample_counts = torch.bincount(sample_ids, minlength=B).to(token_losses.dtype)
    return loss, sample_sums, sample_counts


def prepare_inference_batch(tokenizer, model_args, image, *, min_dimension=256,
                            max_dimension=1024):
    """Prepare one image/prompt without assuming a square maximum-size canvas."""
    processor = ImageProcessor(
        model_args.spatial_patch_size, 1,
        min_pixels=min_dimension ** 2, max_pixels=max_dimension ** 2,
    )
    images = processor.preprocess(images=[image])
    token_ids, selected = tokenize_inputs(
        PROMPT, images, tokenizer, model_args.spatial_patch_size, 1,
        model_args.max_seq_len,
    )
    if len(selected) != 1:
        raise ValueError("Inference prompt did not select exactly one image")
    canvas_h, canvas_w = selected[0].shape[1:3]
    processed = processor.batch_images_with_mask(selected, canvas_h, canvas_w)
    tokens = np.asarray(token_ids, dtype=np.int64)[None, :]
    pos_t, pos_hw = get_pos_thw(
        tokens, processed["padding_mask"], tokenizer,
        model_args.spatial_patch_size, pad_token_id=tokenizer.pad_token_id,
    )
    return {
        "tokens": torch.from_numpy(tokens),
        "pixel_values": torch.from_numpy(processed["pixel_values"]),
        "pixel_mask": torch.from_numpy(processed["padding_mask"]),
        "pos_t": torch.from_numpy(pos_t),
        "pos_hw": torch.from_numpy(pos_hw),
    }
