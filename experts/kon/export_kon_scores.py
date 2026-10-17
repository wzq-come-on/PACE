#!/usr/bin/env python3
"""Export frozen K-ON (semantic expert) scores into the PACE score layout.

K-ON scores every candidate entity from a text prompt.  This exporter runs the
frozen K-ON checkpoint over the queries of one split and writes

    {output_root}/{dataset}/{split}_head/semantic.npy
    {output_root}/{dataset}/{split}_tail/semantic.npy

with shape ``[queries, entities]``, float32, higher is better.  Columns follow
the K-ON entity-id order, which is the order the contracts use.

Requirements
------------
The K-ON code base must be importable (``k_on.py``, ``dataset.py``,
``checkpoint_utils.py``) together with its data folder:

    export PYTHONPATH=/path/to/K-ON:$PYTHONPATH

The checkpoint and config are not redistributed here; see ``configs/*.yaml``
for the settings used in the paper.

Usage
-----
    python experts/kon/export_kon_scores.py \
        --kon-root /path/to/K-ON \
        --config configs/DB15K.yaml \
        --checkpoint /path/to/K-ON/outputs/DB15K/checkpoint-4140 \
        --dataset DB15K --split test --output-root data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import easydict
import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

# K-ON stores the prompt column names per split; both splits expose the same
# pair (forward prompt, inverse prompt) -> (tail queries, head queries).
PROMPT_COLUMNS = {
    "test": ("input_text", "inv_input_text"),
    "train": ("input_text", "inv_input_text"),
}


@torch.inference_mode()
def export_prompts(model, tokenizer, prompts, output: Path, batch_size: int,
                   num_entities: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(len(prompts), num_entities))
    base = model.get_base_model()
    device = next(base.parameters()).device
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        tokenized = tokenizer(list(prompts[start:end]), padding=True,
                              truncation=True, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        input_length = attention_mask.sum(dim=1)
        # Match TrainingArguments(bf16=True): K-ON's float32 RMSNorm weights
        # would otherwise promote activations before the bf16 attention block.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = base.model(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False, return_dict=True)[0]
            last = hidden[torch.arange(end - start, device=device), input_length - 1]
            token_predictions = base.k_on(last)
            raw_scores = base.k_off(token_predictions)
        matrix[start:end] = raw_scores.float().cpu().numpy()
        print(f"{output.name}: {end}/{len(prompts)}", flush=True)
    matrix.flush()
    del matrix
    temporary.replace(output)


def load_model(kon_root: Path, config: dict, checkpoint: Path, device: torch.device):
    """Rebuild the frozen K-ON model exactly as the training entry point does."""
    if str(kon_root) not in sys.path:
        sys.path.insert(0, str(kon_root))
    from checkpoint_utils import load_trainable_state  # noqa: PLC0415
    # The historical preprocessed.pkl files were created by executing
    # dataset.py, so unpickling resolves several helper classes from
    # ``__main__``.  Publish the dataset module's names in this module's
    # namespace, mirroring the original entry point's wildcard import.
    import importlib  # noqa: PLC0415
    dataset_module = importlib.import_module("dataset")
    globals().update({name: value for name, value in vars(dataset_module).items()
                      if not name.startswith("__")})
    KONDataset = dataset_module.KONDataset
    from k_on import KON, KONConfig  # noqa: PLC0415

    dataset = KONDataset.load(str(kon_root / config.dataset.base_dir / "preprocessed.pkl"))

    tokenizer = AutoTokenizer.from_pretrained(
        str(kon_root / config.tokenizer.pretrained_model_name_or_path),
        use_fast=False, add_eos_token=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    entity_tokens = tokenizer(dataset.ent2text.values.tolist(), truncation=True,
                              padding=True, max_length=config.k_on.num_k_on).input_ids
    tokenizer.add_bos_token = True

    model_config = KONConfig.from_pretrained(
        str(kon_root / config.llm_config.pretrained_model_name_or_path))
    model = KON.from_pretrained(
        str(kon_root / config.llm.pretrained_model_name_or_path),
        device_map={"": device.index or 0},
        config=model_config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = get_peft_model(model, LoraConfig(**config.loraconfig))
    model.init_kg_specs(torch.tensor(entity_tokens), config.k_on)
    count, elements = load_trainable_state(model, str(checkpoint))
    base = model.get_base_model()
    # Training relies on Accelerator.prepare() to move modules created after
    # from_pretrained(); the standalone exporter must move them explicitly.
    base.k_on_attn.to(next(base.parameters()).device)
    model.eval()
    print(f"loaded {count} trainable tensors / {elements} elements from {checkpoint}",
          flush=True)
    return model, tokenizer, dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kon-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=tuple(PROMPT_COLUMNS), default="test")
    p.add_argument("--output-root", type=Path, default=Path("data"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    config = easydict.EasyDict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    begin = time.time()
    model, tokenizer, dataset = load_model(args.kon_root, config, args.checkpoint, device)

    frame = getattr(dataset, args.split)
    forward_column, inverse_column = PROMPT_COLUMNS[args.split]
    folder = args.output_root / args.dataset
    # Tail queries score with the forward prompt, head queries with the inverse.
    export_prompts(model, tokenizer, frame[forward_column].tolist(),
                   folder / f"{args.split}_tail" / "semantic.npy",
                   args.batch_size, dataset.num_ent)
    export_prompts(model, tokenizer, frame[inverse_column].tolist(),
                   folder / f"{args.split}_head" / "semantic.npy",
                   args.batch_size, dataset.num_ent)
    print(f"done in {time.time() - begin:.1f}s", flush=True)


if __name__ == "__main__":
    main()
