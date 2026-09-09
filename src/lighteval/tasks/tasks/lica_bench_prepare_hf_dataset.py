#!/usr/bin/env python3
"""Prepare and push the LICA-Bench HuggingFace dataset for lighteval.

Creates one subset per task at ``purvanshi/lica-bench-eval`` on the Hub.

Usage:
    pip install "lica-bench @ git+https://github.com/purvanshi/lica-bench.git"
    python lica_bench_prepare_hf_dataset.py --dataset-root /path/to/lica-benchmarks-dataset

Requires:
    - lica-bench package (see above)
    - huggingface_hub (``pip install huggingface_hub``)
    - ``huggingface-cli login`` to push to the Hub
"""

import argparse
import json
import os
from pathlib import Path

import datasets
from PIL import Image


HF_REPO = "purvanshi/lica-bench-eval"

TASK_IDS = [
    "category-1", "category-2",
    "layout-1", "layout-2", "layout-3", "layout-4",
    "layout-5", "layout-6", "layout-7", "layout-8",
    "svg-1", "svg-2", "svg-3", "svg-4",
    "svg-5", "svg-6", "svg-7", "svg-8",
    "template-1", "template-2", "template-3", "template-4", "template-5",
    "temporal-1", "temporal-2", "temporal-3",
    "temporal-4", "temporal-5", "temporal-6",
    "typography-1", "typography-2", "typography-3", "typography-4",
    "typography-5", "typography-6", "typography-7", "typography-8",
    "lottie-1", "lottie-2",
]


def _serialize_gt(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def build_and_push(dataset_root: str, task_ids: list[str] | None = None, push: bool = True):
    from design_benchmarks import BenchmarkRegistry
    from design_benchmarks.models.base import ModelInput, Modality

    registry = BenchmarkRegistry()
    registry.discover()

    if task_ids is None:
        task_ids = TASK_IDS

    for tid in task_ids:
        print(f"\n--- {tid} ---")
        try:
            bench = registry.get(tid)
        except KeyError:
            print(f"  Task {tid} not found in registry, skipping.")
            continue

        try:
            data_dir = bench.resolve_data_dir(dataset_root)
            samples = bench.load_data(data_dir, dataset_root=dataset_root)
        except Exception as e:
            print(f"  Failed to load data: {e}")
            continue

        rows = {"question": [], "answer": [], "domain": [], "task_id": [], "image": []}

        for sample in samples:
            model_input = bench.build_model_input(sample, modality=Modality.TEXT_AND_IMAGE)
            if not isinstance(model_input, ModelInput):
                continue

            pil_image = None
            for img in model_input.images or []:
                if isinstance(img, (str, Path)):
                    p = Path(img).expanduser().resolve()
                    if p.is_file():
                        try:
                            pil_image = Image.open(str(p)).convert("RGB")
                            break
                        except Exception:
                            pass

            text = model_input.text or ""
            meta = model_input.metadata or {}
            if meta:
                meta_str = json.dumps(meta, ensure_ascii=False, default=str)
                if len(meta_str) > 100_000:
                    meta_str = meta_str[:100_000] + "...[truncated]"
                text = f"{text}\n\n[metadata]\n{meta_str}" if text else f"[metadata]\n{meta_str}"

            gt = _serialize_gt(sample.get("ground_truth", ""))

            rows["question"].append(text)
            rows["answer"].append(gt)
            rows["domain"].append(bench.meta.domain)
            rows["task_id"].append(tid)
            rows["image"].append(pil_image)

        features = datasets.Features({
            "question": datasets.Value("string"),
            "answer": datasets.Value("string"),
            "domain": datasets.Value("string"),
            "task_id": datasets.Value("string"),
            "image": datasets.Image(),
        })

        ds = datasets.Dataset.from_dict(rows, features=features)
        print(f"  {len(ds)} samples")

        if push:
            ds.push_to_hub(HF_REPO, config_name=tid, split="test")
            print(f"  Pushed to {HF_REPO} (config={tid})")
        else:
            out_dir = os.path.join("lica_bench_hf_data", tid)
            ds.save_to_disk(out_dir)
            print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LICA-Bench HuggingFace dataset for lighteval")
    parser.add_argument("--dataset-root", required=True, help="Path to lica-benchmarks-dataset/")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task IDs (default: all)")
    parser.add_argument("--no-push", action="store_true", help="Save locally instead of pushing to Hub")
    args = parser.parse_args()

    build_and_push(
        dataset_root=args.dataset_root,
        task_ids=args.tasks,
        push=not args.no_push,
    )
