"""
Data utilities for custom parallel translation datasets.

Expected data format (CSV, JSON, or JSONL):
    source  | target
    --------|--------
    <src>   | <tgt>

After running generate_unpreferred(), the DPO-ready format becomes:
    source  | preferred (= target) | unpreferred (= m2m100 output)
"""
import logging
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_parallel_data(
    path: str,
    src_col: str = "source",
    tgt_col: str = "target",
    split: str | None = None,
) -> Dataset:
    """
    Load a parallel corpus from CSV, JSON, JSONL, or a HuggingFace dataset ID.

    Args:
        path: Local file path or HuggingFace dataset identifier.
        src_col: Column name for source sentences.
        tgt_col: Column name for target (reference) sentences.
        split: Dataset split (e.g. "train", "test"). Required for HF datasets.

    Returns:
        A HuggingFace Dataset with 'source' and 'target' columns.
    """
    p = Path(path)

    if p.exists():
        ext = p.suffix.lower()
        if ext == ".csv":
            ds = Dataset.from_csv(str(p))
        elif ext == ".json":
            ds = Dataset.from_json(str(p))
        elif ext == ".jsonl":
            ds = Dataset.from_json(str(p))
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .csv, .json, or .jsonl")
    else:
        # Treat as HuggingFace dataset ID
        ds = load_dataset(path, split=split or "train", trust_remote_code=True)

    # Rename columns to standard names if needed
    if src_col != "source" and src_col in ds.column_names:
        ds = ds.rename_column(src_col, "source")
    if tgt_col != "target" and tgt_col in ds.column_names:
        ds = ds.rename_column(tgt_col, "target")

    assert "source" in ds.column_names, f"Missing 'source' column. Available: {ds.column_names}"
    assert "target" in ds.column_names, f"Missing 'target' column. Available: {ds.column_names}"

    logger.info(f"Loaded {len(ds)} parallel sentence pairs from {path}")
    return ds


def generate_unpreferred(
    dataset: Dataset,
    model_name: str = "facebook/m2m100_418M",
    src_lang: str = "hi",
    tgt_lang: str = "en",
    max_length: int = 256,
    batch_size: int = 16,
    device: str | None = None,
) -> Dataset:
    """
    Generate unpreferred (lower-quality) translations using m2m100.

    Creates DPO training triplets: (source, preferred, unpreferred).

    Args:
        dataset: Must have 'source' and 'target' columns.
        model_name: The weaker model to generate unpreferred outputs.
        src_lang: m2m100 source language code (e.g. "hi", "ur", "mr").
        tgt_lang: m2m100 target language code (e.g. "en").
        max_length: Max generation length.
        batch_size: Inference batch size.
        device: "cuda" / "cpu" / None (auto-detect).

    Returns:
        Dataset with columns: source, preferred, unpreferred.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    tokenizer.src_lang = src_lang

    sources = dataset["source"]
    unpreferred = []

    for i in tqdm(range(0, len(sources), batch_size), desc="Generating unpreferred"):
        batch = sources[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                max_length=max_length,
                num_beams=5,
            )
        unpreferred.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()

    dpo_dataset = Dataset.from_dict({
        "source": sources,
        "preferred": dataset["target"],
        "unpreferred": unpreferred,
    })
    logger.info(f"Generated {len(unpreferred)} unpreferred translations")
    return dpo_dataset
