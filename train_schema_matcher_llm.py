#!/usr/bin/env python3
"""
Train Schema Matcher LLM (V5 - Robust Production)
====================================================
Fine-tunes a small instruction-following LLM (Phi-3-mini or similar) using
LoRA adapters for the schema mapping task.

V5 enhancements over V4:
  - LoRA rank 96 (up from 64) with alpha 192, hardcoded 7 target modules
  - Learning rate 5e-5 (down from 1e-4) for stable convergence on larger data
  - Early stopping with patience=3 on eval_loss
  - Frequent eval every 150 steps
  - Auto-selects best checkpoint via load_best_model_at_end
  - Post-training LoRA merge for peft-free inference
  - Optional 4-bit quantization for edge deployment

Usage:
  # Generate training data first
  python generate_llm_training_data.py

  # Fine-tune with production defaults
  python train_schema_matcher_llm.py

  # Fine-tune and merge for fast inference
  python train_schema_matcher_llm.py --merge

  # Fine-tune, merge, and quantize for edge deployment
  python train_schema_matcher_llm.py --merge --quantize

  # Resume interrupted training
  python train_schema_matcher_llm.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------
# Dependency check: install missing packages automatically
# ---------------------------------------------------------------

_REQUIRED_PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "transformers": "transformers",
    "peft": "peft",
    "trl": "trl",
    "datasets": "datasets",
    "accelerate": "accelerate",
}


def _ensure_dependencies():
    """Check and install any missing dependencies."""
    missing = []
    for import_name, pip_name in _REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[SETUP] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("[SETUP] Installation complete.\n")


_ensure_dependencies()

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------
# Import SFTTrainer with full API version detection
# ---------------------------------------------------------------

import inspect as _inspect

_SFTConfig = None
_SFTTrainer = None
_FallbackArgs = None

try:
    from trl import SFTTrainer as _SFTTrainer_cls
    _SFTTrainer = _SFTTrainer_cls
except ImportError:
    print("ERROR: Could not import SFTTrainer from trl.")
    print("Install with: pip install trl>=0.8.0")
    sys.exit(1)

try:
    from trl import SFTConfig as _SFTConfig_cls
    _SFTConfig = _SFTConfig_cls
except ImportError:
    from transformers import TrainingArguments as _FallbackArgs_cls
    _FallbackArgs = _FallbackArgs_cls


def _build_sft_config(common_args: dict, max_seq_length: int):
    """Build training config compatible with whatever trl version is installed."""
    if _SFTConfig is not None:
        # Probe which extra kwargs SFTConfig actually accepts
        sig = _inspect.signature(_SFTConfig.__init__)
        params = set(sig.parameters.keys())

        extra = {}
        if "max_seq_length" in params:
            extra["max_seq_length"] = max_seq_length
        if "dataset_text_field" in params:
            extra["dataset_text_field"] = "text"

        return _SFTConfig(**common_args, **extra), extra
    else:
        return _FallbackArgs(**common_args), {}


def _build_sft_trainer(model, training_args, train_dataset, val_dataset,
                       tokenizer, max_seq_length: int, sft_extra: dict):
    """Build SFTTrainer compatible with whatever trl version is installed."""
    sig = _inspect.signature(_SFTTrainer.__init__)
    params = set(sig.parameters.keys())

    kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # tokenizer arg: "processing_class" (new) vs "tokenizer" (old)
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    # If max_seq_length / dataset_text_field weren't in the config,
    # try passing them to the Trainer directly
    if "max_seq_length" not in sft_extra and "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_length
    if "dataset_text_field" not in sft_extra and "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"

    return _SFTTrainer(**kwargs)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        dev = "cuda"
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        mem_bytes = torch.cuda.get_device_properties(0).total_mem
        print(f"[GPU] Memory: {mem_bytes / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        print("[Device] Apple MPS")
    else:
        dev = "cpu"
        print("[Device] CPU (training will be slow)")
    return dev


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_chat_for_training(record: Dict, tokenizer) -> str:
    """Format a chat record into the model's chat template."""
    messages = record["messages"]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return text
    except Exception:
        # Fallback: manual Phi-3 style formatting
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}<|end|>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}<|end|>")
        return "\n".join(parts)


def _fix_rope_scaling(cfg):
    """Fix Phi-3 rope_scaling incompatibility with newer transformers.

    Newer transformers sets rope_scaling={"rope_type": "default"} for
    standard RoPE, but Phi-3's custom code only recognises None (standard)
    or "longrope". When the type is "default" we clear it to None.
    """
    if not hasattr(cfg, "rope_scaling") or cfg.rope_scaling is None:
        return cfg

    rs = cfg.rope_scaling
    scaling_type = rs.get("type", rs.get("rope_type", ""))

    if scaling_type in ("default", ""):
        cfg.rope_scaling = None
        print("[FIX] Cleared rope_scaling (type='default' -> None for standard RoPE)")
    else:
        # Ensure both keys exist for special types like "longrope"
        if "type" not in rs and "rope_type" in rs:
            rs["type"] = rs["rope_type"]
        elif "rope_type" not in rs and "type" in rs:
            rs["rope_type"] = rs["type"]
        cfg.rope_scaling = rs

    return cfg


# ---------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------

def train(
    data_dir: str = "llm_training_data",
    output_dir: str = "schema_matcher_llm",
    base_model: str = "microsoft/Phi-3-mini-4k-instruct",
    epochs: int = 6,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    lr: float = 5e-5,
    max_seq_length: int = 2048,
    lora_r: int = 96,
    lora_alpha: int = 192,
    lora_dropout: float = 0.05,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    bf16: bool = True,
    fp16: bool = False,
    resume_from_checkpoint: bool = False,
    early_stopping_patience: int = 3,
    merge_after_training: bool = False,
    quantize_after_training: bool = False,
):
    """Fine-tune the LLM with LoRA for schema matching."""
    set_seed(seed)
    device = detect_device()

    # Auto-detect precision
    if bf16 and device == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("[WARN] bf16 not supported, falling back to fp16")
            bf16 = False
            fp16 = True
    elif device != "cuda":
        bf16 = False
        fp16 = False

    print(f"\n{'='*60}")
    print(f"  Schema Matcher LLM Training (Production)")
    print(f"{'='*60}")
    print(f"  Base model:    {base_model}")
    print(f"  Data dir:      {data_dir}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size} x {gradient_accumulation} = {batch_size * gradient_accumulation} effective")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup ratio:  {warmup_ratio}")
    print(f"  LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  Max seq len:   {max_seq_length}")
    print(f"  Precision:     {'bf16' if bf16 else 'fp16' if fp16 else 'fp32'}")
    print(f"  Early stop:    patience={early_stopping_patience}")
    print(f"  Merge after:   {merge_after_training}")
    print(f"  Resume:        {resume_from_checkpoint}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_path = os.path.join(data_dir, "llm_train.jsonl")
    val_path = os.path.join(data_dir, "llm_val.jsonl")

    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}")
        print("Run first: python generate_llm_training_data.py")
        sys.exit(1)

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path) if os.path.exists(val_path) else []
    print(f"Loaded {len(train_data):,} train, {len(val_data):,} val examples")

    # ------------------------------------------------------------------
    # 2. Load tokenizer
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # 3. Load model (with rope_scaling fix)
    # ------------------------------------------------------------------
    print(f"Loading model: {base_model}")
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    # Pre-load config and fix rope_scaling for Phi-3 + newer transformers
    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    cfg = _fix_rope_scaling(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=cfg,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Fix: Newer transformers Trainer passes extra kwargs (e.g. num_items_in_batch)
    # to model.forward(), but Phi-3's custom code doesn't accept them.
    # Patch forward to silently drop unexpected kwargs.
    _original_forward = model.forward

    def _patched_forward(*args, **kwargs):
        import inspect as _fwd_inspect
        sig = _fwd_inspect.signature(_original_forward)
        valid_params = set(sig.parameters.keys())
        # Keep only kwargs the real forward() accepts (+ **kwargs if it has VAR_KEYWORD)
        has_var_keyword = any(
            p.kind == _fwd_inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_keyword:
            kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return _original_forward(*args, **kwargs)

    model.forward = _patched_forward
    print("[FIX] Patched model.forward() to handle extra trainer kwargs")

    print(f"Model loaded: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 4. Apply LoRA
    # ------------------------------------------------------------------
    # Auto-detect the actual linear layer names in the model, then target
    # all of them except lm_head (the final output projection).
    # Phi-3 uses fused layers: qkv_proj (not q/k/v), gate_up_proj (not gate/up).
    linear_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            linear_names.add(leaf)
    print(f"Model linear layer names: {sorted(linear_names)}")

    # Target all linear layers except lm_head
    target_modules = sorted(linear_names - {"lm_head"})
    if not target_modules:
        target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    print(f"LoRA target modules ({len(target_modules)}): {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    trainable_pct = trainable / total * 100
    print(f"Trainable parameters: {trainable:,} ({trainable_pct:.2f}%)")

    # Verify LoRA was applied to enough modules (should be >1% for rank 96)
    # With all 7 modules at rank 96, expect ~2-4% trainable
    # With only 2 modules, expect ~0.5-0.8% trainable
    lora_layers = [n for n, _ in model.named_modules() if "lora_" in n.lower()]
    print(f"LoRA adapter layers created: {len(lora_layers)}")
    if trainable_pct < 1.0:
        print(f"[CRITICAL] Only {trainable_pct:.2f}% params trainable - LoRA may not have targeted all modules!")
        print(f"  Expected >2% with rank {lora_r} on 7 modules. Check model architecture.")
        # Print which modules actually got LoRA adapters
        targeted = set()
        for n, _ in model.named_modules():
            if "lora_A" in n:
                parts = n.split(".")
                for p in parts:
                    if p in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                        targeted.add(p)
        print(f"  Modules that actually received LoRA: {sorted(targeted)}")
    else:
        print(f"[OK] LoRA trainable % looks healthy ({trainable_pct:.2f}%)")

    # ------------------------------------------------------------------
    # 5. Format data into text
    # ------------------------------------------------------------------
    print("\nFormatting training data...")
    train_texts = [format_chat_for_training(r, tokenizer) for r in train_data]
    val_texts = [format_chat_for_training(r, tokenizer) for r in val_data]

    sample_tokens = tokenizer(train_texts[0], return_tensors="pt")
    print(f"Sample input length: {sample_tokens['input_ids'].shape[1]} tokens")

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts}) if val_texts else None

    # ------------------------------------------------------------------
    # 6. Build trainer (auto-adapts to any trl version)
    # ------------------------------------------------------------------
    has_eval = val_dataset is not None
    common_args = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=150,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=150 if has_eval else None,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=seed,
        save_total_limit=5,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
    )

    training_args, sft_extra = _build_sft_config(common_args, max_seq_length)
    trainer = _build_sft_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        sft_extra=sft_extra,
    )

    # Add early stopping callback if we have validation data
    if has_eval and early_stopping_patience > 0:
        try:
            from transformers import EarlyStoppingCallback
            trainer.add_callback(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )
            print(f"[CONFIG] Early stopping enabled (patience={early_stopping_patience})")
        except ImportError:
            print("[WARN] EarlyStoppingCallback not available, skipping early stopping")

    # ------------------------------------------------------------------
    # 7. Train (with optional checkpoint resume)
    # ------------------------------------------------------------------
    ckpt_arg = None
    if resume_from_checkpoint:
        # Find latest checkpoint in output_dir
        import glob as _glob
        ckpt_dirs = sorted(
            _glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda p: int(os.path.basename(p).split("-")[-1]) if os.path.basename(p).split("-")[-1].isdigit() else 0,
        )
        if ckpt_dirs:
            ckpt_arg = ckpt_dirs[-1]
            print(f"\nResuming from checkpoint: {ckpt_arg}")
        else:
            print(f"\n[WARN] --resume flag set but no checkpoint-* found in {output_dir}. Training from scratch.")

    print(f"\nStarting training...")
    t_start = time.time()
    train_result = trainer.train(resume_from_checkpoint=ckpt_arg)
    elapsed = time.time() - t_start

    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Train loss: {train_result.training_loss:.4f}")

    # ------------------------------------------------------------------
    # 8. Save adapter + metadata
    # ------------------------------------------------------------------
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Collect best eval loss from trainer state if available
    best_eval_loss = None
    try:
        if hasattr(trainer, "state") and hasattr(trainer.state, "best_metric"):
            best_eval_loss = trainer.state.best_metric
    except Exception:
        pass

    metadata = {
        "base_model": base_model,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "max_seq_length": max_seq_length,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "training_loss": train_result.training_loss,
        "best_eval_loss": best_eval_loss,
        "early_stopping_patience": early_stopping_patience,
        "training_time_seconds": elapsed,
        "seed": seed,
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {output_dir}/")
    print(f"  Adapter weights: {adapter_dir}/")
    print(f"  Metadata: {output_dir}/training_metadata.json")

    # ------------------------------------------------------------------
    # 9. Quick validation
    # ------------------------------------------------------------------
    if val_dataset:
        print("\nRunning final validation...")
        eval_result = trainer.evaluate()
        final_eval_loss = eval_result.get("eval_loss", None)
        print(f"  Final val loss: {final_eval_loss}")
        if best_eval_loss is not None:
            print(f"  Best val loss:  {best_eval_loss}")

    # ------------------------------------------------------------------
    # 10. Post-training: Merge LoRA into base model
    # ------------------------------------------------------------------
    if merge_after_training:
        print(f"\n{'='*60}")
        print(f"  Merging LoRA adapter into base model...")
        print(f"{'='*60}")
        try:
            merged_dir = os.path.join(output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)

            # merge_and_unload bakes LoRA weights into the base model
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)

            # Save merge metadata
            merge_meta = {
                "base_model": base_model,
                "merged_from_adapter": adapter_dir,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules,
                "training_loss": train_result.training_loss,
                "best_eval_loss": best_eval_loss,
            }
            with open(os.path.join(merged_dir, "merge_metadata.json"), "w") as f:
                json.dump(merge_meta, f, indent=2)

            print(f"  Merged model saved to: {merged_dir}/")
            print(f"  This model can be loaded directly without peft:")
            print(f"    model = AutoModelForCausalLM.from_pretrained('{merged_dir}')")
        except Exception as e:
            print(f"  [ERROR] Merge failed: {e}")
            print(f"  The LoRA adapter is still available at: {adapter_dir}/")

    # ------------------------------------------------------------------
    # 11. Post-training: Optional 4-bit quantization
    # ------------------------------------------------------------------
    if quantize_after_training:
        print(f"\n{'='*60}")
        print(f"  Quantizing model to 4-bit...")
        print(f"{'='*60}")
        try:
            from transformers import BitsAndBytesConfig

            quant_dir = os.path.join(output_dir, "quantized_4bit")
            os.makedirs(quant_dir, exist_ok=True)

            # Load the merged model (or base + adapter) with 4-bit quantization
            source_path = os.path.join(output_dir, "merged") if merge_after_training else base_model
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            quant_model = AutoModelForCausalLM.from_pretrained(
                source_path,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # If loading from base (not merged), apply adapter
            if not merge_after_training:
                from peft import PeftModel as _PeftModel
                quant_model = _PeftModel.from_pretrained(quant_model, adapter_dir)
                quant_model = quant_model.merge_and_unload()

            quant_model.save_pretrained(quant_dir)
            tokenizer.save_pretrained(quant_dir)
            print(f"  4-bit quantized model saved to: {quant_dir}/")
        except ImportError:
            print("  [ERROR] bitsandbytes not installed. Install with:")
            print("    pip install bitsandbytes")
        except Exception as e:
            print(f"  [ERROR] Quantization failed: {e}")

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Artifacts:")
    print(f"    Adapter:   {os.path.join(output_dir, 'adapter')}/")
    if merge_after_training:
        print(f"    Merged:    {os.path.join(output_dir, 'merged')}/")
    if quantize_after_training:
        print(f"    Quantized: {os.path.join(output_dir, 'quantized_4bit')}/")
    print(f"  Next steps:")
    print(f"    python evaluate_schema_matcher_llm.py")
    print(f"    python candidate_generation_v3.py --test")


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Schema Matcher LLM (Production)")
    parser.add_argument("--data", default="llm_training_data",
                        help="Training data directory (default: llm_training_data)")
    parser.add_argument("--output", default="schema_matcher_llm",
                        help="Output directory for model (default: schema_matcher_llm)")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct",
                        help="Base model name/path")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Number of training epochs (default: 6)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size per device (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--lora-r", type=int, default=96,
                        help="LoRA rank (default: 96)")
    parser.add_argument("--lora-alpha", type=int, default=192,
                        help="LoRA alpha (default: 192)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable bf16 (use fp16 instead)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint in output dir")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Early stopping patience on eval_loss (0=disable, default: 3)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA into base model after training for faster inference")
    parser.add_argument("--quantize", action="store_true",
                        help="Export 4-bit quantized model after training (requires GPU)")

    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_dir=args.output,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        gradient_accumulation=args.grad_accum,
        lr=args.lr,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        bf16=not args.no_bf16,
        fp16=args.no_bf16,
        resume_from_checkpoint=args.resume,
        early_stopping_patience=args.early_stop_patience,
        merge_after_training=args.merge,
        quantize_after_training=args.quantize,
    )


if __name__ == "__main__":
    main()
