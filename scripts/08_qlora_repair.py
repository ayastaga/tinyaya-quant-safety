"""QLoRA safety-healing pass. RUN ONLY IF THE AUDIT FINDS DAMAGE, and target
the languages the audit flagged.

Idea: fine-tune a small LoRA adapter ON TOP OF THE 4-BIT MODEL (so the adapter
learns to correct quantized behavior) using safe refusal pairs in the damaged
languages, mixed with general instruction data to prevent over-refusal
regression (XSTest check afterwards!).

Training data recipe (build_dataset):
  harmful prompts  -> CohereLabs/aya_redteaming (multilingual, 8 langs) and/or
                      held-out MultiJail split — NEVER the same items you audit.
  safe completions -> generate with Command A (its contextual safety mode is
                      what distilled safety into Tiny Aya originally, per the
                      paper) — see 08b note in README of this script's output.
  utility mix      -> ~50% mDolly-style benign prompt/response pairs in the
                      same languages so the adapter doesn't collapse to
                      refuse-everything.

Usage:
  python 08_qlora_repair.py --model global --data results/repair_train.jsonl \
      --langs sw bn th --output adapters/global-repair
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

from common import load_config


def build_dataset(path, tok, langs, max_len=1024):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if langs and r.get("lang") not in langs:
                continue
            msgs = [{"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["completion"]}]
            text = tok.apply_chat_template(msgs, tokenize=False)
            rows.append({"text": text})
    ds = Dataset.from_list(rows)

    def tokenize(batch):
        out = tok(batch["text"], truncation=True, max_length=max_len)
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl: {prompt, completion, lang}")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    path = f'{cfg["paths"]["hf_cache"]}/{args.model}'
    tok = AutoTokenizer.from_pretrained(path)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # NOTE: bitsandbytes NF4 != llama.cpp Q4_0/Q4_K_M. The adapter trained here
    # transfers in practice, but for the paper: after training, merge adapter
    # into BF16 weights, re-quantize with 01_quantize.sh, and re-audit the
    # actual GGUF. That closed loop is the claim; document the NF4 caveat.
    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = build_dataset(args.data, tok, args.langs)
    trainer = Trainer(
        model=model,
        train_dataset=ds,
        args=TrainingArguments(
            output_dir=args.output, num_train_epochs=args.epochs,
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_ratio=0.03, logging_steps=10, save_strategy="epoch",
            bf16=True, report_to="none"),
    )
    trainer.train()
    model.save_pretrained(args.output)
    print(f"Adapter saved to {args.output}. Next: 09_reexport_and_reaudit.md")


if __name__ == "__main__":
    main()
