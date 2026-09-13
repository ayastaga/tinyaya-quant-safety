# Does the Safety Floor Survive the Phone? Auditing Tiny Aya's GGUF Quantizations

An independent audit of whether Tiny Aya's headline claims — a best-in-class
multilingual **safety floor** (MultiJail) and highly consistent **language
adherence** — survive in the **actual 4-bit GGUF artifacts users run on
devices** (Q8_0 / Q4_K_M / Q4_0), plus a lightweight **QLoRA repair** for any
degradation found.

Reference: *Tiny Aya: Bridging Scale and Multilingual Depth* (arXiv:2603.11510).
The paper evaluates quantization only on mDolly generation quality (Sec. 6);
safety, language confusion, and translation under quantization are unmeasured.

> **v2:** adds Eval D (GlobalMGSM reasoning: accuracy + CoT adherence + truncation,
> judge-free) and Eval E (structured-output validity ladder, judge-free, gated on a
> BF16 viability check), makes the QLoRA repair unconditional (stress quants Q3_K_M/Q2_K
> as methods demo if shipped precisions are robust), promotes the template check to a
> blocking Phase 0 step (`02b_template_check.py`), adds XSTest harness (`04d`), the
> Eval E prompt builder (`10`), extractor unit tests (`tests/`), and judge-tier
> subsampling (signal 50 / confirm 150 / final 315 per lang). See PROJECT_HANDOFF v2.

## Experiment grid

| Axis | Values |
|---|---|
| Models | tiny-aya-global, tiny-aya-earth, tiny-aya-fire, tiny-aya-water |
| Precision | BF16 (baseline), Q8_0, Q4_K_M, Q4_0 |
| Evals | MultiJail safe-rate (10 langs, Command A judge) · language confusion on mDolly (fastText line-level pass rate) · Flores en→X ChrF |

Priority order if compute-constrained: **Global × all precisions × MultiJail**
first (that's the headline), then Earth (lowest pre-quantization safety floor,
merged-safety fragility question), then confusion, then translation, then
Fire/Water.

## Run order (each script is standalone; all write JSON into `results/`)

```
00_download_models.py      # HF download (needs HF token; accept license on model pages)
01_quantize.sh             # HF -> GGUF f16 -> Q8_0 / Q4_K_M / Q4_0 (CPU ok)
02_inspect_tensor_map.py   # per-tensor quant map: which layers really got 4-bit?  (NO GPU)
02b_template_check.py      # BLOCKING backend parity gate (see below); exit 1 = stop
03_generate.py             # generations for any (model, precision, eval) cell
04_judge_multijail.py      # Command A judge -> safe response rate per language
05_language_confusion.py   # fastText line-level pass rate on mDolly generations
06_translation_chrf.py     # ChrF on Flores generations
07_aggregate.py            # merges results -> tables + paper-style figures
08_qlora_repair.py         # QLoRA safety-healing pass (only if audit finds damage)
09_reexport_and_reaudit.md # merge adapter -> re-GGUF -> rerun 03-07 on repaired model
```

## Backend parity gate

**Evaluation does not begin until BF16 and Q8_0 both pass
`scripts/02b_template_check.py`.** Q8_0 is near-lossless, so under greedy
decoding the two backends must produce near-identical text. If they do not, the
harness — not quantization — is producing the delta, and every number in the
project is garbage.

```bash
python scripts/02b_template_check.py --model global   # exit 0 required before 03_generate.py
```

The gate runs seven probes (English, Swahili, Chinese) and fails on any of:

- **prompt token ids differ** between transformers and llama.cpp. Both backends
  render the turn with the HF tokenizer's `apply_chat_template(...,
  tokenize=False, add_generation_prompt=True)`; llama.cpp then tokenizes that
  exact string with `add_bos=False, special=True`. The ids must match exactly —
  otherwise the two precisions are not being fed the same prompt.
- **no recognized stop token.** Tiny Aya ends a turn with `<EOS_TOKEN>` (3),
  `<|END_OF_TURN_TOKEN|>` (6) or `<|END_RESPONSE|>` (261001). The GGUF metadata
  marks only token 6 as EOG, and `create_chat_completion`'s string `stop=`
  sequences never match the special-token spellings, so the GGUF backend drives
  `llama_cpp.Llama.generate` directly and compares raw generated ids against all
  three. `max_new_tokens` remains a hard fallback only.
- **obvious repetition** in either response — the signature of a turn that ran
  past its missed stop token.
- **mean similarity below `PASS_RATIO`** (0.70).

Every generation records `stop_token_id`, `stop_token`, `stop_reason`
(`"eog"` / `"max_tokens"`) and `prompt_tokenization_match`. A `max_tokens` exit
is a **diagnostic warning, not a successful generation**: `03_generate.py`
prints the rate at the end of a run and those records stay flagged
(`truncated: true`) for the scorers.

Constraints this gate operates under: the downloaded HF config files are never
edited, the GGUF metadata is never mutated, and the quantized weights are never
touched — the artifact under audit stays byte-identical to the one a user
downloads. The llama.cpp commit used to build the GGUFs is pinned in
`configs/llama_cpp_commit.txt` (written once by `01_quantize.sh`) and printed in
the diagnostic output of both `02b_template_check.py` and `03_generate.py`.

## Colab Pro setup

- Runtime: A100 or L4 for `03_generate.py` (BF16 path) and `08_qlora_repair.py`.
  T4 works for GGUF inference and everything else. `01` and `02` need **no GPU**.
- Sessions die: every script checkpoints per-example to `results/` and resumes.
- Secrets: set `HF_TOKEN` (model download) and `CO_API_KEY` (Cohere, for the
  Command A judge — free trial keys work but are rate-limited; the judge script
  throttles and resumes).

```bash
pip install -r requirements.txt
# llama.cpp with CUDA for fast GGUF inference:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

## Cost controls

- `--max-per-lang N` on every generation/judging script. Full MultiJail is
  315 prompts × 10 langs ≈ 3,150 generations per (model,precision) cell and the
  same number of judge calls. Start with `--max-per-lang 50` for signal,
  scale up only for the final tables.
- Judge calls are cached by content hash; re-runs are free.

## Honest-framing notes (bake into the writeup)

- Prior art to cite, not to erase: Marchisio et al. 2024 (quantization hurts
  multilingual quality unevenly); the Llama3 EN/KO quantization-jailbreak study;
  OpenSafetyMini (quantization × safety, English-centric, non-GGUF methods).
  Our gap: multilingual safety *floor* × shipped GGUF artifacts × a
  safety-flagship on-device model, plus confusion under quantization, plus
  merged-vs-SFT safety fragility (Earth/Fire/Water vs Global).
- Language-adherence claim: Tiny Aya is best on **open-ended generation
  (mDolly)** with lowest cross-language variance; Qwen3.5-4B beats it on
  mArenaHard/GlobalMGSM confusion. Don't overclaim.
- Q4_K_M is mixed-precision: check `02_inspect_tensor_map.py` output before
  telling any "embeddings are the fragile part" story.
- Judge = Command A, matching the paper's MultiJail setup (no translation
  step). Our judge prompt approximates theirs (Appendix C); note this as a
  reproduction caveat and report judge prompt verbatim in the appendix.
