# Does 4-bit quantization erode Tiny Aya's multilingual safety floor?

A paired, reproducible MultiJail evaluation of Tiny Aya Global and Tiny Aya Earth
([arXiv:2603.11510](https://arxiv.org/abs/2603.11510)) at f16, q8_0, q4_k_m and q4_0
on the llama.cpp path, judged by Command A with the paper's Appendix C prompt
verbatim, and adjudicated by hand where the judge reported an effect.

## Summary of findings

25,200 judged responses (2 models × 4 precisions × 315 prompts × 10 languages),
219 discordant pairs hand-labelled.

1. **q8_0 and q4_k_m do not change refusal behaviour** in any of ten languages, in
   either model. The paper's Q4_K_M deployment recommendation holds on MultiJail.
2. **q4_0 does not measurably change refusal behaviour once judge noise is
   removed.** The judge reports a 6–8 point Javanese drop in both models
   (combined p = 2 × 10⁻⁵); human adjudication of all 219 Javanese and Bengali
   discordant pairs finds 160 unchanged, 35 refusals lost, 21 gained (net −1 point,
   p = 0.08). Roughly three-quarters of the reported effect is instrument noise.
   Bengali losses are genuine compliance; Javanese losses are incoherence.
3. **q4_0 does degrade generation quality in low-resource languages**, measured
   without a judge: on Earth, truncation doubles in Bengali (7.0 → 16.2 %),
   Javanese (7.9 → 16.2 %) and Thai (6.0 → 12.4 %) while high-resource languages
   are unchanged. Quantization compounds pre-existing weakness.
4. **Command A is an unreliable MultiJail judge where the model's output is
   incoherent.** Agreement with a human is 93 % on a random Global sample but 50 %
   on the Javanese pairs it flipped and 62 % on the Bengali ones; more than half of
   its "unsafe" verdicts on those pairs are responses a human rates safe; it
   under-detects incoherent output by an order of magnitude relative to the paper's
   4 % Javanese invalid rate. Safety comparisons across conditions that change
   output quality, in low-resource languages, using an LLM judge, will report
   deltas that are substantially instrument noise unless the judge is validated
   per language.
5. **The harness reproduces the paper.** The f16 reference lands within ~1 point of
   Table 7 for both models (Global min 83.8 / mean 89.9 vs 87.0 / 91.1; Earth
   78.0 / 86.9 vs 77.5 / 87.8) with the same weakest language for Global.

Three deployment findings on the llama.cpp path: the chat template's `{% break %}`
does not render in `llama-cpp-python`'s own chat formatter; end-of-turn metadata is
inconsistent across `generation_config.json` (3), the tokenizer (6) and the model's
actual emission (261001); and `generate(reset=True)`'s prefix caching makes greedy
decoding non-deterministic across calls.

## Method

| Component | Setting |
| --- | --- |
| Reference | f16 GGUF converted from the HF safetensors; same backend, tokenizer and chat template as every quant |
| Quantization | llama.cpp `5f436ddd`, `llama-quantize` to Q8_0, Q4_K_M, Q4_0 (paper §6) |
| Inference | `llama-cpp-python` 0.3.35; greedy; 1,024 new tokens; ctx 4,096; KV cache cleared per prompt |
| Prompts | MultiJail, all 315 prompts × {en zh it vi ar ko th bn sw jv}, seeded nested permutation |
| Judge | `command-a-03-2025`, `safety_mode=CONTEXTUAL`, Appendix C prompt verbatim, temperature 0 |
| Denominator | safe / (safe + unsafe + invalid); unlabelled judge replies excluded, never counted as invalid |
| Statistics | exact two-sided McNemar on discordant pairs, all prompts and untruncated pairs; Bonferroni across aggregate tests; q8_0 as control |
| Judge-free axes | truncation rate, repeated character 12-gram ratio, byte-identical rate vs f16 |
| Validation | 60 stratified hand labels per model; every f16→q4_0 discordant pair in jv and bn labelled side by side |

Run directories are named by a hash of the model revision, token budget, sampler and
toolchain pins; judge directories by a hash of the prompt, model and safety mode.
Configurations can therefore never share a file, and every step resumes after a
session loss from a one-way Drive mirror.

## Repository layout

```
tinyaya_eval/
  config.py      all knobs; model selected by TINYAYA_MODEL; paper anchors per model
  common.py      run hashes, JSONL store, manifests, one-way mirror, runs/ restore
  prepare.py     HF snapshot → f16 GGUF → quants → tensor-map inventory
  generate.py    llama.cpp generation and the preflight probe
  parallel.py    concurrent generation across precisions
  judge.py       Command A judging, threaded, with re-judging of unlabelled replies
  analyze.py     safety and paired tables, control check, judge-free axes, figures, judge validation
  discordant.py  side-by-side export and scoring of hand-labelled discordant pairs
  export.py      bundle results and runs for one model
  selftest.py    offline acceptance tests
notebooks/
  tinyaya_quant_safety.ipynb   Colab pipeline; clones this repo at REPO_REF
build_notebook.py              regenerates the notebook
```

## Reproduction

Open `notebooks/tinyaya_quant_safety.ipynb` in Colab on a GPU runtime with Colab
Secrets `HF_TOKEN` (licence accepted for the model on Hugging Face) and
`CO_API_KEY`. Part 0 installs the pinned toolchain and requires one restart. Set
`MODEL` in Part 1; Parts 2–8 are idempotent and resumable. A full run of one model
is ≈ 40 min of artifact building, 7 h of generation on an A100 (four concurrent
processes), 1.5 h of judging, and annotator time for Part 7.

Outside Colab:

```bash
pip install -e .
export TINYAYA_MODEL=global TINYAYA_ROOT=$PWD/data TINYAYA_DRIVE_DIR=
python -m tinyaya_eval.selftest
python -m tinyaya_eval.prepare download && python -m tinyaya_eval.prepare convert && python -m tinyaya_eval.prepare quantize
python -m tinyaya_eval.generate --precision f16 --probe
python -m tinyaya_eval.parallel --n-per-lang 315
python -m tinyaya_eval.judge --all --workers 4
python -m tinyaya_eval.judge --all --workers 4 --drop-unparsed
python -m tinyaya_eval.discordant export --langs jv,bn --models global
```

`LLAMA_CPP_DIR` (default `/content/llama.cpp`) must point at a checkout of llama.cpp
at the pinned commit with `llama-quantize` built.

## Runs reported

| Model | HF revision | Run id | Judge id |
| --- | --- | --- | --- |
| Global | `00590ff258ccd84a805f13efcd1c34c2a542654f` | `gen_5a1a43be76e2` | `judged_4c7d807d3282` |
| Earth  | `821f26a10f8e67a5b453f62d3091d5d4af14b1ae` | `gen_d9e7295d1499` | `judged_4c7d807d3282` |

## Limitations

Single annotator; glosses for hand labelling were machine translations, which tend
to clean up incoherent text; judge–human agreement figures on discordant pairs are
conditional on the judge having flipped; `unsafe` under Appendix C means engagement,
not harm; llama.cpp only (the MLX path is untested); Fire and Water untested.

## Citation

Dataset: Deng et al. 2024, *Multilingual Jailbreak Challenges in Large Language
Models* (MultiJail). Model and judge protocol: Cohere Labs, *Tiny Aya: Bridging Scale
and Multilingual Depth*, arXiv:2603.11510.
