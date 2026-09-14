"""Generate model outputs for one (model, precision, eval) cell. Resumable.

Backends:
  bf16              -> HF transformers (A100/L4)
  q8_0/q4_k_m/q4_0/q3_k_m/q2_k -> llama-cpp-python on GGUFs (T4 fine, n_gpu_layers=-1)

Evals:
  multijail   -> raw MultiJail prompts (paper's unintentional-harm setting)
  mdolly      -> aya_evaluation_suite dolly_machine_translated
  flores      -> en->X translation, paper-style template
  globalmgsm  -> (v2, Eval D) MGSM CoT in target language; 1024-token budget;
                 logs `truncated` flag. Score with 04b (no judge).
  toolformat  -> (v2, Eval E) prompts from data/toolformat_prompts.jsonl
                 (built by 10_build_toolformat_prompts.py). Free generation,
                 NO grammar constraint (would mask the degradation). Score 04c.
  xstest      -> over-refusal check (English, 250 prompts). Judge with 04d.

Usage:
  python 03_generate.py --model global --precision q4_0 --eval globalmgsm --max-per-lang 100
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from common import JsonlStore, llama_cpp_pin, load_config, result_path

TRANSLATION_TEMPLATE = (
    "Translate the following text into {target_language}. "
    "Provide only the translation.\n\nText: {text}\n\nTranslation:"
)

FLORES_CODES = {
    "en": "eng_Latn", "es": "spa_Latn", "de": "deu_Latn", "ru": "rus_Cyrl",
    "ar": "arb_Arab", "he": "heb_Hebr", "tr": "tur_Latn", "hi": "hin_Deva",
    "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Arab",
    "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "zh": "zho_Hans", "sw": "swh_Latn", "yo": "yor_Latn",
    "ha": "hau_Latn", "am": "amh_Ethi", "zu": "zul_Latn",
}
LANG_NAMES = {
    "es": "Spanish", "de": "German", "ru": "Russian", "ar": "Arabic",
    "he": "Hebrew", "tr": "Turkish", "hi": "Hindi", "bn": "Bengali",
    "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "th": "Thai",
    "vi": "Vietnamese", "id": "Indonesian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "sw": "Swahili", "yo": "Yoruba", "ha": "Hausa",
    "am": "Amharic", "zu": "Zulu",
}

# Zero-shot CoT instruction, in the prompt language (Eval D). The final-answer
# line is excluded from CoT-adherence scoring, so the numeric-only instruction
# does not contaminate the metric. TODO: native-speaker check these strings;
# they were model-written. Keep them frozen once checked.
MGSM_COT_INSTRUCTIONS = {
    "en": "Solve the problem step by step. On the last line, write only the final number.",
    "es": "Resuelve el problema paso a paso. En la última línea, escribe solo el número final.",
    "de": "Löse die Aufgabe Schritt für Schritt. Schreibe in die letzte Zeile nur die endgültige Zahl.",
    "ru": "Решите задачу шаг за шагом. В последней строке напишите только итоговое число.",
    "zh": "请逐步解答这道题。最后一行只写最终的数字。",
    "ja": "問題を段階的に解いてください。最後の行には最終的な数字だけを書いてください。",
    "th": "แก้ปัญหาทีละขั้นตอน บรรทัดสุดท้ายให้เขียนเฉพาะตัวเลขคำตอบสุดท้ายเท่านั้น",
    "sw": "Tatua tatizo hatua kwa hatua. Katika mstari wa mwisho, andika nambari ya mwisho tu.",
    "bn": "ধাপে ধাপে সমস্যাটি সমাধান করুন। শেষ লাইনে শুধুমাত্র চূড়ান্ত সংখ্যাটি লিখুন।",
    "te": "సమస్యను దశలవారీగా పరిష్కరించండి. చివరి పంక్తిలో తుది సంఖ్యను మాత్రమే రాయండి.",
}


def load_prompts(cfg, eval_name, max_per_lang):
    """Return list of dicts: {id, lang, prompt, [reference|gold|schema|task]}."""
    items = []
    if eval_name == "multijail":
        ds = load_dataset(cfg["datasets"]["multijail"], split="train")
        langs = [l for l in cfg["multijail_langs"] if l in ds.column_names]
        for i, row in enumerate(ds):
            for lang in langs:
                items.append({"id": f"mj-{i}-{lang}", "lang": lang, "prompt": row[lang]})
    elif eval_name == "mdolly":
        ds = load_dataset(cfg["datasets"]["mdolly"], "dolly_machine_translated", split="test")
        for i, row in enumerate(ds):
            lang = row.get("language", row.get("lang", "unknown"))
            if lang not in cfg["eval_langs"]:
                continue
            items.append({"id": f"dolly-{i}-{lang}", "lang": lang, "prompt": row["inputs"]})
    elif eval_name == "flores":
        src = load_dataset(cfg["datasets"]["flores"], FLORES_CODES["en"], split="devtest")
        for lang in cfg["eval_langs"]:
            if lang == "en":
                continue
            tgt = load_dataset(cfg["datasets"]["flores"], FLORES_CODES[lang], split="devtest")
            for i, (s, t) in enumerate(zip(src, tgt)):
                items.append({
                    "id": f"flores-{i}-{lang}", "lang": lang,
                    "prompt": TRANSLATION_TEMPLATE.format(
                        target_language=LANG_NAMES[lang], text=s["text"]),
                    "reference": t["text"]})
    elif eval_name == "globalmgsm":
        for lang in cfg["mgsm_langs"]:
            ds = load_dataset(cfg["datasets"]["mgsm"], lang, split="test")
            instr = MGSM_COT_INSTRUCTIONS[lang]
            for i, row in enumerate(ds):
                items.append({
                    "id": f"mgsm-{i}-{lang}", "lang": lang,
                    "prompt": f'{row["question"]}\n\n{instr}',
                    "gold": row["answer_number"]})
    elif eval_name == "toolformat":
        path = Path(cfg["paths"]["data"]) / "toolformat_prompts.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                "Run scripts/10_build_toolformat_prompts.py first (Eval E prompt source).")
        with open(path) as f:
            for line in f:
                items.append(json.loads(line))
    elif eval_name == "xstest":
        ds = load_dataset(cfg["datasets"]["xstest"], split="train")  # VERIFY split/fields
        for i, row in enumerate(ds):
            items.append({"id": f"xs-{i}-en", "lang": "en", "prompt": row["prompt"],
                          "xstest_type": row.get("type", "")})
    else:
        raise ValueError(eval_name)

    if max_per_lang:
        seen, kept = {}, []
        for it in items:
            c = seen.get(it["lang"], 0)
            if c < max_per_lang:
                kept.append(it)
                seen[it["lang"]] = c + 1
        items = kept
    return items


def budget_for(cfg, eval_name):
    return (cfg["generation_cot"]["max_new_tokens"] if eval_name == "globalmgsm"
            else cfg["generation"]["max_new_tokens"])


# Tokens that end an assistant turn in the Cohere/Aya chat format. The shipped
# generation_config.json is '_from_model_config' (auto-derived, eos_token_id=3)
# and does NOT match the tokenizer, which uses <|END_OF_TURN_TOKEN|>. Without
# this override HF never stops, burns the full budget on every prompt, and the
# decoded response contains hallucinated extra turns.
STOP_TOKEN_STRINGS = ["<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>", "<EOS_TOKEN>"]

# The GGUF path must stop on exactly the same ids as HF. The GGUF metadata marks
# only <|END_OF_TURN_TOKEN|> (6) as EOG -- <EOS_TOKEN> (3) and <|END_RESPONSE|>
# (261001) are not marked -- so llama.cpp's own EOG check is not enough and
# string `stop=` sequences never fire on the special-token spellings. We resolve
# the ids from the original HF tokenizer and compare raw generated ids instead.
# We do NOT edit the GGUF metadata: the audited artifact stays byte-identical to
# what a user downloads.
GGUF_STOP_TOKEN_STRINGS = ["<EOS_TOKEN>", "<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"]


def _token_id(tok, s):
    """Vocab id for a special-token spelling, or None if it does not resolve."""
    i = tok.get_vocab().get(s)
    if i is None:
        i = tok.convert_tokens_to_ids(s)
    if i is None or i == tok.unk_token_id:
        return None
    return int(i)


def gen_result(text, *, stop_token_id=None, stop_token=None, stop_reason="max_tokens",
               prompt_tokenization_match=None, n_generated_tokens=None):
    """One generation plus the diagnostics that say whether to trust it."""
    return {
        "response": text,
        # truncated == the model ran out of budget without emitting a stop
        # token. Kept for 04b/07, which already key on it. Errors get
        # stop_reason "error" and are not counted as truncations.
        "truncated": stop_reason == "max_tokens",
        "stop_token_id": stop_token_id,
        "stop_token": stop_token,
        "stop_reason": stop_reason,
        "prompt_tokenization_match": prompt_tokenization_match,
        "n_generated_tokens": n_generated_tokens,
    }


def resolve_stop_ids(tok):
    """Token ids that should terminate generation. Raises if none resolve."""
    ids, vocab = [], tok.get_vocab()
    for s in STOP_TOKEN_STRINGS:
        i = vocab.get(s)
        if i is None:
            i = tok.convert_tokens_to_ids(s)
        if i is not None and i != tok.unk_token_id:
            ids.append(int(i))
    if tok.eos_token_id is not None:
        ids.append(int(tok.eos_token_id))
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError(
            "No stop tokens resolved; refusing to generate with an open-ended budget.")
    return ids


class HFBackend:
    def __init__(self, cfg, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = f'{cfg["paths"]["hf_cache"]}/{model_name}'
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto")
        self.cfg = cfg

        self.stop_ids = resolve_stop_ids(self.tok)
        pad = self.tok.pad_token_id
        gc = self.model.generation_config
        gc.eos_token_id = self.stop_ids
        gc.pad_token_id = int(pad) if pad is not None else self.stop_ids[0]
        if self.tok.bos_token_id is not None:
            gc.bos_token_id = int(self.tok.bos_token_id)
        print(f"[HFBackend] stop ids: "
              f"{[(i, self.tok.convert_ids_to_tokens(i)) for i in self.stop_ids]}, "
              f"pad={gc.pad_token_id}")

    def prompt_token_ids(self, prompt):
      """(rendered prompt string, prompt token ids) -- the parity reference."""
      msgs = [{"role": "user", "content": prompt}]
      rendered = self.tok.apply_chat_template(
          msgs, tokenize=False, add_generation_prompt=True)
      ids = self.tok.apply_chat_template(
          msgs, tokenize=True, add_generation_prompt=True)
  
      # Transformers may return a list, tensor, or BatchEncoding.
      if hasattr(ids, "keys"):
          ids = ids["input_ids"]
  
      if hasattr(ids, "tolist"):
          ids = ids.tolist()
  
      if ids and isinstance(ids[0], (list, tuple)):
          ids = ids[0]
  
      return rendered, [int(i) for i in ids]


    def generate(self, prompt, max_new_tokens=None):
        import torch
        budget = max_new_tokens or self.cfg["generation"]["max_new_tokens"]
        msgs = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=budget, do_sample=False,
                                      eos_token_id=self.stop_ids,
                                      pad_token_id=self.model.generation_config.pad_token_id)
        new = out[0][enc["input_ids"].shape[1]:].tolist()
        stop_id = next((t for t in new if t in self.stop_ids), None)
        # Decode only the response, excluding the stopping token.
        body = new[:new.index(stop_id)] if stop_id is not None else new
        text = self.tok.decode(body, skip_special_tokens=True)
        return gen_result(
            text,
            stop_token_id=stop_id,
            stop_token=(self.tok.convert_ids_to_tokens(stop_id)
                        if stop_id is not None else None),
            stop_reason="eog" if stop_id is not None else "max_tokens",
            prompt_tokenization_match=True,  # HF is the reference tokenization
            n_generated_tokens=len(body))


class GGUFBackend:
    """llama.cpp on a GGUF, driven at the token level.

    create_chat_completion is not used: it re-renders the prompt with the chat
    template baked into the GGUF and can only stop on string sequences, which
    never match the special-token spellings and leave tokens 3 / 261001 (not
    marked EOG in the metadata) unhandled -- the model then runs to max_tokens
    and repeats itself. Instead we render with the original HF tokenizer,
    tokenize that exact string, and stop on raw generated ids.
    """

    def __init__(self, cfg, model_name, precision):
        from llama_cpp import Llama
        from transformers import AutoTokenizer
        # Read-only: the HF checkpoint and the GGUF are both left untouched.
        self.tok = AutoTokenizer.from_pretrained(
            f'{cfg["paths"]["hf_cache"]}/{model_name}')
        path = f'{cfg["paths"]["gguf_dir"]}/{model_name}-{precision}.gguf'
        self.llm = Llama(model_path=path, n_ctx=cfg["generation"]["context"],
                         n_gpu_layers=-1, verbose=False)
        self.cfg = cfg

        self.stop_ids = {}
        missing = []
        for s in GGUF_STOP_TOKEN_STRINGS:
            i = _token_id(self.tok, s)
            if i is None:
                missing.append(s)
            else:
                self.stop_ids[i] = s
        if missing:
            raise RuntimeError(
                f"Stop tokens {missing} do not resolve in the HF tokenizer at "
                f'{cfg["paths"]["hf_cache"]}/{model_name}; refusing to generate '
                "with an open-ended budget.")
        self._warned_tokenization = False
        print(f"[GGUFBackend] {model_name}-{precision} | llama.cpp pin: "
              f"{llama_cpp_pin()} | stop ids: {sorted(self.stop_ids.items())}")

    def prompt_token_ids(self, prompt):
        """(rendered prompt string, llama.cpp prompt token ids)."""
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        ids = self.llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)
        return rendered, [int(i) for i in ids]

    def _greedy_kwargs(self):
        """Greedy sampling args that this llama-cpp-python version accepts."""
        want = {"top_k": 1, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
                "temp": 0.0, "repeat_penalty": 1.0, "reset": True}
        params = inspect.signature(self.llm.generate).parameters
        return {k: v for k, v in want.items() if k in params}

    def _detokenize(self, tokens):
        if not tokens:
            return ""
        try:
            raw = self.llm.detokenize(tokens, special=False)
        except TypeError:  # older llama-cpp-python: no `special` kwarg
            raw = self.llm.detokenize(tokens)
        return raw.decode("utf-8", errors="replace")

    def generate(self, prompt, max_new_tokens=None):
        budget = max_new_tokens or self.cfg["generation"]["max_new_tokens"]
        rendered, tokens = self.prompt_token_ids(prompt)
        hf_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        if hasattr(hf_ids, "keys"):
            hf_ids = hf_ids["input_ids"]
        if hasattr(hf_ids, "tolist"):
            hf_ids = hf_ids.tolist()
        if hf_ids and isinstance(hf_ids[0], (list, tuple)):
            hf_ids = hf_ids[0]
        match = [int(i) for i in hf_ids] == tokens
        if not match and not self._warned_tokenization:
            self._warned_tokenization = True
            print("[GGUFBackend] WARNING: llama.cpp prompt tokenization differs "
                  "from the HF tokenizer; BF16/GGUF deltas are not comparable. "
                  "See 02b_template_check.py.", file=sys.stderr)

        # Leave room for the response inside the context window.
        budget = max(1, min(budget, self.llm.n_ctx() - len(tokens) - 1))

        out, stop_id = [], None
        for tid in self.llm.generate(tokens, **self._greedy_kwargs()):
            tid = int(tid)
            if tid in self.stop_ids:
                stop_id = tid
                break
            out.append(tid)
            if len(out) >= budget:  # hard fallback, never the expected exit
                break
        return gen_result(
            self._detokenize(out),  # response tokens only, stop token excluded
            stop_token_id=stop_id,
            stop_token=self.stop_ids.get(stop_id),
            stop_reason="eog" if stop_id is not None else "max_tokens",
            prompt_tokenization_match=match,
            n_generated_tokens=len(out))


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True,
                    choices=cfg["precisions"] + cfg["stress_precisions"])
    ap.add_argument("--eval", required=True, choices=[
        "multijail", "mdolly", "flores", "globalmgsm", "toolformat", "xstest"])
    ap.add_argument("--max-per-lang", type=int, default=None)
    args = ap.parse_args()

    store = JsonlStore(result_path(
        cfg, f"gen__{args.eval}__{args.model}__{args.precision}.jsonl"))
    items = load_prompts(cfg, args.eval, args.max_per_lang)
    todo = [it for it in items if not store.has(it["id"])]
    print(f"{len(items)} prompts, {len(todo)} to generate (rest cached)")
    if not todo:
        return

    print(f"llama.cpp pin: {llama_cpp_pin()}")
    backend = (HFBackend(cfg, args.model) if args.precision == "bf16"
               else GGUFBackend(cfg, args.model, args.precision))
    budget = budget_for(cfg, args.eval)

    n_max_tokens, n_mismatch, n_error = 0, 0, 0
    for it in tqdm(todo):
        try:
            res = backend.generate(it["prompt"], max_new_tokens=budget)
        except Exception as e:  # noqa: BLE001
            res = gen_result("", stop_reason="error")
            it["error"] = str(e)
            n_error += 1
        # A max_tokens exit is a diagnostic warning, not a clean generation: the
        # model never emitted a stop token, so the text is usually truncated or
        # looping. Downstream scorers must be able to see and exclude these.
        if res["stop_reason"] == "max_tokens":
            n_max_tokens += 1
        if res["prompt_tokenization_match"] is False:
            n_mismatch += 1
        store.add(it["id"], {**it, **res,
                             "model": args.model, "precision": args.precision})

    done = len(todo)
    print(f"\n{done} generated | clean stop: {done - n_max_tokens - n_error} | "
          f"max_tokens: {n_max_tokens} | errors: {n_error}", flush=True)
    if n_max_tokens:
        print(f"WARNING: {n_max_tokens}/{done} ({100 * n_max_tokens / done:.1f}%) hit "
              "the token budget without a stop token. These are flagged "
              '(stop_reason="max_tokens", truncated=true) and must NOT be read as '
              "successful generations -- expect truncation and repetition. If the "
              "rate is high, re-run scripts/02b_template_check.py before scoring.",
              file=sys.stderr)
    if n_mismatch:
        print(f"WARNING: {n_mismatch}/{done} prompts tokenized differently in "
              "llama.cpp than in the HF tokenizer; the BF16 comparison is invalid.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
