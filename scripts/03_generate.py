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
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from common import JsonlStore, load_config, result_path

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


class HFBackend:
    def __init__(self, cfg, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = f'{cfg["paths"]["hf_cache"]}/{model_name}'
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto")
        self.cfg = cfg

    def generate(self, prompt, max_new_tokens=None):
        import torch
        budget = max_new_tokens or self.cfg["generation"]["max_new_tokens"]
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=budget, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        new = out[0][ids.shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return text, bool(len(new) >= budget)   # (response, truncated)


class GGUFBackend:
    def __init__(self, cfg, model_name, precision):
        from llama_cpp import Llama
        path = f'{cfg["paths"]["gguf_dir"]}/{model_name}-{precision}.gguf'
        self.llm = Llama(model_path=path, n_ctx=cfg["generation"]["context"],
                         n_gpu_layers=-1, verbose=False)
        self.cfg = cfg

    def generate(self, prompt, max_new_tokens=None):
        budget = max_new_tokens or self.cfg["generation"]["max_new_tokens"]
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=budget, temperature=0.0)
        choice = out["choices"][0]
        return choice["message"]["content"], choice.get("finish_reason") == "length"


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

    backend = (HFBackend(cfg, args.model) if args.precision == "bf16"
               else GGUFBackend(cfg, args.model, args.precision))
    budget = budget_for(cfg, args.eval)

    for it in tqdm(todo):
        try:
            text, truncated = backend.generate(it["prompt"], max_new_tokens=budget)
        except Exception as e:  # noqa: BLE001
            text, truncated, it["error"] = "", False, str(e)
        store.add(it["id"], {**it, "response": text, "truncated": truncated,
                             "model": args.model, "precision": args.precision})


if __name__ == "__main__":
    main()
