"""Generate MultiJail responses for one precision through llama.cpp.

One backend for every precision, with llama.cpp's own tokenizer and the chat
template embedded in the GGUF -- the path a user running the model on a device
gets. Greedy decoding (paper §4.3).

Why this drives ``llm.generate()`` rather than ``create_chat_completion()``:

* The chat template uses ``{% break %}``, which needs Jinja2's ``loopcontrols``
  extension; llama-cpp-python's chat formatter builds a plain environment and
  raises ``TemplateSyntaxError``. The template is rendered here with the
  extension enabled.
* End-of-turn metadata is inconsistent (generation_config says 3, the tokenizer
  says 6, the model emits 261001). The stop set is resolved from the vocabulary
  by token string and matched on raw token ids; string stops cannot work because
  llama.cpp strips special tokens before matching.
* ``generate(reset=True)`` prefix-matches the KV cache, which changes prefill
  batch shape between calls and flips greedy decoding on near-tied logits. The
  cache is cleared before every prompt.

    python -m tinyaya_eval.generate --precision q4_0 --n-per-lang 315
    python -m tinyaya_eval.generate --precision q4_0 --n-per-lang 315 --langs en,zh,it,vi,ar
    python -m tinyaya_eval.generate --precision q4_0 --probe
"""
import argparse
import gc
import inspect

from tqdm import tqdm

from . import common, config
from .prepare import gguf_path

PROBES = ["What is the capital of France? Answer in one word.",
          "List three colours, comma-separated.",
          "Habari yako? Jibu kwa sentensi moja."]
END_TOKEN_STRINGS = ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>", "<EOS_TOKEN>"]


def _single_token_id(llm, s):
    try:
        t = llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)
    except Exception:  # noqa: BLE001
        return None
    return int(t[0]) if len(t) == 1 else None


class Backend:
    def __init__(self, precision):
        from llama_cpp import Llama
        path = gguf_path(precision)
        if not path.exists():
            raise SystemExit(f"{path} missing; run prepare first")
        self.precision = precision
        self.llm = Llama(model_path=str(path), n_ctx=config.N_CTX, n_batch=config.N_BATCH,
                         n_gpu_layers=-1, verbose=False)
        self.template = self.llm.metadata.get("tokenizer.chat_template")
        if not self.template:
            raise SystemExit("GGUF carries no tokenizer.chat_template; prompts cannot be rendered faithfully")
        bos_id = self.llm.metadata.get("tokenizer.ggml.bos_token_id")
        self.bos = ""
        if bos_id is not None:
            try:
                self.bos = self.llm.detokenize([int(bos_id)], special=True).decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                pass
        self.stop_ids = {}
        for s in END_TOKEN_STRINGS:
            i = _single_token_id(self.llm, s)
            if i is not None:
                self.stop_ids[i] = s
        for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
            v = self.llm.metadata.get(key)
            if v is not None:
                try:
                    self.stop_ids.setdefault(int(v), "<meta:" + key.rsplit(".", 1)[-1] + ">")
                except (TypeError, ValueError):
                    pass
        if not self.stop_ids:
            raise SystemExit("no end-of-turn token resolved; refusing to generate with an open-ended budget")
        want = {"top_k": 1, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0, "temp": 0.0,
                "repeat_penalty": 1.0, "reset": True}
        params = inspect.signature(self.llm.generate).parameters
        self.kw = {k: v for k, v in want.items() if k in params}

    def render(self, user_text):
        from jinja2 import Environment
        env = Environment(extensions=["jinja2.ext.loopcontrols"])

        def _raise(msg):
            raise ValueError(msg)

        env.globals["raise_exception"] = _raise
        return env.from_string(self.template).render(
            messages=[{"role": "user", "content": user_text}],
            add_generation_prompt=True, bos_token=self.bos, eos_token="")

    def generate(self, user_text):
        rendered = self.render(user_text)
        tokens = self.llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)
        budget = max(1, min(config.MAX_NEW_TOKENS, self.llm.n_ctx() - len(tokens) - 8))
        out, finish, stop_token = [], "length", None
        if config.KV_RESET_PER_PROMPT:
            self.llm.reset()
        for tid in self.llm.generate(tokens, **self.kw):
            if tid in self.stop_ids:
                finish, stop_token = "stop", self.stop_ids[tid]
                break
            out.append(int(tid))
            if len(out) >= budget:
                break
        return {"response": self.llm.detokenize(out).decode("utf-8", errors="replace"),
                "finish_reason": finish, "stop_token": stop_token,
                "truncated": finish == "length",
                "n_prompt_tokens": len(tokens), "n_generated_tokens": len(out)}

    def close(self):
        del self.llm
        gc.collect()


def load_prompts(n_per_lang, langs=None):
    from datasets import load_dataset
    ds = load_dataset(config.DATASET, split="train")
    items = []
    for i in common.chosen_indices(n_per_lang):
        row = ds[int(i)]
        for lang in (langs or config.LANGS):
            items.append({"id": f"mj-{i}-{lang}", "lang": lang, "prompt_idx": int(i), "prompt": row[lang]})
    return items


def probe(precision):
    """Preflight: template renders, stop set includes the model's real end-of-turn
    token, generation stops, decoding is deterministic across three runs."""
    b = Backend(precision)
    print(f"--- {precision} ---")
    print(f"  stop ids        : {sorted(b.stop_ids.items())}")
    r = b.render("ping")
    t = b.llm.tokenize(r.encode("utf-8"), add_bos=False, special=True)
    print(f"  template        : renders to {len(t)} tokens for a one-word prompt")
    print(f"  rendered head   : {r[:96]!r}")
    if len(t) > 1200:
        print("  WARNING: far more tokens than the ~370-token preamble implies; special tokens may be literal text")
    elif not 340 <= len(t) <= 430:
        print("  NOTE: outside the expected ~370-400 band")
    ok = True
    for p in PROBES:
        a = b.generate(p)
        same = all(b.generate(p)["response"] == a["response"] for _ in range(2))
        good = same and a["finish_reason"] == "stop"
        ok &= good
        print(f"  [{'ok  ' if good else 'FAIL'}] prompt={a['n_prompt_tokens']} gen={a['n_generated_tokens']} "
              f"finish={a['finish_reason']} stop={a['stop_token']} deterministic={same}")
        print(f"         {a['response'][:110]!r}")
    b.close()
    print(f"  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", required=True, choices=config.PRECISIONS)
    ap.add_argument("--n-per-lang", type=int, default=config.N_LADDER[0])
    ap.add_argument("--langs", default=None, help="comma-separated subset of LANGS")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe:
        raise SystemExit(0 if probe(args.precision) else 1)

    rev = common.revision()
    rd = common.run_dir(rev)
    store = common.JsonlStore(common.gen_path(rev, args.precision))
    langs = args.langs.split(",") if args.langs else None
    items = load_prompts(args.n_per_lang, langs)
    todo = [it for it in items if not store.has(it["id"])]
    print(f"run dir: {rd}")
    print(f"{len(items)} prompts at n={args.n_per_lang}/lang, {len(todo)} to generate")
    if not todo:
        return

    backend = Backend(args.precision)
    print(f"stop ids: {sorted(backend.stop_ids.items())}")
    n_trunc = n_err = 0
    gpu = common.environment().get("gpu")
    for it in tqdm(todo):
        try:
            res = backend.generate(it["prompt"])
        except Exception as e:  # noqa: BLE001
            res = {"response": "", "finish_reason": "error", "stop_token": None, "truncated": False,
                   "n_prompt_tokens": None, "n_generated_tokens": None, "error": str(e)[:300]}
            n_err += 1
        n_trunc += bool(res["truncated"])
        store.add(it["id"], {**it, **res, "model": config.MODEL_NAME, "precision": args.precision,
                             "revision": rev, "gpu": gpu})
    backend.close()
    common.mirror(store.path)
    common.write_manifest(rd, {"revision": rev, "max_n_per_lang_seen": args.n_per_lang})
    print(f"\n{len(todo)} generated | truncated: {n_trunc} | errors: {n_err}")


if __name__ == "__main__":
    main()
