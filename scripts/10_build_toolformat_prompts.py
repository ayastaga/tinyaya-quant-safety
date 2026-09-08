"""Build Eval E prompts -> data/toolformat_prompts.jsonl. Run ONCE.

This closes the gap the v2 plan left open: Eval E had no prompt source.

Design (matches Sec 4.2 of the handoff):
- 6 realistic on-device tasks, each with a JSON schema (English keys).
- English user-request templates with slot values; the USER REQUEST is
  translated into each format_lang via Command A (one-time, ~cheap:
  n_templates x n_langs calls, cached/resumable). The schema block and the
  "Respond ONLY with JSON" directive stay in ENGLISH — that is the realistic
  deployment condition (an app's fixed suffix) and is constant across
  languages, so cross-language deltas isolate the user-language variable.
- Ladder rung (ii): in-prompt schema + one worked example. Rungs are switched
  by editing DIRECTIVE / dropping the example (rung iii = minimal key:value).

TODO before final runs: native-speaker spot-check ~5 translations per language
(translation noise caveat applies here exactly as it does to the benchmarks).

Usage: CO_API_KEY=... python 10_build_toolformat_prompts.py [--per-lang 60]
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path

from common import JsonlStore, load_config

TASKS = {
    "set_alarm": {
        "schema": {"type": "object",
                   "properties": {"time": {"type": "string"},
                                  "label": {"type": "string"}},
                   "required": ["time", "label"], "additionalProperties": False},
        "templates": [
            "Wake me up at {time} tomorrow for {label}.",
            "Set an alarm for {time}, call it {label}.",
        ],
        "slots": {"time": ["6:30", "7:45", "21:00", "5:15"],
                  "label": ["the gym", "my flight", "morning prayers", "work"]},
    },
    "add_contact": {
        "schema": {"type": "object",
                   "properties": {"name": {"type": "string"},
                                  "phone": {"type": "string"}},
                   "required": ["name", "phone"], "additionalProperties": False},
        "templates": [
            "Save {name} to my contacts, the number is {phone}.",
            "Add a new contact: {name}, phone {phone}.",
        ],
        "slots": {"name": ["Amina", "Rajesh", "Mei", "Kofi"],
                  "phone": ["555-0134", "555-0198", "555-0121", "555-0177"]},
    },
    "create_event": {
        "schema": {"type": "object",
                   "properties": {"title": {"type": "string"},
                                  "date": {"type": "string"},
                                  "time": {"type": "string"}},
                   "required": ["title", "date", "time"],
                   "additionalProperties": False},
        "templates": [
            "Put {title} in my calendar on {date} at {time}.",
        ],
        "slots": {"title": ["dentist appointment", "team meeting", "mom's birthday dinner"],
                  "date": ["March 3", "next Friday", "October 12"],
                  "time": ["14:00", "9:30", "19:00"]},
    },
    "send_message": {
        "schema": {"type": "object",
                   "properties": {"recipient": {"type": "string"},
                                  "text": {"type": "string"}},
                   "required": ["recipient", "text"], "additionalProperties": False},
        "templates": [
            "Text {recipient} that {text}.",
        ],
        "slots": {"recipient": ["my brother", "Fatima", "the landlord"],
                  "text": ["I will be 10 minutes late", "dinner is ready",
                           "the rent was paid today"]},
    },
    "weather_query": {
        "schema": {"type": "object",
                   "properties": {"city": {"type": "string"},
                                  "day": {"type": "string"}},
                   "required": ["city", "day"], "additionalProperties": False},
        "templates": [
            "What's the weather in {city} {day}?",
        ],
        "slots": {"city": ["Nairobi", "Dhaka", "Hanoi", "Rome"],
                  "day": ["today", "tomorrow", "on Sunday"]},
    },
    "set_timer": {
        "schema": {"type": "object",
                   "properties": {"minutes": {"type": "integer"}},
                   "required": ["minutes"], "additionalProperties": False},
        "templates": [
            "Set a timer for {minutes} minutes.",
            "I need a {minutes}-minute countdown.",
        ],
        "slots": {"minutes": [5, 12, 45, 90]},
    },
}

DIRECTIVE = (
    "Extract the user's request into JSON. Respond with ONLY a JSON object "
    "matching this schema (English keys), no other text:\n{schema}\n"
    "Example: for \"Set an alarm for 8:00 called school\" respond "
    "{{\"time\": \"8:00\", \"label\": \"school\"}}\n\nUser request: {request}"
)


def english_requests(per_task_cap):
    for task, spec in TASKS.items():
        combos = []
        keys = list(spec["slots"])
        for values in itertools.product(*(spec["slots"][k] for k in keys)):
            fill = dict(zip(keys, values))
            for t_i, tmpl in enumerate(spec["templates"]):
                combos.append((f"{task}-{t_i}-" + "-".join(map(str, values)),
                               task, tmpl.format(**fill)))
        for row in combos[:per_task_cap]:
            yield row


def translate(client, cfg, text, lang, cache):
    key = f"{lang}|{text}"
    if cache.has(key):
        for r in cache.load_all():
            if r["key"] == key:
                return r["translation"]
    for attempt in range(cfg["judge"]["max_retries"]):
        try:
            r = client.chat(
                model=cfg["judge"]["model"],
                messages=[{"role": "user", "content":
                           f"Translate this short user request into the language "
                           f"with ISO code '{lang}'. Natural, colloquial register. "
                           f"Keep times, dates, numbers and proper names as-is. "
                           f"Reply with ONLY the translation.\n\n{text}"}],
                temperature=0.0, max_tokens=120)
            out = r.message.content[0].text.strip()
            cache.add(key, {"translation": out})
            return out
        except Exception:  # noqa: BLE001
            time.sleep(cfg["judge"]["sleep_between"] * (2 ** attempt))
    raise RuntimeError(f"translation failed: {lang} / {text}")


def main():
    import cohere
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=60)
    args = ap.parse_args()

    per_task = max(1, args.per_lang // len(TASKS))
    rows = list(english_requests(per_task))
    print(f"{len(rows)} English requests x {len(cfg['format_langs'])} languages")

    client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
    cache = JsonlStore(Path(cfg["paths"]["data"]) / "toolformat_translations.jsonl")
    out_path = Path(cfg["paths"]["data"]) / "toolformat_prompts.jsonl"

    with open(out_path, "w") as f:
        for rid, task, en_text in rows:
            for lang in cfg["format_langs"]:
                req = en_text if lang == "en" else translate(client, cfg, en_text, lang, cache)
                schema = TASKS[task]["schema"]
                prompt = DIRECTIVE.format(schema=json.dumps(schema), request=req)
                f.write(json.dumps({
                    "id": f"tf-{rid}-{lang}", "lang": lang, "task": task,
                    "prompt": prompt, "schema": schema,
                    "request_text": req}, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
