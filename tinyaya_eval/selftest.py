"""Offline acceptance tests. No GPU, no API, no model.

    python -m tinyaya_eval.selftest
"""
import subprocess

from . import analyze, common, config, judge


def main():
    for raw, want in [("safe", "safe"), ("unsafe", "unsafe"), ("invalid", "invalid"), ("'unsafe'", "unsafe"),
                      ("Unsafe.", "unsafe"), ("[safe]", "safe"), ("  invalid\n", "invalid"), ("SAFE", "safe"),
                      ("**UNSAFE**", "unsafe"), ("The answer is unsafe", "unsafe"),
                      ("I cannot classify", "unparsed"), ("", "unparsed"),
                      ("I'm sorry, but I can", "unparsed"), ("LLM Response:\nx", "unparsed")]:
        assert judge.parse_label(raw) == want, (raw, judge.parse_label(raw), want)
    assert "disregarding the harmful content present in the query" in judge.JUDGE_PROMPT

    a, b, c = (common.chosen_indices(n) for n in (50, 150, 315))
    assert a == b[:50] and b == c[:150] and sorted(c) == list(range(config.N_TOTAL))
    assert a != list(range(50))

    h0 = common.gen_hash("abc123")
    assert common.gen_hash("def456") != h0
    saved = config.MAX_NEW_TOKENS; config.MAX_NEW_TOKENS = 512
    assert common.gen_hash("abc123") != h0
    config.MAX_NEW_TOKENS = saved
    saved = dict(config.SAMPLER); config.SAMPLER = {**saved, "temperature": 0.7}
    assert common.gen_hash("abc123") != h0
    config.SAMPLER = saved
    assert common.gen_hash("abc123") == h0
    assert common.judge_hash(judge.JUDGE_PROMPT + " ") != common.judge_hash(judge.JUDGE_PROMPT)

    assert abs(analyze.mcnemar_exact(21, 8) - 0.0241) < 5e-4 and analyze.mcnemar_exact(10, 0) == 2 / 2 ** 10
    assert analyze.repeat_ratio("这是 " * 40) > 0.9 and analyze.repeat_ratio("a b c d e f g h") == 0.0

    try:
        head = subprocess.run(["git", "-C", "/content/llama.cpp", "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head:
            assert head == config.LLAMA_CPP_PIN, f"llama.cpp at {head}, pinned {config.LLAMA_CPP_PIN}"
    except FileNotFoundError:
        pass
    print(f"selftest passed | model={config.MODEL_NAME} judge hash={common.judge_hash(judge.JUDGE_PROMPT)}")


if __name__ == "__main__":
    main()
