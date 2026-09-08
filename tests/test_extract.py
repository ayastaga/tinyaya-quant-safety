"""Unit tests for the Eval D answer extractor. FREEZE the extractor once these
pass plus a hand-check on >=20 real BF16 outputs per script family.

Run: python tests/test_extract.py   (no pytest dependency needed)
"""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))  # so 04b's `from common import ...` resolves

spec = importlib.util.spec_from_file_location("score_mgsm", SCRIPTS / "04b_score_mgsm.py")
m = importlib.util.module_from_spec(spec)
sys.modules["score_mgsm"] = m
spec.loader.exec_module(m)

CASES = [
    # (response, expected_extracted_value)
    ("The answer is 42.", 42.0),
    ("Step 1: 5*3=15. Step 2: 23-15=8.\n8", 8.0),
    ("Final answer: 1,234", 1234.0),                       # thousands comma
    ("Cela fait 3,5 au total", 3.5),                        # decimal comma
    ("মোট 4 * 5 = 20টি, তাই 9 + 20 = ২৯", 29.0),            # Bengali digits
    ("คำตอบคือ ๑๕", 15.0),                                  # Thai digits
    ("जवाब है ४२", 42.0),                                    # Devanagari digits
    ("సమాధానం ౧౨", 12.0),                                    # Telugu digits
    ("الإجابة هي ٢٥", 25.0),                                 # Arabic-Indic digits
    ("جواب ۱۷ ہے", 17.0),                                    # Extended Arabic-Indic (Urdu)
    ("It costs $1 250 in total", 1250.0),                   # space thousands sep
    ("負の答え: -7", -7.0),                                  # negative
    ("The result is 29. Wait, no — 30.", 30.0),             # last number wins
    ("No numbers here at all.", None),
    ("", None),
]

GOLD_CASES = [
    (29.0, 29, True),
    (29.4, 29, False),
    (None, 29, False),
    (1234.0, 1234, True),
]


def main():
    failures = 0
    for text, expected in CASES:
        got = m.extract_answer(text)
        ok = (got is None and expected is None) or (
            got is not None and expected is not None and abs(got - expected) < 1e-9)
        status = "ok " if ok else "FAIL"
        failures += (not ok)
        print(f"[{status}] {text[:45]!r:48s} -> {got} (want {expected})")
    for pred, gold, expected in GOLD_CASES:
        ok = m.is_correct(pred, gold) == expected
        failures += (not ok)
        print(f"[{'ok ' if ok else 'FAIL'}] is_correct({pred}, {gold}) == {expected}")
    if failures:
        print(f"\n{failures} FAILURES")
        sys.exit(1)
    print("\nAll extractor tests pass.")


if __name__ == "__main__":
    main()
