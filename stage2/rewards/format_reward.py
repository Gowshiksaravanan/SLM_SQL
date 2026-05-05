import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config_stage2 as cfg
from rewards.utils import completion_text
FORMAT_RE = re.compile(r"<think>.+?</think>\s*\S", re.DOTALL)


def format_reward_func(completions, prompts, **kwargs) -> list[float]:# Returns FORMAT_WEIGHT if output has the required <think>...</think>\n\nSQL structure, else 0
    return [cfg.FORMAT_WEIGHT if FORMAT_RE.search(completion_text(c)) else 0.0 for c in completions]


if __name__ == "__main__":
    cases = [
        ("correct format",
         "<think>I need the orders table.</think>\n\nSELECT * FROM orders",
         True),
        ("missing think tags",
         "SELECT * FROM orders",
         False),
        ("empty think block",
         "<think></think>\n\nSELECT * FROM orders",
         False),
        ("think present but no sql after",
         "<think>I need orders table</think>",
         False),
        ("multi-line reasoning",
         "<think>\n1. Identify tables\n2. Build query\n</think>\n\nSELECT id FROM users",
         True),
    ]

    completions = [c for _, c, _ in cases]
    results = format_reward_func(completions, [""] * len(cases))
    print("format_reward_func tests:")
    all_pass = True
    for (desc, _, expect_pass), score in zip(cases, results):
        passed = (score > 0) == expect_pass
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc:35s} → {score:.1f}")
    print("ALL PASSED" if all_pass else "SOME TESTS FAILED")
