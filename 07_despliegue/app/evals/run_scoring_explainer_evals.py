"""Run regression evals for the scoring explainer.

With ANTHROPIC_API_KEY configured, this evaluates the provider-backed path. Without
it, it evaluates the deterministic safe fallback so the suite is runnable in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainer import answer_question, evaluate_answer_contract, load_eval_cases

SCORE = {
    "score_pd": 0.20,
    "score_ead": 0.80,
    "score_lgd": 0.90,
    "perdida_esperada_relativa": 0.144,
}


def main() -> None:
    cases = load_eval_cases(Path(__file__).with_name("scoring_explainer_cases.json"))
    failures: list[str] = []
    for case in cases:
        reply = answer_question(case["question"], SCORE)
        verdict = evaluate_answer_contract(reply.content, SCORE)
        missing = [fragment for fragment in case["required_fragments"] if fragment.lower() not in reply.content.lower()]
        if not verdict.valid or missing:
            failures.append(f"{case['name']}: {verdict.reason or f'faltan: {missing}'}")
        else:
            print(f"PASS {case['name']} ({reply.mode})")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
