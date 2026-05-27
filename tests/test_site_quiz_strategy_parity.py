from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vocab_book_cli import adaptive_uncertainty_light_random_site_words


def _run_site_quiz(seed: int, quiz_size: int, observed: dict[str, int]) -> list[str]:
    script = REPO_ROOT / "tests" / "site_app_quiz_strategy_checks.cjs"
    result = subprocess.run(
        ["node", str(script), str(seed), str(quiz_size), json.dumps(observed)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "site quiz strategy check failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return json.loads(result.stdout.strip())


def test_adaptive_uncertainty_light_random_matches_cli_reference() -> None:
    model_path = REPO_ROOT / "site" / "data" / "best_grouped_irt_model_model_data.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    words = [str(word) for word in model["words"]]
    accuracy = np.asarray(model["accuracy"], dtype=np.float64)
    candidate_pool = [str(word) for word in model.get("adaptive_candidate_pool", model["query_pool"])]

    observed_cases: list[dict[str, int]] = [
        {},
        {"the": 1, "and": 1, "because": 1, "zygote": 0},
        {"apple": 1, "banana": 1, "Wednesday": 1, "xylophone": 0, "henceforth": 0},
    ]
    seeds = [1, 42, 12345, 314159265]
    quiz_size = 60

    for observed in observed_cases:
        for seed in seeds:
            site_words = _run_site_quiz(seed=seed, quiz_size=quiz_size, observed=observed)
            ref_words = adaptive_uncertainty_light_random_site_words(
                words=words,
                accuracy=accuracy,
                candidate_pool_words=candidate_pool,
                observed_answers=observed,
                quiz_size=quiz_size,
                seed=seed,
            )
            assert site_words == ref_words
