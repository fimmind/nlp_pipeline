from __future__ import annotations

import subprocess
from pathlib import Path


def test_site_app_contextual_checks() -> None:
    script = Path("tests/site_app_contextual_checks.cjs")
    result = subprocess.run(
        ["node", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "site/app.js contextual checks failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
