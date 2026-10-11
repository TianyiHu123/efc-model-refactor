"""Run-location resolver.

Implements the repository output policy (README §6, AGENTS.md §1):

    {output_root}/{site}/{case_name}/{mode}/attempt{N}/

- ``mode`` is ``test`` or ``operational`` (kept separate).
- Each run/retry gets its own ``attempt{N}`` subdirectory.
- The next free attempt number is selected automatically.
- Outputs are never written inside the git repo.
"""

from __future__ import annotations

import re
from pathlib import Path

VALID_MODES = ("test", "operational")
_ATTEMPT_RE = re.compile(r"^attempt(\d+)$")


def next_attempt_dir(
    output_root: str | Path,
    site: str,
    case_name: str,
    mode: str,
    create: bool = True,
) -> Path:
    """Return (and by default create) the next free ``attempt{N}`` directory.

    Parameters
    ----------
    output_root : base output root (default caller passes the repo default).
    site, case_name : identify the case subtree.
    mode : ``test`` or ``operational``.
    create : if True, create the directory tree.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")

    base = Path(output_root).expanduser().resolve() / site / case_name / mode
    n = _highest_existing_attempt(base) + 1
    attempt = base / f"attempt{n}"
    if create:
        attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _highest_existing_attempt(base: Path) -> int:
    """Largest N among existing ``attempt{N}`` dirs under ``base`` (0 if none)."""
    if not base.is_dir():
        return 0
    highest = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = _ATTEMPT_RE.match(child.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest
