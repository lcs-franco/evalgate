import datetime
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evalgate.scorers import ScoreResult


class GitInfo(BaseModel):
    commit: str | None
    branch: str | None


class CaseResult(BaseModel):
    id: str
    category: str
    expected_output_hash: str
    output: dict[str, Any] | None
    raw_output: str | None
    scores: list[ScoreResult]
    passed: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float


class CategoryTotals(BaseModel):
    passed: int
    total: int
    pass_rate: float


class Totals(CategoryTotals):
    per_category: dict[str, CategoryTotals]


class RunRecord(BaseModel):
    run_id: str
    feature: str
    prompt_version: str
    prompt_hash: str
    model: str
    golden_dataset_version: str
    git: GitInfo
    timestamp: datetime.datetime
    cases: list[CaseResult]
    totals: Totals


def totals(cases: list[CaseResult]) -> Totals:
    per_category: dict[str, list[bool]] = {}
    for case in cases:
        per_category.setdefault(case.category, []).append(case.passed)
    overall = _rate([case.passed for case in cases])
    return Totals(
        **overall.model_dump(),
        per_category={category: _rate(results) for category, results in per_category.items()},
    )


def _rate(results: list[bool]) -> CategoryTotals:
    passed = sum(results)
    return CategoryTotals(passed=passed, total=len(results), pass_rate=passed / len(results))


def git_info(cwd: Path) -> GitInfo:
    """Reads the commit and branch from GitHub Actions when set, since PR checkouts are detached.

    On a pull request, both `GITHUB_SHA` and the checkout are GitHub's synthetic merge commit,
    so the head commit is read from the event payload instead.
    """
    commit = (
        _pull_request_head_sha() or os.environ.get("GITHUB_SHA") or _git(cwd, "rev-parse", "HEAD")
    )
    branch = (
        os.environ.get("GITHUB_HEAD_REF")
        or os.environ.get("GITHUB_REF_NAME")
        or _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    )
    return GitInfo(commit=commit, branch=branch)


def _pull_request_head_sha() -> str | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    try:
        event = json.loads(Path(event_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sha = event.get("pull_request", {}).get("head", {}).get("sha")
    return sha if isinstance(sha, str) else None


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
