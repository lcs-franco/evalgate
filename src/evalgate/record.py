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


def compute_totals(cases: list[CaseResult]) -> Totals:
    results_by_category: dict[str, list[bool]] = {}

    for case in cases:
        results_by_category.setdefault(case.category, []).append(case.passed)

    overall = _pass_rate_of([case.passed for case in cases])

    return Totals(
        **overall.model_dump(),
        per_category={
            category: _pass_rate_of(results) for category, results in results_by_category.items()
        },
    )


def _pass_rate_of(results: list[bool]) -> CategoryTotals:
    passed = sum(results)

    return CategoryTotals(passed=passed, total=len(results), pass_rate=passed / len(results))


def read_git_info(repo_dir: Path) -> GitInfo:
    # On a pull request, GITHUB_SHA and the checkout are GitHub's synthetic merge commit,
    # which exists nowhere in the repo, so the head commit comes from the event payload.
    commit = (
        _pull_request_head_sha()
        or os.environ.get("GITHUB_SHA")
        or _run_git(repo_dir, "rev-parse", "HEAD")
    )

    branch = (
        os.environ.get("GITHUB_HEAD_REF")
        or os.environ.get("GITHUB_REF_NAME")
        or _run_git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    )

    return GitInfo(commit=commit, branch=branch)


def _pull_request_head_sha() -> str | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if not event_path:
        return None

    try:
        github_event = json.loads(Path(event_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None

    head_sha = github_event.get("pull_request", {}).get("head", {}).get("sha")

    return head_sha if isinstance(head_sha, str) else None


def _run_git(repo_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip() or None
