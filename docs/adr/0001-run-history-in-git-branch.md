# Run history lives in an orphan git branch

A candidate run in CI must read the baseline (the latest run on `main`), and trend and drift detection need the last N runs on `main`. CI runners are ephemeral, so the results must persist somewhere outside them. We store each `main` run as a JSON file, together with its HTML report, in an orphan branch `evalgate-history`, and serve that branch with GitHub Pages so every report has a permanent URL for Slack and PR comments.

## Considered Options

- **SQLite in the runner** (as the guide suggests): lost when the job ends. SQLite stays only as an optional local cache.
- **GitHub Actions artifacts**: expire after 90 days, are awkward to fetch across workflows, and need a GitHub login to view.
- **External storage (S3, a database)**: adds infrastructure and credentials to a tool that otherwise needs none.

## Consequences

- The `main` workflow needs write access to the repo (`contents: write`) to push to `evalgate-history`.
- Two `main` runs finishing at the same time can race on the push; the workflow serializes them with a `concurrency` group.
- Candidate runs never write to the branch, so a pull request cannot change the baseline.
