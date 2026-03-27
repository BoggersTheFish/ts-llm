# Contributing to ts-llm

## Scope

Changes should stay focused on the **attractor substrate** and documented training/eval paths unless a broader fix is required.

## Changelog

The attractor workstream log is **[`docs/CHANGELOG.md`](docs/CHANGELOG.md)**. Append entries for notable changes (timestamp, what changed, commands/metrics when relevant, commit message).

## Branches

Feature work for the attractor roadmap typically lives on `dev_route_1_attractor_v1` (from `dev_route_1`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[dev]"  # if optional dev extras are defined
```

## Tests

```bash
.venv/bin/pytest -m "not heavy"   # default: fast tests only
RUN_HEAVY_TESTS=1 .venv/bin/pytest  # long-running / large-dim stress tests
```

Heavy tests are **opt-in** so laptops and CI stay responsive.

## Pushing without melting your laptop

TinyStories archives and extracted JSON shards **must not** be committed (they were removed from Git history). See [`data/tinystories/README.md`](data/tinystories/README.md).

Use low-compression, single-threaded push:

```bash
./scripts/push_safe.sh -u origin dev_route_1_attractor_v1
```

Or once: `git config --global pack.compression 0` and `git config --global pack.threads 1`.

## Style

- Match existing typing and module layout.
- Prefer small, reviewable commits with conventional prefixes: `feat(attractor):`, `test(attractor):`, `docs(attractor):`.
