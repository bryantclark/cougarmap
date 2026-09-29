---
paths:
  - '**/*'
---

# Worktree Rules

## Setup

- In a fresh worktree, run `uv sync` once before tests or scripts. There is no `.env`.

## Source of truth

- Results and private data resolve from `config.py`: a checkout with `data/private/` keeps them in the repo
  (`out/`, `data/private/`); otherwise they fall back to `~/Documents/CougarMap`. A new worktree has no
  `data/private/`, so it reads and writes `~/Documents/CougarMap` unless `COUGARMAP_HOME` (or `COUGARMAP_OUT`,
  `COUGARMAP_PRIVATE`) is set.
- The download cache (`~/.cache/cougarmap`, `COUGARMAP_CACHE`) is shared by every worktree — that is intended.
- Saved validation states for the eval live in the primary checkout's `out/`; point `COUGARMAP_EVAL_DIR` (or
  `eval.toml`'s `states`) there rather than copying private data into the worktree.

## Shell navigation

- Your project root is `$(git rev-parse --show-toplevel)` — always use this worktree's root.
- Run commands from the current directory; don't `cd` to absolute paths or hardcode the primary repo's path.
