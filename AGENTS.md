# Project Agent Instructions

Read these files before making project changes:

1. `STACK.md` for the stack, package manager, test runner, and common commands.
2. `CONTEXT.md` for domain vocabulary, core entities, examples, and open questions.
3. `.ai/code-style.md`, `.ai/testing.md`, and `.ai/worktree.md` when the file you are editing matches their scope. Claude Code auto-loads these via `.claude/rules/` symlinks (path-scoped by each file's `paths:` header); other agents should read them directly.
4. `DECISIONS.md` (the index) for durable technical decisions that must not be contradicted silently — each line links to a `decisions/<YYYY-MM-DD>-<slug>.md` file with the ~5-line body.
5. `.ai/issue-tracker-github.md` for how issues and handoffs work — the GitHub issue is where a feature's spec, slices, and technical plan live.

`AGENTS.md` is the canonical shared agent instruction file. `CLAUDE.md` exists only so Claude Code can import these same instructions.

The user-facing docs: `README.md` (what it is, quick start), `docs/USAGE.md` (commands, agent apps, files, data
sources), `docs/HOW_IT_WORKS.md` (the model), `docs/VALIDATION.md` (how it is tested), `docs/experiments/` (what
was tried and measured), `docs/FIELD_PROTOCOL.md` (field data collection), `docs/DEVELOPING.md` (setup, code map,
speed), and `src/cougarmap/PLAYBOOK.md` (how an agent should drive the tools and report results).

## Issue state

GitHub issues are the source of truth for what work exists. Each open issue carries at most one
`workflow:*` label (`definition` → `implementation` → `review`); closing the issue means done. See
`.ai/issue-tracker-github.md`.

- **Know what you're working on.** If the branch name starts with an issue number (`<number>-<slug>`), read that
  issue with `gh issue view <number>` before starting. If there's no number, ask or work from the request.
- **Hand off through the issue, not chat.** When a step is finished, comment what's done and what's left, and move
  the label.

## Working Rules

- Use the project language from `CONTEXT.md`; do not invent new terms for existing concepts.
- Treat `STACK.md` frontmatter as machine-readable source of truth. Update it when stack facts change.
- Preserve existing project-specific rules in `.ai/`; add narrow rules instead of replacing them wholesale.
- `./scripts/check.sh` must pass before every commit. Commits carry no `Co-Authored-By` trailers.
- Model changes are measured, not argued: rerun the test areas (`./scripts/rerun_eval.sh`), compare the
  pin-free eval against the baseline in `docs/VALIDATION.md`, run the GPS check, and record what the change did in
  `docs/experiments/` (including negative results).
- Record a new durable technical decision as `decisions/<YYYY-MM-DD>-<slug>.md` (the date it
  was decided; frontmatter with `name`/`date`/`description`/`tags`, an H1 that states what was
  decided, and a ~5-line body), then add its one-line entry to the `DECISIONS.md` index under
  the right area. Keep entries live-only: when a decision is superseded, delete its file and
  its index line — git history is the journal. Before handing off, re-check your diff against
  the `DECISIONS.md` index and flag any departure.
- Keep feature work sliced vertically. A feature's spec, slices, and technical plan live in its GitHub issue (description + comments), not in repo docs.

## No personal data in the repo

This repository is public. Never commit personal or identifying information, in any file, test, fixture, doc,
commit message, branch name, issue or PR description. That includes:

- real coordinates of anyone's cameras, home, property or favorite areas, and bounding boxes or file names that
  encode them (an analysis folder like `48-74...-r3km` is a coordinate);
- street addresses, and place names that point at a private person's land (small towns, roads, creeks and area
  names taken from a user's KML);
- people's names, emails or phone numbers, and who chose, owns or deployed a camera;
- camera names, per-camera results, field-log contents and anything copied from `data/private/` or `out/`.

Use made-up or public examples instead: the synthetic landscape in `tests/synthetic.py` for tests, and public land
far from any user's data (e.g. `47.3712, -116.1029`, `"Missoula, MT"`) in docs. Report private validation results
only in aggregate ("25 human-picked sites in three areas"). Before every commit, read the staged diff
(`git diff --cached`) for coordinates, names and addresses. If something private was committed, stop and tell the
user: removing it means rewriting history, which is their decision.

## Safety

- **Private data never leaves the machine.** `data/private/` (Google Earth files with camera locations, the
  field log, `eval.toml`) and anything under `out/` stay out of git, issues, PRs, commits, and any external
  service.
- Do not read, print, or edit `.env`, `.env.*`, private keys, certificates, service-account files, credential files, or secrets files unless the user explicitly asks and confirms it is safe.
- Prefer project-provided scripts over ad hoc commands.
- When commands write outside the repo or need network access, ask before running them. Data downloads into
  `~/.cache/cougarmap` during an analysis are expected.
