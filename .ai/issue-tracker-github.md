# Issue Tracker — GitHub

GitHub issues (`bryantclark/cougarmap`) are the source of truth for what work exists. Agents read and write that
state through `gh`; nothing about an issue's state lives only in a chat or a local file.

## Commands

| Need | Command |
|------|---------|
| List issues | `gh issue list --json number,title,labels,assignees --limit 100` |
| View one issue | `gh issue view <number> --json title,body,labels,comments` |
| Comment | `gh issue comment <number> --body "…"` |
| Set state | `gh issue edit <number> --add-label <new> --remove-label <old>` |
| Close (done) | `gh issue close <number> --comment "…"` |

## Lifecycle state

Each open issue carries at most one `workflow:*` label. Move it when a phase is finished:

| State | Phase |
|-------|-------|
| *(no label)* | New / untriaged |
| `workflow:definition` | What to build: the question, the expected effect, how it will be measured |
| `workflow:implementation` | Plan in the issue, build, `./scripts/check.sh`, eval where the model changes |
| `workflow:review` | PR open; review the diff and the eval numbers |
| *closed* | Merged |

Ideas not yet committed to carry an `idea` label and no `workflow:*` label.

## Binding: which issue is this work?

A branch that starts with an issue number (`<number>-<slug>`, e.g. `42-winter-range`) belongs to that issue; put
`Closes #<number>` in the PR description. No leading number → ask, or work from the request.

## Privacy

Issues and PRs never carry camera coordinates, private KML files, field-log contents, private area or camera
names, or people's names. Describe private validation data in aggregate.
