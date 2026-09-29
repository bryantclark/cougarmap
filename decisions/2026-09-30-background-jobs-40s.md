---
name: 2026-09-30-background-jobs-40s
date: 2026-09-30
description: Every tool call returns within about 40 seconds; longer work runs as a background job
tags: [agent, mcp]
---

# Tool calls return within 40 s; long work is a job

Agent apps have short tool timeouts (Cursor ~60 s, Antigravity ~180 s), so long operations return a `job_id`
for `job_status` after at most 40 s. MCP server instructions stay under 2 KB because Claude Code truncates longer
ones; the full guidance is the `playbook` tool and skills. Rules out long blocking tool calls.
