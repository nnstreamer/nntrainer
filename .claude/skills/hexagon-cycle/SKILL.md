---
name: hexagon-cycle
description: Run one supervision → planning → implementation cycle of the HTP MoE decode work on the htp_moe branch, driven by issue labels on dlwlzzero/nntrainer. Args: "--guide" refreshes the decode guide, "--dry-run" only reports what would run.
disable-model-invocation: true
---

You are the orchestrator for one cycle. The contract is
`docs/plans/0001-htp-moe-decode-agent-system.md`; do not restate it, follow it.

Preconditions (check, do not fix silently): current branch is `htp_moe`
or a `htp/*` branch with a clean tree; `source tools/htp/env.sh` prints
its summary line with no `missing` warning; `gh auth status` is logged
in. If the tree is dirty, stop and say what is uncommitted.

## Steps

1. Fetch the open `hexagon` issues (`list_issues`, label `hexagon`).
   Print a one-line board: counts per `state:*` and the `needs-user` list.
   With `--dry-run`, stop here and say which subagents would run.
2. Spawn `hexagon-supervisor` (subagent) with the board. Wait. If it
   edited docs, commit them on `htp_moe` as `[docs] ...` with `-s` (docs
   only; refuse if the diff touches anything else).
3. If any issue is `state:needs-plan` and not `needs-user`: spawn
   `hexagon-planner` with the highest-priority one. Wait.
4. If no issue is `state:in-progress` (a `state:needs-measurement` issue
   does not block, as long as the new issue needs no device step): pick
   the highest-priority `state:planned` (or a `state:measured` the
   supervisor handed back as `in-progress`) and spawn
   `hexagon-implementer` with it. Wait. If an issue is `state:in-progress`
   with unanswered PR review comments, spawn the implementer on it instead.
5. If `--guide` was given, or a PR into `htp_moe` merged since the last
   guide commit (compare `git log -1 -- docs/htp_moe/guide` with merges on
   `htp_moe`), or a measurement handoff was filled since that commit:
   spawn `hexagon-guide-writer`, naming the measurement files (with
   `<branch>:<path>` for ones not yet on `htp_moe`). Wait. Commit its
   output on a `htp/guide-<date>` branch and open a PR (docs only).
6. Final report to the user, in this shape and nothing more:
   * what each subagent did (one line each, issue numbers and PR links),
   * **Your to-do**: every `needs-measurement` handoff (path + estimated
     minutes), every PR waiting for merge, every `needs-user` question,
     and new commits on upstream PR #4327 if the supervisor reported any,
   * whether another cycle would do anything right now (yes/no and why).

Never run `adb`, never push to `htp_moe` except the docs-only commits in
steps 2 and 5, never edit `.github/workflows`.
