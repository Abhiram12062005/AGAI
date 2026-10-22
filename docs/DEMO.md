# Live demo script (~8 minutes)

## Before you present (5 min, once)

1. `.venv\Scripts\activate`, and check `ANTHROPIC_API_KEY` in `.env`.
2. Warm the caches. The first run downloads the MiniLM embedder (~90 MB):
   `codefix demo py04-reserve-exact-stock --llm mock`
3. `codefix serve` and open http://127.0.0.1:8000 in one tab, with the Dashboard in another.
4. Have a benchmark report open: `benchmarks/results/<latest live bench>/report.md`.
5. Backup if the network or API fails: the identical flow works with **LLM = mock** (an
   offline oracle; tools, workspaces, tests and Blackboard are all real). Say so out loud
   when you use it.

## 1. The problem (30 s)

Open **New fix**. Pick the seeded bug **py05-region-case**: "Checkout crashes with
KeyError: 'CA'". Point out the model table, the per-role reasoning, and the compiled graph
image (the Manager hub, the cycle, the human gate on `open_pr`).

## 2. Run it, hierarchical (3 min)

Click **Run**. On the live run view:

- **Pipeline**: nodes light up as envelopes arrive. Manager → Diagnoser is a *dispatch*;
  the hot edge is the latest message.
- **ReAct transcript**: Thought / Action / Observation streams in. Note it starts from the
  parsed stack trace and the retrieved chunks. The point to make: the crash site (`tax_for`,
  where the KeyError is raised) is not the defect (`normalize_region`), so the agent has to
  read the code rather than blame the innermost frame.
- **Diagnosis**: root cause, files, confidence, **citation chips** (click one to show the
  cited source lines).
- **Patch** and **Tests**: the diff and the Tester's verdict from an isolated clone.
- If a retry happens: the attempt tabs show ✖ attempt 1 → ✔ attempt 2, and Manager
  decisions shows `tests_failed → fixer`. The dispatch carried the TestReport **by
  reference** (open the message log).
- **Outcome**: "Nothing has been pushed." Click **Open Pull Request…** to show the explicit
  confirmation (review checkbox). For a repo without a remote it reports that the branch
  stays local.

## 3. Show the audit point (30 s)

Click **state dump**: the complete Blackboard state at this instant. In a terminal:
`codefix trace <task_id> --react` prints the persisted ReAct transcript.

## 4. Same bug, other architectures (1 min)

Run **py11-sku-dashes** with `sequential`. A failure restarts from the Diagnoser, with no
feedback, so the Fixer prompt never contains the failure. Then `p2p`: the Tester hands the
failure straight to the Fixer, with no Manager node (greyed out in the diagram).

## 5. Dashboard (2 min)

- **Bottleneck**: per-agent mean latency, with the slowest agent flagged.
- **Accuracy weak point**: localisation (Diagnoser found the right file) vs. repair given
  correct localisation (Fixer). The lower bar is where to invest.
- **Architecture comparison**: resolved %, judge pass %, judge accuracy, attempts, cost.
- **Judge vs. tests**: the judge never sees test results; agreement is the judge's accuracy.
- **Per-run timeline**: Gantt per agent; failed attempts are outlined in red.

## 6. Safety in one sentence each (30 s)

Never mutates the repo (per-attempt clones) · argv-only subprocesses with timeouts · paths
outside the repo and `.env`/`.git`/CI secrets refused · PR only on confirmation, only on
`codefix/<task_id>` · Fixer may not edit tests.

## Useful commands

```bash
codefix demo py08-mutable-default            # hard bug: mutable default argument
codefix demo js03-template-falsy --arch p2p  # JS repo, node:test
codefix bench --arch all --runs 1            # 48 logged runs + report
codefix runs --outcome escalated
```
