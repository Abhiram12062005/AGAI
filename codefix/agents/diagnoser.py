"""Diagnoser — explicit ReAct (Thought -> Action -> Observation) over bug report, trace and repo.

Each step the model returns a structured {thought, action, action_input, final_answer}. We
execute the action through the MCP tools (bound as LangChain tools) and feed the observation
back. The full transcript is persisted verbatim to the trace (event `react_step`) and to
traces/<task_id>.react.md — it is a deliverable, not debug output.

Iterations are capped (pipeline.react_max_iterations); at the cap the model is forced to
answer, and if it still cannot, a best-effort diagnosis is assembled from the evidence
gathered so far (`degraded=True`).
"""

from __future__ import annotations

import json
from pathlib import Path

from codefix import config
from codefix.agents.base import ReActStep, RunContext, chunk_ids, clip, format_chunks
from codefix.blackboard import Diagnosis
from mcp_server.sandbox import norm_rel

TOOLS = ["code_search", "read_file", "list_tree", "parse_stacktrace", "rag_retrieve", "static_analysis"]

SYSTEM = """You are the Diagnoser in CodeFix AI, a multi-agent bug-resolution system.
Your only job is to find the ROOT CAUSE of the reported bug in the repository. You do not write the fix.

Work in an explicit ReAct loop. Each turn reply with JSON:
  thought      - what you know so far and what you need next
  action       - one tool name from the list below, or "final_answer"
  action_input - the tool arguments as a JSON object encoded in a string (use "{{}}" for final_answer)
  final_answer - null, except when action == "final_answer"

After each action you receive an Observation. Be efficient: you have at most {max_iter} steps.
Read the actual code before concluding; cite evidence as file:line. Prefer the innermost repo
frame of any stack trace as a starting point, but verify — the crash site is often not the defect.

final_answer fields: root_cause, implicated_files (repo-relative paths that must change, not test files),
implicated_symbols, evidence, confidence (0-1), citations (ids such as code:src/x.py:10-40 or bugs:<id>
from the retrieved context/observations that you actually relied on).

Tools:
{tools}
"""


def _tool_docs(tools) -> str:
    docs = []
    for t in tools:
        schema = t.args_schema.model_json_schema() if t.args_schema else {}
        args = {k: v.get("type", v.get("anyOf", "")) for k, v in schema.get("properties", {}).items()}
        docs.append(f"- {t.name}{json.dumps(args)}: {t.description.strip().splitlines()[0]}")
    return "\n".join(docs)


def _seed_context(ctx: RunContext, k: int) -> tuple[str, set[str]]:
    query = f"{ctx.bug_report}\n{ctx.stack_trace}"[:2000]
    parts, ids = [], set()
    trace_text = ctx.stack_trace or (ctx.bug_report if "Traceback" in ctx.bug_report or " at " in ctx.bug_report else "")
    if trace_text:
        st = ctx.tool("diagnoser", "parse_stacktrace", text=trace_text)
        parts.append("Parsed stack trace:\n" + clip(st, 2500))
    for coll, n in (("code", k), ("bugs", 2), ("docs", 2)):
        try:
            chunks = ctx.tool("diagnoser", "rag_retrieve", query=query, k=n, collection=coll)["chunks"]
        except Exception as e:  # retrieval is an aid, never a hard dependency
            chunks = []
            ctx.event("rag_error", agent="diagnoser", error=str(e))
        ids |= {c["id"] for c in chunks}
        ctx.event("rag_retrieval", agent="diagnoser", collection=coll, ids=[c["id"] for c in chunks])
        parts.append(f"Retrieved {coll} chunks:\n" + format_chunks(chunks, 900))
    return "\n\n".join(parts), ids


def run(ctx: RunContext) -> tuple[Diagnosis, list[dict]]:
    cfg = ctx.settings.pipeline
    max_iter = cfg.react_max_iterations
    tools = {t.name: t for t in ctx.toolbox.langchain_tools(TOOLS, task_id=ctx.task_id, agent="diagnoser")}
    system = SYSTEM.format(max_iter=max_iter, tools=_tool_docs(tools.values()))
    seed, known_ids = _seed_context(ctx, ctx.settings.pipeline.rag_k)
    messages: list[dict] = [{"role": "user", "content": (
        f"BUG REPORT:\n{ctx.bug_report}\n\n"
        + (f"STACK TRACE:\n{ctx.stack_trace}\n\n" if ctx.stack_trace else "")
        + f"INITIAL CONTEXT (retrieval-augmented):\n{seed}\n\nBegin. Step 1 of {max_iter}.")}]
    transcript: list[dict] = []
    files_seen: list[str] = []
    final = None

    for step in range(1, max_iter + 2):  # +1 = forced-answer turn after the cap
        forced = step > max_iter
        if forced:
            messages.append({"role": "user", "content": "Step limit reached. You MUST reply now with "
                             "action='final_answer' and your best-effort final_answer."})
        res = ctx.llm.structured(role="diagnoser", system=system, messages=messages, schema=ReActStep,
                                 task_id=ctx.task_id, agent="diagnoser")
        st: ReActStep = res.parsed
        entry = {"step": step, "thought": st.thought, "action": st.action, "action_input": st.action_input,
                 "thinking_summary": res.thinking, "observation": None, "forced": forced}
        messages.append({"role": "assistant", "content": res.assistant_content})
        if st.action == "final_answer" or st.final_answer is not None:
            final = st.final_answer
            entry["observation"] = "(final answer)"
            transcript.append(entry)
            ctx.event("react_step", agent="diagnoser", **entry)
            if final is not None:
                break
            messages.append({"role": "user", "content": "final_answer was null. Provide it."})
            continue
        if forced:
            transcript.append(entry)
            ctx.event("react_step", agent="diagnoser", **entry)
            break
        try:
            args = json.loads(st.action_input or "{}")
            if not isinstance(args, dict):
                raise ValueError("action_input must be a JSON object")
            if st.action not in tools:
                raise ValueError(f"unknown tool '{st.action}'. Available: {', '.join(tools)}")
            obs = tools[st.action].invoke(args)
            known_ids |= chunk_ids(obs)
            if st.action == "read_file" and isinstance(obs, dict):
                files_seen.append(obs.get("path", ""))
            obs_text = clip(obs)
        except Exception as e:
            obs_text = f"ERROR: {e}"
        entry["observation"] = obs_text
        transcript.append(entry)
        ctx.event("react_step", agent="diagnoser", **entry)
        messages.append({"role": "user", "content": f"Observation:\n{obs_text}\n\nStep {step + 1} of {max_iter}."})

    degraded = final is None or transcript[-1]["forced"]
    if final is None:  # graceful degradation: best effort from what was gathered
        last = transcript[-1]["thought"] if transcript else "no reasoning produced"
        diag = Diagnosis(root_cause=f"(best-effort, iteration cap reached) {last}",
                         implicated_files=list(dict.fromkeys(f for f in files_seen if f)),
                         evidence=[t["thought"][:300] for t in transcript[-3:]], confidence=0.2,
                         degraded=True, react_steps=len(transcript))
    else:
        cited = [c for c in final.citations if c in known_ids]
        dropped = [c for c in final.citations if c not in known_ids]
        if dropped:
            ctx.event("citation_dropped", agent="diagnoser", ids=dropped)
        diag = Diagnosis(root_cause=final.root_cause,
                         implicated_files=[norm_rel(f) for f in final.implicated_files],
                         implicated_symbols=final.implicated_symbols, evidence=final.evidence,
                         confidence=max(0.0, min(1.0, final.confidence * (0.7 if degraded else 1.0))),
                         citations=cited, degraded=degraded, react_steps=len(transcript))
    _write_markdown(ctx.task_id, ctx.bug_report, transcript, diag)
    return diag, transcript


def _write_markdown(task_id: str, bug: str, transcript: list[dict], diag: Diagnosis) -> None:
    lines = [f"# ReAct transcript — {task_id}\n", f"**Bug report:** {bug}\n"]
    for t in transcript:
        lines.append(f"## Step {t['step']}{' (forced)' if t['forced'] else ''}\n")
        if t.get("thinking_summary"):
            lines.append(f"*Thinking (summary):* {t['thinking_summary']}\n")
        lines.append(f"**Thought:** {t['thought']}\n\n**Action:** `{t['action']}` `{t['action_input']}`\n")
        lines.append(f"**Observation:**\n```\n{t['observation']}\n```\n")
    lines.append(f"## Diagnosis\n```json\n{diag.model_dump_json(indent=2)}\n```\n")
    p = Path(config.TRACES_DIR) / f"{task_id}.react.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:  # append: re-diagnoses add a second transcript
        f.write("\n".join(lines))
