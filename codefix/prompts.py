"""Prompt variants: the course asks for runs that differ by prompt *architecture* (ReAct vs
few-shot) and by *persona/content*. A run picks one of each (CLI `--variant/--persona`, the
benchmark's `--variants/--personas`, or `pipeline.prompt_variant/persona` in models.yaml) and
the choice is stored in the run's meta, so every metric can be split by it.

  variant   react          Diagnoser runs an explicit ReAct tool loop (default).
            fewshot        Diagnoser answers in one shot from retrieved context + worked examples,
                           no tool loop.
            react-reflect  ReAct diagnosis, and the Fixer's patch is critiqued by a Reviewer agent
                           (Reflection pattern) before it goes to the Tester.
  persona   a short role preamble prepended to the Diagnoser and Fixer system prompts.
"""

from __future__ import annotations

VARIANTS = ("react", "fewshot", "react-reflect")

PERSONAS: dict[str, str] = {
    "default": "",
    "senior": ("You are a senior software engineer with 15 years of production debugging experience. "
               "You distrust the first plausible explanation, check edge cases, and prefer the smallest "
               "change that addresses the root cause."),
    "novice": ("You are a junior developer in your first year. Work carefully and step by step, explain "
               "your reasoning in simple terms, and do not assume anything you have not read in the code."),
    "security": ("You are an application-security reviewer. Besides fixing the defect, make sure the change "
                 "does not introduce unsafe behaviour (injection, unchecked input, file or network access)."),
}


def check(variant: str, persona: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant '{variant}' (choose from {', '.join(VARIANTS)})")
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona '{persona}' (choose from {', '.join(PERSONAS)})")


def with_persona(system: str, persona: str) -> str:
    pre = PERSONAS.get(persona or "default", "")
    return f"{pre}\n\n{system}" if pre else system


FEWSHOT_SYSTEM = """You are the Diagnoser in CodeFix AI, a multi-agent bug-resolution system.
Your only job is to find the ROOT CAUSE of the reported bug. You do not write the fix.

You get the bug report, a parsed stack trace if any, retrieved code chunks and the full text of the
most relevant files. Answer in ONE step (there is no tool loop) as JSON with: root_cause,
implicated_files (repo-relative source paths that must change, never test files),
implicated_symbols, evidence (file:line + what you saw), confidence (0-1), citations (ids of the
retrieved chunks or file:<path> excerpts you relied on).

Worked examples:

Example 1
BUG: "cart total is too low when the cart has more than one item"
CONTEXT: [code:shop/cart.py:10-18] def subtotal(items): total = 0; for i in range(len(items) - 1): total += items[i].price
ANSWER: {"root_cause": "subtotal() iterates range(len(items) - 1), so the last item is never added.",
 "implicated_files": ["shop/cart.py"], "implicated_symbols": ["subtotal"],
 "evidence": ["shop/cart.py:12 loop stops one element early"], "confidence": 0.9,
 "citations": ["code:shop/cart.py:10-18"]}

Example 2
BUG: "slugify('Hello World Again') returns 'hello-world again'"
CONTEXT: [code:src/text.js:3-6] return s.toLowerCase().replace(' ', '-')
ANSWER: {"root_cause": "String.replace with a string pattern only replaces the first space; it needs a
 global regex.", "implicated_files": ["src/text.js"], "implicated_symbols": ["slugify"],
 "evidence": ["src/text.js:4 replace(' ', '-') without /g"], "confidence": 0.85,
 "citations": ["code:src/text.js:3-6"]}

Now diagnose the real bug the same way. Read the provided file text before concluding; the crash site is
often not the defect.
"""

REVIEWER_SYSTEM = """You are the Reviewer in CodeFix AI (Reflection pattern). A Fixer agent proposed
a patch for a diagnosed bug. Critique it before it is tested:
- Does it fix the ROOT CAUSE described in the diagnosis, not just the symptom?
- Is it minimal, and does it avoid editing tests?
- Could it break other behaviour or introduce unsafe code?
Reply as JSON {approved, issues, revised}. If the patch is right, approved=true, issues=[], revised=null.
Otherwise approved=false, list the issues, and put a corrected Fixer output in `revised`
(same shape: rationale, edits[{path, find, replace}], citations, confidence) where each `find` is copied
exactly from the CURRENT files.
"""
