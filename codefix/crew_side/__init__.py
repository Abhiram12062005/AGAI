"""CrewAI side of CodeFix AI: Fixer (+ Reviewer), Tester and PR agents, served over A2A."""

import os

# No anonymous telemetry or cloud trace upload from CrewAI: run data stays in runs.db / traces/.
for _k, _v in (("CREWAI_DISABLE_TELEMETRY", "true"), ("CREWAI_TRACING_ENABLED", "false"),
               ("CREWAI_DISABLE_TRACKING", "true")):
    os.environ.setdefault(_k, _v)
