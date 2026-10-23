# CodeFix AI — web app + ADK orchestration + CrewAI A2A workers + MCP tool server, one container.
# Built by Cloud Build for Cloud Run (`deploy/deploy_cloudrun.ps1`), or locally:
#   docker build -t codefix-ai . && docker run -p 8080:8080 -e GEMINI_API_KEY=... codefix-ai
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git nodejs ripgrep \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system user.name "CodeFix AI" \
 && git config --system user.email "codefix-ai@users.noreply.github.com" \
 && git config --system init.defaultBranch main

WORKDIR /app
# CPU-only torch keeps the image ~1 GB smaller (sentence-transformers needs torch, not CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY codefix ./codefix
COPY mcp_server ./mcp_server
COPY web ./web
RUN pip install --no-cache-dir -e .

# bake the RAG embedding model into the image: no download on every cold start
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

COPY config ./config
COPY benchmarks ./benchmarks
COPY docs ./docs

# Cloud Run's filesystem is writable only under /tmp (and is per-instance, not persistent):
# runs.db, traces and sandboxes live there.
ENV CODEFIX_DATA_DIR=/tmp/codefix \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn web.api.app:app --host 0.0.0.0 --port ${PORT}"]
