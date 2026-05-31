# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# System deps required by faiss + sentence-transformers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Pre-download the embedding model so containers start with a warm cache.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# After warm-up, switch the runtime to offline mode so HF Hub network checks
# don't slow down inference or fail in air-gapped deployments.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY supply_chain_genai_agent_groq_hf.py api.py constants.py ./
COPY supply_chain_agent_training_outputs ./supply_chain_agent_training_outputs
# Copy data/ so --replay-mode works inside the container.
COPY data ./data

# Best practice: build the FAISS index at image build time so cold starts are fast.
# Comment this out if the index will be mounted from a volume instead.
RUN python supply_chain_genai_agent_groq_hf.py index || \
    echo "FAISS index build skipped (artifacts may be mounted at runtime)."

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
