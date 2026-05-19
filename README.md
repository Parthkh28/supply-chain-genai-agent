# Supply Chain GenAI Agent

End-to-end supply-chain disruption resolution system combining machine learning, RAG, and GenAI agent workflows.

## Overview

This project detects supply-chain disruptions using trained XGBoost models and resolves exceptions through a LangGraph-based GenAI workflow.

The system supports:

- Disruption detection using a trained XGBoost classifier
- Cost prediction using a trained XGBoost regressor
- RAG over a mitigation playbook using FAISS and HuggingFace embeddings
- Groq-hosted LLM reasoning for risk assessment and mitigation planning
- Structured Pydantic outputs
- Human-in-the-loop approval for high-risk or low-confidence decisions
- Audit trail and execution payload generation
- Replay mode for historical Kaggle rows
- Leakage-safe production inference design

## Architecture

Training and inference are separated:

- `notebooks/scm-model_v2_patched.ipynb` trains the ML models and writes artifacts.
- `supply_chain_genai_agent_groq_hf.py` loads trained artifacts and runs the GenAI/RAG/LangGraph workflow.

Expected artifact layout:

```text
supply_chain_agent_training_outputs/
  models/
    disruption_classifier.pkl
    cost_predictor.pkl
    feature_config.json
  knowledge_base/
    mitigation_playbook.txt
    faiss_index/
      index.faiss
      index.pkl
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Parthkh28/supply-chain-genai-agent.git
   cd supply-chain-genai-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Copy `.env.example` to `.env` and configure your settings:
   ```bash
   cp .env.example .env
   ```
   Ensure you add your `GROQ_API_KEY`.

## Usage

To run the agent workflow:

```bash
python supply_chain_genai_agent_groq_hf.py
```