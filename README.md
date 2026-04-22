# PoBot - HK Labor Law Chatbot

## Quick Start

### 1. Setup Environment
```bash
cd /mnt/c/Work/Migrasia/PoBot
source PoBot_env/bin/activate  # Windows: PoBot_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Preprocessing (if needed)
```bash
cd notebooks
python preprocess.py
```

### 3. Build Vectorstore (if needed)
```bash
python embedding_setup.py
```

### 4. Launch Chatbot UI ⭐
```bash
python chatbot_ui.py
```
Then open **http://localhost:7860** in your browser.

---

## Alternative: Command Line Mode
```bash
python rag_pipeline.py
```
Then type questions directly in the terminal.

---

## Run Evaluation

### Old Keyword-Based Evaluation
```bash
python evaluate.py
```

### New DeepEval (Recommended)
```bash
python evaluate_using_deepeval.py
```

Generates detailed JSON report with 5 metrics:
- Faithfulness (hallucination detection)
- Answer Relevancy
- Context Precision
- Context Recall
- Answer Correctness

---

## Project Structure

```
PoBot/
├── notebooks/
│   ├── preprocess.py          # Document loading & chunking
│   ├── embedding_setup.py     # Create FAISS vectorstore
│   ├── rag_pipeline.py        # Main RAG logic
│   ├── evaluate.py            # Old keyword-based eval
│   ├── evaluate_using_deepeval.py  # New LLM-based eval
│   ├── chatbot_ui.py          # Gradio web interface ⭐
│   └── data/
│       ├── raw/               # Source PDFs
│       └── processed/         # Chunks & JSON
├── vectorstore/
│   └── faiss_index/           # Embeddings
├── requirements.txt
└── README.md
```

---

## Recent Fixes (2026-04-22)

### 1. Fixed RAG Pipeline Prompt
- Changed overly conservative NULL response rule
- Now returns partial information instead of "don't know"
- Fixes minimum wage question issue

### 2. Fixed Context Precision Metric
- Now uses source_documents metadata instead of text search
- Was returning 0.0 for all questions (bug)
- Now correctly measures retrieval ranking quality

### 3. Added Chatbot UI
- Simple Gradio web interface
- Sample questions for quick testing
- Shows sources for each answer
- Clean, modern design

---

## Expected Performance

After fixes, target metrics:
- **Overall Score:** 0.75+ (was 0.63)
- **Faithfulness:** 0.85+ (low hallucination)
- **Answer Correctness:** 0.75+ (was 0.62)
- **Source Recall:** 75%+ (good retrieval)

Current weak spot: **wage** category questions (0% correctness)
- Requires investigation of Minimum Wage PDF chunking

---

## API Configuration

DeepSeek API is configured in:
- `rag_pipeline.py` line 11: `DEEPSEEK_API_KEY`
- `evaluate_using_deepeval.py` line 13: `DEEPSEEK_API_KEY`

Update if your key changes.

---

## Troubleshooting

### "Vectorstore not found"
→ Run `embedding_setup.py` first

### "API key error"
→ Check DeepSeek API key in rag_pipeline.py

### "Gradio not installed"
→ Run `pip install gradio`

### Low evaluation scores
→ Check chunk sizes in preprocess.py (try 1000-1500 chars)
→ Verify source documents contain expected information
