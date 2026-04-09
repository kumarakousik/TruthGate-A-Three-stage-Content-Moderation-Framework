# TruthGate

**TruthGate** is a three-stage content moderation framework for detecting toxic language and verifying factual claims in social media posts, with a focus on the Indian digital context.

---

## Pipeline Overview

```
Social Media Post
        │
        ▼
┌─────────────────────────────────┐
│        Text Preprocessing       │
│  URL / hashtag / mention        │
│  extraction · NER (spaCy +      │
│  EntityRuler) · emoji           │
│  normalisation · noise removal  │
└────────────────┬────────────────┘
                 │
        ┌────────▼────────┐
        │    Stage 1      │
        │ Toxicity        │
        │ Detection       │
        │ RoBERTa + CNN   │
        │ + MLP           │
        │ → RED / ORANGE  │
        │   / GREEN       │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │    Stage 2      │
        │ Domain          │
        │ Classification  │
        │ DeBERTa + MLP   │
        │ → political /   │
        │   movies /      │
        │   other         │
        └────────┬────────┘
                 │
        ┌────────▼────────────────────────────────┐
        │           Stage 3 — Agentic RAG          │
        │                                          │
        │  Node 1: Stage 1 + Stage 2 outputs       │
        │          fed into LLM (Gemma3:4b)        │
        │          → claim extraction              │
        │          → search query generation       │
        │          → domain routing                │
        │                   │                      │
        │  Node 2: Parallel domain agents          │
        │          Political → NewsAPI → SerpAPI   │
        │          Movies   → TMDB → NewsAPI       │
        │                        → SerpAPI         │
        │          Other    → LLM intrinsic only   │
        │          Retrieved docs → ChromaDB       │
        │                   │                      │
        │  Node 3: Verdict generation              │
        │          Post vs retrieved context       │
        │          → TRUE / FALSE /                │
        │             MISLEADING / UNVERIFIABLE    │
        │          → discrepancy score (0.0–1.0)   │
        └──────────────────────────────────────────┘
                 │
        ┌────────▼──────────────────┐
        │   Moderation report +     │
        │   discrepancy score       │
        └───────────────────────────┘
```

---

## Stages

### Text Preprocessing

Every post passes through a preprocessing pipeline built with spaCy (`en_core_web_lg`) and a custom `EntityRuler`.

- URLs, hashtags, and user mentions are extracted and stored separately — used as features in Stage 1 and retrieval cues in Stage 3
- Named Entity Recognition (NER) runs on properly-cased text to correctly identify Indian political figures and celebrities
- Emojis are converted to their textual equivalents
- Punctuation, special characters, and excessive whitespace are removed
- Stopword removal and lemmatization are intentionally skipped to preserve semantic structure for downstream transformer models and the RAG pipeline

---

### Stage 1 — Toxicity Detection

The cleaned text, along with binary features derived from user mentions, extracted names, and organisation entities, is fed to the toxicity model.

**Architecture:** `unitary/unbiased-toxic-roberta` + 3 parallel CNN layers (kernel sizes 2, 3, 4) + 3 MLP layers

**Output:** RED / ORANGE / GREEN flag + confidence score + toxic class label

| Metric | Score |
|---|---|
| Precision | 84.63% |
| Recall | 87.14% |
| F1-score | 85.56% |
| Accuracy | 95.29% |
| AUC | 97.56% |

---

### Stage 2 — Domain Classification

The post is classified into one of three domains to route the correct retrieval agents in Stage 3.

**Architecture:** `microsoft/deberta-base` + 3-layer MLP

**Output:** `political` / `movies` / `other` + confidence score

| Metric | Score |
|---|---|
| Precision | 94.83% |
| Recall | 94.57% |
| F1-score | 94.70% |
| Accuracy | 96.47% |
| AUC | 99.39% |

---

### Stage 3 — Agentic RAG Verification

A three-node agentic pipeline built with LangChain (LangGraph migration planned). State is passed via a shared dictionary designed for forward LangGraph compatibility.

#### Node 1 — Claim extraction and query generation

Receives the toxicity class, domain class, confidence scores, and full post context (cleaned text, hashtags, mentions, URLs) from Stages 1 and 2.

A local LLM (`gemma3:4b` via Ollama, with Gemini API as fallback) then:
- Extracts factual claims from the post
- Generates targeted search queries per claim
- Determines the dominant domain for agent routing
- Stores embeddings to ChromaDB (`mxbai-embed-large` via OllamaEmbeddings)

#### Node 2 — Parallel domain agents

Retrieval agents run in parallel via `ThreadPoolExecutor`. On each run, agents first query ChromaDB before hitting external APIs.

| Domain | Primary | Fallback |
|---|---|---|
| Political | NewsAPI | SerpAPI |
| Movies | TMDB → NewsAPI | SerpAPI |
| Other | — (LLM intrinsic knowledge only, RAG skipped) | — |

Retrieved documents are embedded using `mxbai-embed-large` and stored in ChromaDB at `../TruthGate_db/vector_store` (collection: `TruthGate`).

#### Node 3 — Verdict generation

The LLM compares the original post against all retrieved context and outputs:
- **Verdict:** `TRUE` / `FALSE` / `MISLEADING` / `UNVERIFIABLE`
- **Discrepancy score:** 0.0–1.0
- **Specific differences:** list of factual discrepancies between the post and retrieved sources

Results are stored back to ChromaDB for future lookups.

---

## Tech Stack

| Component | Tool |
|---|---|
| Toxicity model | `unitary/unbiased-toxic-roberta` (HuggingFace) |
| Domain model | `microsoft/deberta-base` (HuggingFace) |
| Local LLM | Ollama — `gemma3:4b` |
| LLM fallback | Gemini API |
| Embeddings | `mxbai-embed-large` via OllamaEmbeddings |
| Vector store | ChromaDB (persistent, local) |
| Orchestration | LangChain (LangGraph migration planned) |
| NLP pipeline | spaCy `en_core_web_lg` + EntityRuler |
| Retrieval APIs | NewsAPI · SerpAPI · TMDB · Google Fact Check API |
| Deep learning | PyTorch + HuggingFace Transformers |

---

## Dataset Sources

### Toxicity Detection (Stage 1)

Combined from two Kaggle competition datasets:
- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data)

Total: 357,334 samples (balanced: 178,667 non-toxic / 178,667 across six toxic classes)

### Domain Classification (Stage 2)

- [IMDB Movie Reviews](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews) — movie domain
- [Twitter Data – Indian General Election 2019](https://www.kaggle.com/datasets/yogesh239/twitter-data-about-2019-indian-general-election) — political domain
- [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) — other domain

---

## Project Structure

```
TruthGate/
├── stage1_toxicity/          # RoBERTa + CNN + MLP toxicity model
├── stage2_domain/            # DeBERTa + MLP domain classifier
├── stage3_rag/
│   ├── node1_query_gen.py    # Claim extraction and query generation
│   ├── node2_agents.py       # Parallel domain retrieval agents
│   └── node3_verdict.py      # Verdict generation
├── preprocessing/            # spaCy pipeline + EntityRuler patterns
└── TruthGate_db/
    └── vector_store/         # ChromaDB persistent store
```

---

## Authors

- **Bhavirisetty Kumara Kousik** — Department of Mathematics, SAS, VIT-AP University
- **A. Manimaran** — Department of Mathematics, SAS, VIT-AP University
