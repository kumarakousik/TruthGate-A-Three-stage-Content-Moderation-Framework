import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import Embedding
import Ollama_LLM
import Vector_store
from AgenticRAGState import AgenticRAGState


def node4_verdict(
    state: AgenticRAGState,
    embedding_model=None,   # FIX: do NOT evaluate at import time
    vector_db=None,         # FIX: do NOT evaluate at import time
) -> AgenticRAGState:
    """
    Node 4 — Final Misinformation Judgment + ChromaDB Storage

    Reads from state:
        original, claim, retrieved_context,
        toxic_scores, primary_domain, agent_results

    Writes to state:
        verdict, discrepancy_score, differences, summary, sources_used,
        toxicity_flag, misleading_angle, misinformation_type,
        target_audience, corrected_claim, severity
        + ReAct: thoughts, actions, observations
    """

    # ── FIX: Resolve defaults at call time, not import time ───────────────
    if embedding_model is None:
        embedding_model = Embedding.Embedding_Manager
    if vector_db is None:
        vector_db = Vector_store.vector_db

    llm    = Ollama_LLM.ollama_llm
    parser = StrOutputParser()

    # ── Prompt — extended for misleading judgment ──────────────────────────
    template = """
You are a misinformation verdict agent tasked with identifying what is misleading people.

Compare the original post/claim against the retrieved real-world sources.

Original Post:
{original}

Extracted Claim:
{claim}

Retrieved Context from Sources:
{context}

Toxicity Flag: {flag}
Domain: {domain}

Your Tasks:
1. Identify key factual differences between the claim and the sources.
2. Rate the discrepancy: 0.0 (fully accurate) to 1.0 (completely false).
3. Give a final verdict.
4. Identify WHAT SPECIFICALLY is misleading people:
   - What part of the post is misleading (wording, framing, omission, exaggeration)?
   - What is the misleading narrative vs the actual truth?
   - Who is likely being misled and how?
5. Classify the type of misinformation.
6. Write the corrected claim in simple, clear words.
7. Rate the severity of the misinformation.

Return STRICT JSON only, no markdown, no explanation:
{{
  "verdict"            : "TRUE" | "FALSE" | "MISLEADING" | "UNVERIFIABLE",
  "discrepancy_score"  : 0.0,
  "differences"        : ["...", "..."],
  "summary"            : "...",
  "sources_used"       : ["...", "..."],
  "misleading_angle"   : "What specifically is misleading and why — wording, framing, omission, or exaggeration used to mislead",
  "misinformation_type": "exaggeration" | "fabrication" | "omission" | "framing" | "satire_misread" | "none",
  "target_audience"    : "Who this post is likely targeting or misleading",
  "corrected_claim"    : "What the truth actually is in simple, clear words",
  "severity"           : "LOW" | "MEDIUM" | "HIGH"
}}
"""

    prompt = PromptTemplate(
        input_variables=["original", "claim", "context", "flag", "domain"],
        template=template,
    )

    # ── Helper: safe JSON parse ────────────────────────────────────────────
    def safe_json_parse(text: str) -> dict:
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse verdict JSON:\n{text}")

    # ── Helper: store to ChromaDB ──────────────────────────────────────────
    def store_to_chromadb(texts: list, labels: list, meta: dict) -> None:
        if not texts:
            return
        try:
            prefixed   = [
                f"Represent this sentence for searching relevant passages: {t}"
                for t in texts
            ]
            embeddings = embedding_model.generate_text_embeddings(prefixed)
            docs       = [
                SimpleNamespace(
                    page_content=text,
                    metadata={**meta, "label": label},
                )
                for text, label in zip(texts, labels)
            ]
            vector_db.add_document(documents=docs, embeddings=embeddings)
            print(f"✅ Stored {len(docs)} node4 embeddings to ChromaDB")
        except Exception as e:
            print(f"⚠️  ChromaDB store failed (node4): {e}")

    # ── Step 1: Determine toxicity flag ───────────────────────────────────
    toxic = state.get("toxic_scores", {})
    if isinstance(toxic, str):
        try:
            toxic = json.loads(toxic)
        except Exception:
            toxic = {}

    if toxic.get("severe_toxic", 0) > 0.5 or toxic.get("threat", 0) > 0.5:
        flag = "RED"
    elif toxic.get("obscene", 0) > 0.5 or toxic.get("insult", 0) > 0.5:
        flag = "ORANGE"
    else:
        flag = "GREEN"

    # ── Step 2: ReAct trace ────────────────────────────────────────────────
    current_time = datetime.now(timezone.utc).isoformat()

    state.setdefault("thoughts", []).append(
        "Analyzing what is misleading people — comparing claim vs sources for final judgment."
    )
    state.setdefault("actions", []).append({
        "action"   : "LLM misinformation judgment",
        "timestamp": current_time,
    })

    # ── Step 3: Guard — no context retrieved ──────────────────────────────
    retrieved_context = state.get("retrieved_context", "").strip()
    if not retrieved_context:
        state.setdefault("observations", []).append(
            "No retrieved context available. Marking verdict as UNVERIFIABLE."
        )
        state.update({
            "verdict"            : "UNVERIFIABLE",
            "discrepancy_score"  : 1.0,
            "differences"        : [],
            "summary"            : "No sources were retrieved to verify the claim.",
            "sources_used"       : [],
            "toxicity_flag"      : flag,
            "misleading_angle"   : "Cannot determine — no sources available.",
            "misinformation_type": "none",
            "target_audience"    : "Unknown",
            "corrected_claim"    : "Could not verify.",
            "severity"           : "LOW",
        })
        return state

    # ── Step 4: LLM call ──────────────────────────────────────────────────
    formatted = prompt.format(
        original = state.get("original", ""),
        claim    = state.get("claim", ""),
        context  = retrieved_context,
        flag     = flag,
        domain   = state.get("primary_domain", "unknown"),
    )

    try:
        raw_response  = llm.invoke(formatted)
        response_text = parser.invoke(raw_response).strip()
        parsed        = safe_json_parse(response_text)
    except Exception as e:
        state.setdefault("observations", []).append(
            f"LLM verdict call failed: {e}. Defaulting to UNVERIFIABLE."
        )
        state.update({
            "verdict"            : "UNVERIFIABLE",
            "discrepancy_score"  : 1.0,
            "differences"        : [],
            "summary"            : f"Verdict generation failed: {e}",
            "sources_used"       : [],
            "toxicity_flag"      : flag,
            "misleading_angle"   : "Judgment failed — could not process.",
            "misinformation_type": "none",
            "target_audience"    : "Unknown",
            "corrected_claim"    : "Could not verify.",
            "severity"           : "LOW",
        })
        return state

    # ── Step 5: Extract parsed fields ─────────────────────────────────────
    verdict             = parsed.get("verdict",             "UNVERIFIABLE")
    discrepancy_score   = float(parsed.get("discrepancy_score", 1.0))
    differences         = parsed.get("differences",         [])
    summary             = parsed.get("summary",             "")
    sources_used        = parsed.get("sources_used",        [])
    misleading_angle    = parsed.get("misleading_angle",    "Not identified.")
    misinformation_type = parsed.get("misinformation_type", "none")
    target_audience     = parsed.get("target_audience",     "Unknown")
    corrected_claim     = parsed.get("corrected_claim",     "")
    severity            = parsed.get("severity",            "LOW")

    state.setdefault("observations", []).append(
        f"Verdict: {verdict} | Discrepancy: {discrepancy_score} | "
        f"Flag: {flag} | Severity: {severity} | Type: {misinformation_type}"
    )

    # ── Step 6: Store to ChromaDB ──────────────────────────────────────────
    texts_to_store  = []
    labels_to_store = []

    if summary:
        texts_to_store.append(summary)
        labels_to_store.append("verdict_summary")

    if misleading_angle:
        texts_to_store.append(misleading_angle)
        labels_to_store.append("misleading_angle")

    if corrected_claim:
        texts_to_store.append(corrected_claim)
        labels_to_store.append("corrected_claim")

    for i, diff in enumerate(differences):
        if diff:
            texts_to_store.append(diff)
            labels_to_store.append(f"difference_{i}")

    for r in state.get("agent_results", []):
        obs = r.get("observation", "")
        if obs:
            texts_to_store.append(obs)
            labels_to_store.append(f"agent_result_{r.get('agent', 'unknown')}")

    store_to_chromadb(
        texts  = texts_to_store,
        labels = labels_to_store,
        meta   = {
            "source"             : "node4",
            "verdict"            : verdict,
            "discrepancy_score"  : discrepancy_score,
            "toxicity_flag"      : flag,
            "severity"           : severity,
            "misinformation_type": misinformation_type,
            "primary_domain"     : state.get("primary_domain", "unknown"),
            "timestamp"          : current_time,
            "claim"              : state.get("claim", "")[:200],
        },
    )

    # ── Step 7: Final state update ─────────────────────────────────────────
    state.update({
        "verdict"            : verdict,
        "discrepancy_score"  : discrepancy_score,
        "differences"        : differences,
        "summary"            : summary,
        "sources_used"       : sources_used,
        "toxicity_flag"      : flag,
        "misleading_angle"   : misleading_angle,
        "misinformation_type": misinformation_type,
        "target_audience"    : target_audience,
        "corrected_claim"    : corrected_claim,
        "severity"           : severity,
    })

    return state