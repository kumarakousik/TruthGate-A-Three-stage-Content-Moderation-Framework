import json
from datetime import datetime, timezone

import pandas as pd
import torch

from AgenticRAGState import AgenticRAGState
from Domain_model import domain_model
from Preprocessing import extract_features_batch
from Toxic_model import toxic_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
toxic_model.to(device)
domain_model.to(device)


def node1_toxic_and_domain_classifier(state: AgenticRAGState) -> AgenticRAGState:


    text: str = state.get("original", "").strip()

    # ── ReAct: Thought ─────────────────────────────────────────────────────
    state.setdefault("thoughts", []).append(
        "Received raw post. Starting preprocessing, toxic classification, and domain classification."
    )

    if not text:
        # ── ReAct: Observation ─────────────────────────────────────────────
        state.setdefault("observations", []).append(
            "Empty input received. Skipping all classification steps."
        )
        state.setdefault("actions", []).append({
            "action"    : "Node 1 skipped — empty input",
            "timestamp" : datetime.now(timezone.utc).isoformat(),
        })
        return state

    try:
        # ── Step 1: Preprocessing ──────────────────────────────────────────
        texts    = pd.DataFrame({"text/post": [text]})
        features = extract_features_batch(texts=texts["text/post"])
        feat     = features[0]

        clean_text = feat["cleaned_text"]
        has_person = feat["has_person"]
        has_org    = feat["has_org"]

        state.setdefault("actions", []).append({
            "action"    : "Preprocessing completed via extract_features_batch",
            "timestamp" : datetime.now(timezone.utc).isoformat(),
        })

        # ── Step 2: Toxic Prediction ───────────────────────────────────────
        encoding                  = toxic_model.preprocess_text(clean_text)
        toxic_preds, toxic_probs  = toxic_model.predict_single(
            encoding   = encoding,
            device     = device,
            has_person = has_person,
            has_org    = has_org,
        )
        toxic_probs = toxic_probs.detach().cpu().numpy().flatten()
        toxic_scores = {
            label: round(float(prob), 2)
            for label, prob in zip(toxic_model.LABELS, toxic_probs)
        }

        state["actions"].append({
            "action"    : f"Toxic model predicted: {toxic_preds}",
            "timestamp" : datetime.now(timezone.utc).isoformat(),
        })

        # ── Step 3: Domain Prediction ──────────────────────────────────────
        domain_input_text            = domain_model.build_input_text(feat)
        domain_encoding              = domain_model.preprocess_text(domain_input_text)
        domain_preds, domain_probs   = domain_model.predict_single(
            encoding  = domain_encoding,
            device    = device,
            threshold = 0.5,
        )
        domain_probs = domain_probs.detach().cpu().numpy().flatten()
        domain_scores = {
            label: round(float(prob), 2)
            for label, prob in zip(domain_model.LABELS, domain_probs)
        }

        state["actions"].append({
            "action"    : f"Domain model predicted: {domain_preds}",
            "timestamp" : datetime.now(timezone.utc).isoformat(),
        })

        # ── Step 4: Write all outputs back to state ────────────────────────
        state["cleaned_text"]  = clean_text
        state["urls"]          = feat.get("urls", [])
        state["hashtags"]      = feat.get("hashtags", [])
        state["mentions"]      = feat.get("mentions", [])
        state["persons"]       = feat.get("persons", [])
        state["organizations"] = feat.get("organizations", [])
        state["has_person"]    = has_person
        state["has_org"]       = has_org
        state["toxic_scores"]  = toxic_scores
        state["domain_scores"] = domain_scores

        # ── ReAct: Observation ─────────────────────────────────────────────
        state.setdefault("observations", []).append(
            f"Toxic scores: {json.dumps(toxic_scores)} | "
            f"Domain scores: {json.dumps(domain_scores)}"
        )

    finally:
        # ── GPU Cleanup ────────────────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    return state