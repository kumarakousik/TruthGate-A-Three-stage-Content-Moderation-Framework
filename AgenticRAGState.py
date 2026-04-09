from typing import Dict, List, TypedDict


class AgenticRAGState(TypedDict):

    # ── Input ──────────────────────────────────────────────────────────────
    original: str

    # ── Node 1 Outputs: Preprocessing + Toxic & Domain Classification ──────
    cleaned_text:  str
    urls:          List[str]
    hashtags:      List[str]
    mentions:      List[str]
    persons:       List[str]
    organizations: List[str]
    has_person:    int
    has_org:       int
    toxic_scores:  Dict[str, float]
    domain_scores: Dict[str, float]

    # ── Node 2 Outputs: Claim Extraction + Query Generation ────────────────
    claim:          str
    search_queries: List[str]
    search_query:   str          # first query, convenience shortcut
    primary_domain: str

    # ── Node 3 Outputs: Vector Search + Sequential Agent Retrieval ─────────
    vector_search_results: List[str]   # raw ChromaDB hits (page_content strings)
    agent_results:         List[Dict]  # full agent output dicts per query
    retrieved_context:     str         # final merged context string → passed to Node 4
    active_agents:         List[str]   # agents that succeeded
    failed_agents:         List[str]   # agents that failed
    queries_used:          List[str]   # which queries actually ran
    retrieval_count:       int         # total context parts collected
    node3_status:          str         # "success" | "all_failed"

    # ── Node 4 Outputs: Misinformation Judgment ────────────────────────────
    verdict:             str           # "TRUE" | "FALSE" | "MISLEADING" | "UNVERIFIABLE"
    discrepancy_score:   float         # 0.0 (accurate) → 1.0 (completely false)
    differences:         List[str]     # key factual differences found
    summary:             str           # short verdict summary
    sources_used:        List[str]     # sources referenced in judgment
    toxicity_flag:       str           # "GREEN" | "ORANGE" | "RED"
    misleading_angle:    str           # what specifically is misleading people
    misinformation_type: str           # "exaggeration"|"fabrication"|"omission"|"framing"|"satire_misread"|"none"
    target_audience:     str           # who is being misled
    corrected_claim:     str           # what the truth actually is
    severity:            str           # "LOW" | "MEDIUM" | "HIGH"

    # ── ReAct Trace ────────────────────────────────────────────────────────
    thoughts:     List[str]            # what each node was thinking
    observations: List[str]            # what each node observed
    actions:      List[Dict]           # {"action": str, "timestamp": str}