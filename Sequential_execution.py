from datetime import datetime, timezone
from types import SimpleNamespace
import Embedding
import Vector_store
from AgenticRAGState import AgenticRAGState
from Agents import run_politics_agent, run_movie_agent, run_general_agent

# ── Helper: Vector Search ──────────────────────────────────────────────────────
def _vector_search(query: str, embedding_model, vector_db, top_k: int = 3) -> list:
    """
    Embed the query and search ChromaDB.
    Returns list of matching text strings, or [] if nothing found.
    """
    try:
        prefixed   = [f"Represent this sentence for searching relevant passages: {query}"]
        embeddings = embedding_model.generate_text_embeddings(prefixed)  # np.ndarray
        results    = vector_db.search(embeddings[0], top_k=top_k)        # returns list of docs
        return [r.page_content for r in results if r.page_content.strip()]
    except Exception as e:
        return []


# ── Helper: Store to ChromaDB ──────────────────────────────────────────────────
def _store_to_chromadb(
    texts: list,
    labels: list,
    metadata_base: dict,
    embedding_model,
    vector_db,
    node_label: str = "node3",
) -> None:
    """Embed and store a list of texts into ChromaDB."""
    if not texts:
        return
    try:
        prefixed   = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        embeddings = embedding_model.generate_text_embeddings(prefixed)
        docs       = [
            SimpleNamespace(
                page_content=text,
                metadata={**metadata_base, "source": node_label, "label": label},
            )
            for text, label in zip(texts, labels)
        ]
        vector_db.add_document(documents=docs, embeddings=embeddings)
    except Exception as e:
        print(f"ChromaDB store failed ({node_label}): {e}")


# ── Node 3 ─────────────────────────────────────────────────────────────────────
def node3_retrieval(state: AgenticRAGState,embedding_model = Embedding.Embedding_Manager,vector_db = Vector_store.vector_db) -> AgenticRAGState:
    """
    Node 3 — Vector Search → Sequential Domain Agent Fallback

    Flow per query:
        1. Search ChromaDB first
           ├── Hit  → use cached context, skip API call
           └── Miss → run domain agent sequentially
                      ├── domain_politics : NewsAPI → SerpAPI fallback
                      ├── domain_movies   : TMDB → NewsAPI → SerpAPI fallback
                      └── other / None    : NewsAPI → SerpAPI fallback

    Reads from state:
        search_queries, primary_domain, claim

    Writes to state:
        vector_search_results, agent_results,
        retrieved_context, retrieval_count,
        active_agents, failed_agents, queries_used,
        node3_status
        + ReAct: thoughts, actions, observations
    """

    queries        = state.get("search_queries", [])
    primary_domain = state.get("primary_domain", "other_domain")
    claim          = state.get("claim", "")
    current_time   = datetime.now(timezone.utc).isoformat()

    # ── Pick 2-3 queries to run (at least half of available) ──────────────
    queries_to_run = queries[: max(2, len(queries) // 2 + 1)]

    # ── Domain → Agent mapping ─────────────────────────────────────────────
    agent_fn = {
        "domain_politics": run_politics_agent,  # NewsAPI → SerpAPI fallback
        "domain_movies"  : run_movie_agent,     # TMDB → NewsAPI → SerpAPI fallback
    }.get(primary_domain, run_general_agent)    # fallback for unknown/None domain

    # ── Per-query state collectors ─────────────────────────────────────────
    vector_search_results = []   # raw ChromaDB hits per query
    agent_results         = []   # agent outputs for queries that missed cache
    context_parts         = []   # final merged readable context
    active_agents         = []
    failed_agents         = []
    queries_used          = []

    # ── Metadata base for ChromaDB stores ─────────────────────────────────
    meta_base = {
        "primary_domain": primary_domain,
        "timestamp"     : current_time,
        "claim"         : claim[:200],
    }

    # ══════════════════════════════════════════════════════════════════════
    # Sequential loop over selected queries
    # ══════════════════════════════════════════════════════════════════════
    for query in queries_to_run:
        if not query:
            continue

        queries_used.append(query)

        # ── Step 1: Vector Search ──────────────────────────────────────────
        state.setdefault("thoughts", []).append(
            f"Searching ChromaDB for cached results on query: '{query}'"
        )
        state.setdefault("actions", []).append({
            "action"   : f"vector_search: {query}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        hits = _vector_search(query, embedding_model, vector_db)

        if hits:
            # ── Cache HIT → use vector results directly ────────────────────
            state.setdefault("observations", []).append(
                f"ChromaDB returned {len(hits)} cached results for query: '{query}'"
            )
            vector_search_results.extend(hits)
            for h in hits:
                context_parts.append(f"[vector_cache] {h}")

            # Store the query itself so future similar queries hit cache
            _store_to_chromadb(
                texts     = [query],
                labels    = [f"cache_query"],
                metadata_base = meta_base,
                embedding_model = embedding_model,
                vector_db = vector_db,
            )

        else:
            # ── Cache MISS → call domain agent sequentially ────────────────
            state["observations"].append(
                f"ChromaDB returned no results for query: '{query}'. Invoking agent."
            )
            state["thoughts"].append(
                f"No cache hit. Domain is '{primary_domain}'. Running agent for: '{query}'"
            )
            state["actions"].append({
                "action"   : f"{primary_domain}_agent call: {query}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            result = agent_fn(query, primary_domain)
            agent_results.append(result)

            if result["status"] == "success":
                active_agents.append(result["agent"])
                state["observations"].append(result["observation"])

                # Flatten result into context
                if isinstance(result["result"], list):
                    for item in result["result"]:
                        part = item.get("title", "") or item.get("overview", "")
                        desc = item.get("description", "") or item.get("release", "")
                        if part:
                            context_parts.append(
                                f"[{result['agent']}] {part}. {desc}".strip()
                            )
                elif isinstance(result["result"], str) and result["result"]:
                    context_parts.append(f"[{result['agent']}] {result['result']}")

                # ── Store agent result + observation to ChromaDB ───────────
                texts_to_store = []
                labels_to_store = []

                if result["observation"]:
                    texts_to_store.append(result["observation"])
                    labels_to_store.append(f"observation_{result['agent']}")
                if result["thought"]:
                    texts_to_store.append(result["thought"])
                    labels_to_store.append(f"thought_{result['agent']}")

                _store_to_chromadb(
                    texts         = texts_to_store,
                    labels        = labels_to_store,
                    metadata_base = meta_base,
                    embedding_model = embedding_model,
                    vector_db     = vector_db,
                )

            else:
                # Agent failed
                failed_agents.append(result["agent"])
                state["observations"].append(
                    f"Agent '{result['agent']}' failed for query: '{query}'"
                )
                state["actions"].append({
                    "action"   : f"Agent failed: {result['agent']} on query: '{query}'",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    # ── Merge all context ──────────────────────────────────────────────────
    retrieved_context = "\n\n".join(context_parts)

    # Store full merged context to ChromaDB for future reuse
    if retrieved_context:
        _store_to_chromadb(
            texts         = [retrieved_context],
            labels        = ["retrieved_context"],
            metadata_base = meta_base,
            embedding_model = embedding_model,
            vector_db     = vector_db,
        )

    # ── Final node3 status ─────────────────────────────────────────────────
    has_vector   = len(vector_search_results) > 0
    has_agent    = any(r["status"] == "success" for r in agent_results)
    node3_status = "success" if (has_vector or has_agent) else "all_failed"

    state["thoughts"].append(
        f"Node 3 complete. Vector hits: {len(vector_search_results)} | "
        f"Agent results: {len(agent_results)} | Status: {node3_status}"
    )

    # ── Write to state ─────────────────────────────────────────────────────
    state["vector_search_results"] = vector_search_results
    state["agent_results"]         = agent_results
    state["retrieved_context"]     = retrieved_context
    state["retrieval_count"]       = len(context_parts)
    state["active_agents"]         = active_agents
    state["failed_agents"]         = failed_agents
    state["queries_used"]          = queries_used
    state["node3_status"]          = node3_status

    return state