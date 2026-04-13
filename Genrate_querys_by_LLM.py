import json
import ast
import re
import ast
from datetime import datetime
import Ollama_LLM
from AgenticRAGState import AgenticRAGState
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def node2_generate_query(state: AgenticRAGState, Ollama_LLM = Ollama_LLM.ollama_llm) -> AgenticRAGState:
    llm = Ollama_LLM

    template = """
You are an intelligent misinformation detection agent.

From the given input JSON:
1. Extract the MAIN factual claim from the post.
2. Generate 3-4 precise web search queries upto minmum of 10 to 15 words to verify that claim.

Guidelines:
- Use hashtags + cleaned_text as main signal.
- Use domain_scores to guide context (politics, movies, etc.)
- If toxic tone exists, neutralize wording in query.
- Keep queries factual and verification-focused.

Return STRICT JSON only, no markdown, no explanation:
{{
  "claim": "...",
  "queries": ["...", "..."]
}}

Input JSON:
{data}
"""

    prompt = PromptTemplate(input_variables=['data'], template=template)
    parser = StrOutputParser()

    def safe_parse(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return ast.literal_eval(val)
        return {}

    def safe_json_parse(text: str) -> dict:
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse LLM output:\n{text}")

    state["domain_scores"] = safe_parse(state.get("domain_scores"))
    state["toxic_scores"]  = safe_parse(state.get("toxic_scores"))

    # ── ReAct trace ────────────────────────────────────────────────────────
    current_time = datetime.utcnow().isoformat()
    state.setdefault("thoughts",     []).append("Extracting claim and generating verification queries")
    state["actions"].append({
    "action"   : "LLM call: claim + query generation",
    "timestamp": current_time
    })
    state.setdefault("timestamps",   []).append(current_time)

    # ── LLM call ───────────────────────────────────────────────────────────
    formatted = prompt.format(data=json.dumps(state, indent=2))
    raw_response = llm.invoke(formatted)
    response_text = parser.invoke(raw_response).strip()
    parsed = safe_json_parse(response_text)

    claim   = parsed.get("claim", "")
    queries = parsed.get("queries", [])

    state.setdefault("observations", []).append(
        f"Claim: {claim} | Queries: {queries}"
    )

    domain_scores   = state["domain_scores"]
    dominant_domain = max(domain_scores, key=domain_scores.get) if domain_scores else None

    state["claim"]          = claim
    state["search_queries"] = queries
    state["search_query"]   = queries[0] if queries else None
    state["primary_domain"] = dominant_domain

    return state