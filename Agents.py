import os
import requests
from newsapi import NewsApiClient 
from datetime import datetime, timedelta
from types import SimpleNamespace

from langchain_classic.agents import initialize_agent, AgentType
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_community.utilities import SerpAPIWrapper
import Ollama_LLM


newsapi = NewsApiClient(api_key=os.getenv('news_api_key'))
Ollama_LLM = Ollama_LLM.ollama_llm

def Safe_News_API(query:str, domain:str = '')-> list:
    domain_suffix = {
        "domain_politics": "politics",
        "domain_movie":    "entertainment"
    }.get(domain, "")

    to_date = datetime.now().date()
    from_date = to_date-timedelta(days=7)

    try:
        articles = newsapi.get_everything(
            q = f'{query} {domain_suffix}'. strip(),
            from_param = from_date,
            to_param = to_date,
            language='en',
            sort_by = 'relevancy'
        )
        return[
            {
                "title"      : a.get("title"),
                "source"     : a.get("source", {}).get("name"),
                "description": a.get("description"),
                "url"        : a.get("url"),
                "published"  : a.get("publishedAt"),
            }
            for a in articles.get('articles',[])
        ]
    except Exception as e:
        print(f'NewAPI failed:{e}')
        return[]

news_tool = StructuredTool.from_function(
    func=Safe_News_API,
    name='NewsAPI',
    description='Search recent news articles by query and domain (politics/entertainment).',
    handle_validation_error=True
)


def Serpapi_call(query: str)->str:
    try:
        search = SerpAPIWrapper(serpapi_api_key=os.getenv('serpapi_key'))
        result = search.run(query)
        return result if result and result.strip() else "No result found"
    except Exception as e:
        return f"SerpAPI failed: {e}"


serpapi_tool= StructuredTool.from_function(
        name="Google Search",
        func=Serpapi_call,
        description="Useful for answering current events and factual verification of the queries",
        handle_validation_error=True
)

def Safe_tmdb_call(query:str)->list:
    try:
        url = "https://api.themoviedb.org/3/search/multi"
        params = {
            "api_key": os.getenv('TMDB_key'),
            "query"  : query,
            "language": "en-US",
            "page"   : 1
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        results = resp.json().get("results", [])[:5]
        return [
            {
                "title"      : r.get("title") or r.get("name"),
                "media_type" : r.get("media_type"),
                "overview"   : r.get("overview"),
                "popularity" : r.get("popularity"),
                "release"    : r.get("release_date") or r.get("first_air_date"),
            }
            for r in results
        ]
    except Exception as e:
        print(f"TMDB failed: {e}")
        return []

tmdb_tool = StructuredTool.from_function(
    func=Safe_tmdb_call,
    name="TMDBSearch",
    description="Search TMDB for movies, TV shows, and celebrities.",
    handle_validation_error=True
)


def _run_agent(tools: list, query: str, label: str) -> dict:
    """
    Runs a zero-shot ReAct agent with given tools.
    Returns dict with result, thoughts, and status.
    When the other agents fails This model will come in handy.
    """
    agent = initialize_agent(
        tools=tools,
        llm=Ollama_LLM,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=4
    )
    thought  = f"[{label}] Running agent on query: {query}"
    try:
        response = agent.run(query)
        return {
            "agent"      : label,
            "query"      : query,
            "result"     : response,
            "thought"    : thought,
            "action"     : f"{label} agent completed successfully",
            "observation": f"{label} result: {str(response)[:300]}",
            "status"     : "success"
        }
    except Exception as e:
        return {
            "agent"      : label,
            "query"      : query,
            "result"     : "",
            "thought"    : thought,
            "action"     : f"{label} agent failed: {e}",
            "observation": f"{label} failed with: {str(e)[:200]}",
            "status"     : "failed"
        }


def run_politics_agent(query:str, domain:str)->dict:
    news_result = Safe_News_API(query, domain)
    if news_result:
        return{
            'agent' : 'politics',
            'query' : query,
            'result' : news_result,
            'thought' : f"[politics] NewsAPI returned {len(news_result)} articles",
            'action' : 'NewsAPI call',
            "observation": f"Top article: {news_result[0].get('title', '')}",
            "status"     : "success"
        }
        # Fallback to SerpAPI agent
    return _run_agent([serpapi_tool], query, label="politics-serp-fallback")

def run_movie_agent(query: str, domain: str) -> dict:
    """TMDB primary → NewsAPI → SerpAPI fallback."""
    tmdb_result = Safe_tmdb_call(query)
    if tmdb_result:
        return {
            "agent"      : "movie",
            "query"      : query,
            "result"     : tmdb_result,
            "thought"    : f"[movie] TMDB returned {len(tmdb_result)} results",
            "action"     : "TMDB call",
            "observation": f"Top result: {tmdb_result[0].get('title', '')}",
            "status"     : "success"
        }
    # Fallback 1: NewsAPI
    news_result = Safe_News_API(query, domain)
    if news_result:
        return {
            "agent"      : "movie-news-fallback",
            "query"      : query,
            "result"     : news_result,
            "thought"    : f"[movie] TMDB failed, NewsAPI returned {len(news_result)} articles",
            "action"     : "NewsAPI fallback call",
            "observation": f"Top article: {news_result[0].get('title', '')}",
            "status"     : "success"
        }
    # Fallback 2: SerpAPI
    return _run_agent([serpapi_tool], query, label="movie-serp-fallback")

def run_general_agent(query: str, domain: str) -> dict:
    """NewsAPI primary → SerpAPI fallback."""
    news_result = Safe_News_API(query, domain)
    if news_result:
        return {
            "agent"      : "general",
            "query"      : query,
            "result"     : news_result,
            "thought"    : f"[general] NewsAPI returned {len(news_result)} articles",
            "action"     : "NewsAPI call",
            "observation": f"Top article: {news_result[0].get('title', '')}",
            "status"     : "success"
        }
    print("⚠️  NewsAPI empty → falling back to SerpAPI (general)")
    return _run_agent([serpapi_tool], query, label="general-serp-fallback")
