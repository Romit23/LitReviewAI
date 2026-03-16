"""
Paper Search Agent - Searches Semantic Scholar and ArXiv for relevant papers.

Key design rules:
- Queries must be SHORT (3-6 words max) — both APIs reject long/boolean queries silently
- Multiple short focused queries beat one long complex query
- Robust fallback chain: always returns papers even if LLM query expansion fails
"""
import httpx
import asyncio
import logging
import re
from typing import List, Dict, Optional
import feedparser
from utils.groq_client import groq_json

logger = logging.getLogger(__name__)

SEARCH_DELAY = 2.0  # seconds between API calls
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,authors,year,citationCount,externalIds,openAccessPdf,venue,url"


def clean_query(query: str) -> str:
    """
    Strip boolean operators and parentheses that break Semantic Scholar / ArXiv.
    Also truncate to ~60 chars so the APIs don't choke.
    """
    # Remove boolean operators and brackets
    q = re.sub(r'\b(AND|OR|NOT)\b', ' ', query)
    q = re.sub(r'[()"\[\]]', ' ', q)
    # Collapse whitespace
    q = re.sub(r'\s+', ' ', q).strip()
    # Hard cap at 80 chars, breaking at a word boundary
    if len(q) > 80:
        q = q[:80].rsplit(' ', 1)[0]
    return q


def normalize_semantic_scholar_paper(paper: Dict) -> Dict:
    return {
        "paperId":       paper.get("paperId"),
        "title":         paper.get("title", ""),
        "abstract":      paper.get("abstract", ""),
        "authors":       paper.get("authors", []),
        "year":          paper.get("year"),
        "citationCount": paper.get("citationCount", 0),
        "externalIds":   paper.get("externalIds", {}),
        "openAccessPdf": paper.get("openAccessPdf"),
        "venue":         paper.get("venue", ""),
        "url":           paper.get("url", ""),
        "source":        "semantic_scholar",
    }


async def search_semantic_scholar(query: str, limit: int = 15) -> List[Dict]:
    """Search Semantic Scholar with a clean, short query."""
    q = clean_query(query)
    if not q:
        return []
    params = {"query": q, "limit": limit, "fields": FIELDS}
    headers = {"User-Agent": "LitReviewBot/1.0 (research tool)"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                resp = await client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
                    params=params,
                    headers=headers,
                )
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    logger.warning(f"Semantic Scholar rate-limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", [])
                logger.info(f"SemanticScholar '{q}' → {len(results)} papers")
                return results
            except Exception as e:
                logger.warning(f"SemanticScholar attempt {attempt+1} failed for '{q}': {e}")
                await asyncio.sleep(3)
    return []


async def search_arxiv(query: str, limit: int = 15) -> List[Dict]:
    """Search ArXiv with a clean, short query."""
    q = clean_query(query)
    if not q:
        return []
    # ArXiv works best with ti+abs search for simple phrases
    params = {
        "search_query": f"ti:{q} OR abs:{q}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
    }
    headers = {"Accept": "application/atom+xml", "User-Agent": "LitReviewBot/1.0"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                resp = await client.get(
                    "https://export.arxiv.org/api/query",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
                papers = []
                for entry in feed.entries:
                    paper_id = entry.id.split('/')[-1] if '/' in entry.id else entry.id
                    # Strip version suffix e.g. "2301.00001v2" → "2301.00001"
                    paper_id = re.sub(r'v\d+$', '', paper_id)
                    papers.append({
                        "paperId":       entry.id,
                        "title":         entry.title.replace('\n', ' ').strip(),
                        "abstract":      entry.summary.replace('\n', ' ').strip(),
                        "authors":       [{"name": a.name} for a in getattr(entry, 'authors', [])],
                        "year":          int(entry.published[:4]) if getattr(entry, 'published', None) else None,
                        "citationCount": 0,
                        "externalIds":   {"ArXiv": paper_id},
                        "openAccessPdf": {"url": f"https://arxiv.org/pdf/{paper_id}.pdf"},
                        "venue":         "ArXiv",
                        "url":           entry.id,
                        "source":        "arxiv",
                    })
                logger.info(f"ArXiv '{q}' → {len(papers)} papers")
                return papers
            except Exception as e:
                logger.warning(f"ArXiv attempt {attempt+1} failed for '{q}': {e}")
                await asyncio.sleep(3)
    return []


async def expand_search_queries(topic: str) -> List[str]:
    """
    Ask Groq for 5 SHORT, simple keyword queries (no boolean operators).
    Falls back to auto-generated simple queries if LLM fails.
    """
    try:
        result = await groq_json(
            system_prompt=(
                "You are a research librarian. Generate short, simple keyword search queries "
                "suitable for academic search APIs like Semantic Scholar and ArXiv. "
                "Queries must be 2-5 words only. No boolean operators (AND/OR/NOT). "
                "No parentheses. No quotes. Just plain keyword phrases."
            ),
            user_prompt=f"""Generate 5 short keyword search queries (2-5 words each) to find academic papers about: "{topic}"

Rules:
- Maximum 5 words per query
- No AND, OR, NOT operators
- No parentheses or quotes
- Each query should cover a different angle of the topic

Return JSON: {{"queries": ["short query 1", "short query 2", "short query 3", "short query 4", "short query 5"]}}"""
        )
        raw_queries = result.get("queries", [])
        # Clean every query regardless — safety net
        queries = [clean_query(q) for q in raw_queries if q]
        queries = [q for q in queries if q]  # drop empties
        if queries:
            logger.info(f"Expanded queries: {queries}")
            return queries
    except Exception as e:
        logger.warning(f"Query expansion via Groq failed: {e}. Using fallback queries.")

    # ── Fallback: generate simple queries from the topic itself ──────────────
    words = topic.split()
    fallback = [
        topic,                                          # full topic as-is
        " ".join(words[:4]) if len(words) > 4 else topic,  # first 4 words
        " ".join(words[-4:]) if len(words) > 4 else topic, # last 4 words
    ]
    # Add "survey" and "review" variants
    core = " ".join(words[:3]) if len(words) >= 3 else topic
    fallback.append(f"{core} survey")
    fallback.append(f"{core} deep learning")
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in fallback:
        cq = clean_query(q)
        if cq and cq not in seen:
            seen.add(cq)
            unique.append(cq)
    logger.info(f"Fallback queries: {unique}")
    return unique


async def run_search_agent(topic: str, max_papers: int, job_id: str, state_manager) -> List[Dict]:
    """
    Main search agent: expands queries → searches SS + ArXiv → deduplicates → ranks.
    """
    state_manager.update(job_id, current_agent="🔍 Paper Search Agent", progress=5)
    state_manager.add_log(job_id, f"Generating search queries for: {topic}")

    queries = await expand_search_queries(topic)
    state_manager.add_log(job_id, f"Generated {len(queries)} search queries: {queries}")

    all_papers: Dict[str, Dict] = {}

    for i, query in enumerate(queries):
        state_manager.add_log(job_id, f"Searching: '{query}'")

        # Semantic Scholar
        ss_results = await search_semantic_scholar(query, limit=15)
        for paper in ss_results:
            normalized = normalize_semantic_scholar_paper(paper)
            key = normalized["title"].lower().strip()
            if key and key not in all_papers:
                all_papers[key] = normalized

        await asyncio.sleep(SEARCH_DELAY)

        # ArXiv
        arxiv_results = await search_arxiv(query, limit=15)
        for paper in arxiv_results:
            key = paper["title"].lower().strip()
            if key and key not in all_papers:
                all_papers[key] = paper

        await asyncio.sleep(SEARCH_DELAY)
        state_manager.update(job_id, progress=5 + int(10 * (i + 1) / len(queries)))

    # ── Hard fallback: if still nothing, search the raw topic directly ───────
    if not all_papers:
        logger.warning("All expanded queries returned 0 results — trying raw topic directly")
        state_manager.add_log(job_id, "No results yet — trying direct topic search...")
        for fallback_q in [topic, clean_query(topic)]:
            ss = await search_semantic_scholar(fallback_q, limit=max_papers)
            for paper in ss:
                n = normalize_semantic_scholar_paper(paper)
                all_papers[n["title"].lower().strip()] = n
            ax = await search_arxiv(fallback_q, limit=max_papers)
            for paper in ax:
                all_papers[paper["title"].lower().strip()] = paper
            if all_papers:
                break

    state_manager.add_log(job_id, f"Found {len(all_papers)} unique papers before filtering")

    # ── Filter: need abstract, drop if year unreasonably old ─────────────────
    filtered = [
        p for p in all_papers.values()
        if p.get("abstract") and len(p.get("abstract", "")) > 50
        and (p.get("year") is None or p.get("year", 0) >= 2010)
    ]

    if not filtered:
        # Relax filter completely — take everything with a title
        filtered = [p for p in all_papers.values() if p.get("title")]

    # ── Rank: citation count + recency ────────────────────────────────────────
    filtered.sort(
        key=lambda p: (
            (p.get("citationCount") or 0) * 0.7 +
            ((p.get("year") or 2015) - 2010) * 5
        ),
        reverse=True,
    )

    # ── AI-assisted relevance selection ──────────────────────────────────────
    if len(filtered) > max_papers:
        paper_summaries = [
            {
                "idx":       i,
                "title":     p.get("title", ""),
                "abstract":  (p.get("abstract") or "")[:150],
                "year":      p.get("year"),
                "citations": p.get("citationCount", 0),
            }
            for i, p in enumerate(filtered[:50])
        ]
        try:
            selection = await groq_json(
                system_prompt="You are a research expert selecting the most relevant papers for a literature review.",
                user_prompt=f"""Topic: "{topic}"

Papers:
{paper_summaries}

Select the {max_papers} most relevant and diverse papers.
Return JSON: {{"selected_indices": [0, 1, 2, ...]}}"""
            )
            indices = selection.get("selected_indices", [])
            if indices:
                filtered = [filtered[i] for i in indices if i < len(filtered)]
            else:
                filtered = filtered[:max_papers]
        except Exception as e:
            logger.warning(f"AI paper selection failed: {e} — using top-ranked")
            filtered = filtered[:max_papers]

    papers = filtered[:max_papers]

    state_manager.set_papers(
        job_id,
        [{"title": p.get("title"), "year": p.get("year"), "paperId": p.get("paperId")} for p in papers],
    )
    state_manager.add_log(job_id, f"Selected {len(papers)} papers for review", "success")
    state_manager.update(job_id, progress=20)
    return papers
