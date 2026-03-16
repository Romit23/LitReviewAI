"""
Comparison Agent - Compares papers, identifies themes and relationships.
Uses PRIMARY Groq API key.
"""
import logging
from typing import Dict, List
from utils.groq_client import groq_json, groq_chat

logger = logging.getLogger(__name__)


def format_papers_for_comparison(papers: List[Dict]) -> str:
    """Format paper summaries for the comparison prompt."""
    parts = []
    for i, p in enumerate(papers):
        s = p.get("summary", {})
        authors = p.get("authors", [])
        if authors and isinstance(authors[0], dict):
            author_names = [a.get("name", "") for a in authors[:3]]
        else:
            author_names = [str(a) for a in authors[:3]]
        author_str = ", ".join(author_names) + (" et al." if len(authors) > 3 else "")

        parts.append(f"""
Paper [{i+1}]: {p.get('title', 'Unknown')}
Authors: {author_str}
Year: {p.get('year', 'N/A')} | Venue: {p.get('venue', 'N/A')} | Citations: {p.get('citationCount', 0)}
Contribution: {s.get('key_contribution', 'N/A')}
Methodology: {s.get('methodology', 'N/A')}
Results: {s.get('results', 'N/A')}
Limitations: {s.get('limitations', 'N/A')}
Type: {s.get('paper_type', 'N/A')} | Domain: {s.get('domain', 'N/A')}
Keywords: {', '.join(s.get('keywords', []))}
""")
    return "\n---\n".join(parts)


async def run_comparison_agent(papers: List[Dict], topic: str, job_id: str, state_manager) -> Dict:
    """Comparison Agent: identifies themes, clusters, gaps, and relationships."""
    state_manager.update(job_id, current_agent="⚖️ Comparison Agent", progress=55)
    state_manager.add_log(job_id, "Comparing papers and identifying research themes")

    papers_text = format_papers_for_comparison(papers)

    # 1. Research themes
    state_manager.add_log(job_id, "Identifying research themes...")
    themes_result = await groq_json(
        system_prompt="You are an expert academic researcher performing a systematic literature analysis.",
        user_prompt=f"""Topic: "{topic}"

Here are {len(papers)} papers:
{papers_text}

Identify the main research themes/clusters in these papers.
Return JSON:
{{
  "themes": [
    {{
      "name": "Theme name",
      "description": "2-3 sentence description",
      "paper_indices": [1, 2, 3],
      "key_finding": "The key insight from papers in this theme"
    }}
  ],
  "evolution": "How the research has evolved over time in 2-3 sentences",
  "dominant_methods": ["method1", "method2", "method3"],
  "common_datasets": ["dataset1", "dataset2"]
}}""",
        max_tokens=1500,
    )
    state_manager.update(job_id, progress=62)

    # 2. Research gaps
    state_manager.add_log(job_id, "Identifying research gaps...")
    gaps_result = await groq_json(
        system_prompt="You are a senior researcher identifying open problems and research gaps.",
        user_prompt=f"""Topic: "{topic}"

Based on these papers:
{papers_text}

Identify research gaps and future directions.
Return JSON:
{{
  "gaps": [
    {{"gap": "Description of gap", "papers_noting_it": [1, 2], "severity": "high|medium|low"}}
  ],
  "future_directions": ["direction1", "direction2", "direction3", "direction4"],
  "contradictions": [
    {{"description": "Contradiction or debate", "paper_a": 1, "paper_b": 2}}
  ],
  "consensus": ["agreed point 1", "agreed point 2", "agreed point 3"]
}}""",
        max_tokens=1200,
    )
    state_manager.update(job_id, progress=70)

    # 3. Methodology comparison
    state_manager.add_log(job_id, "Building methodology comparison...")
    comparison_result = await groq_json(
        system_prompt="You are an expert at comparing research methodologies.",
        user_prompt=f"""Topic: "{topic}"

Papers:
{papers_text}

Create a structured comparison. Return JSON:
{{
  "comparison_dimensions": ["Dimension1", "Dimension2", "Dimension3", "Dimension4", "Dimension5"],
  "paper_comparisons": [
    {{
      "paper_index": 1,
      "title_short": "Short title",
      "values": ["value1", "value2", "value3", "value4", "value5"]
    }}
  ],
  "best_practices": ["practice1", "practice2", "practice3"],
  "field_maturity": "emerging|developing|mature",
  "field_maturity_reasoning": "Why you assess the field at this maturity level"
}}""",
        max_tokens=1500,
    )
    state_manager.update(job_id, progress=75)

    state_manager.add_log(job_id, "Comparison analysis complete", "success")

    return {
        "themes":                 themes_result,
        "gaps":                   gaps_result,
        "comparison":             comparison_result,
        "papers_text_for_writer": papers_text,
    }
