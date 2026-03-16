"""
Writer Agent - Generates full literature review in Markdown and LaTeX.
Uses the WRITER Groq API key (GROQ_API_KEY_WRITER).
Includes APA 7th edition reference generation.
"""
import asyncio
import logging
import re
from typing import Dict, List, Tuple
from utils.groq_client import groq_chat_writer

logger = logging.getLogger(__name__)


# ── Citation & Reference Utilities ───────────────────────────────────────────

def build_citation_key(paper: Dict) -> str:
    """Generate a BibTeX citation key."""
    authors = paper.get("authors", [])
    if authors:
        a = authors[0]
        name = a.get("name", "") if isinstance(a, dict) else str(a)
        last = name.strip().split()[-1] if name.strip() else "Unknown"
    else:
        last = "Unknown"
    year = paper.get("year", 2020)
    title_words = (paper.get("title") or "").split()
    first_word = re.sub(r'[^a-zA-Z]', '', title_words[0]).lower() if title_words else "paper"
    return f"{last.lower()}{year}{first_word}"


def _author_names(paper: Dict) -> List[str]:
    """Extract plain author name strings."""
    raw = paper.get("authors", [])
    names = []
    for a in raw:
        if isinstance(a, dict):
            names.append(a.get("name", ""))
        else:
            names.append(str(a))
    return [n for n in names if n]


def generate_bibtex(papers: List[Dict]) -> str:
    """Generate BibTeX entries for all papers."""
    entries = []
    for paper in papers:
        key    = build_citation_key(paper)
        authors = _author_names(paper) or ["Unknown"]
        author_str = " and ".join(authors)
        title  = (paper.get("title") or "Unknown Title").replace("{", "").replace("}", "")
        year   = paper.get("year", 2020)
        venue  = paper.get("venue", "") or ""
        ids    = paper.get("externalIds") or {}
        doi    = ids.get("DOI", "")
        arxiv  = ids.get("ArXiv", "")

        entry_type = "inproceedings" if any(
            w in venue.lower() for w in
            ["proceedings", "conference", "workshop", "symposium",
             "iclr", "neurips", "icml", "cvpr", "emnlp", "acl", "aaai", "ijcai"]
        ) else "article"

        e  = f"@{entry_type}{{{key},\n"
        e += f"  author = {{{author_str}}},\n"
        e += f"  title  = {{{{{title}}}}},\n"
        e += f"  year   = {{{year}}},\n"
        if entry_type == "inproceedings":
            e += f"  booktitle = {{{venue}}},\n"
        else:
            e += f"  journal = {{{venue}}},\n"
        if doi:   e += f"  doi = {{{doi}}},\n"
        if arxiv: e += f"  eprint = {{{arxiv}}},\n  archivePrefix = {{arXiv}},\n"
        e += "}"
        entries.append(e)
    return "\n\n".join(entries)


def generate_apa_references(papers: List[Dict]) -> str:
    """
    Generate APA 7th edition reference list.
    Format: Author, A. A., & Author, B. B. (Year). Title. Journal. DOI/URL
    """
    def fmt_author(name: str) -> str:
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0]
        last     = parts[-1]
        initials = ". ".join(p[0].upper() for p in parts[:-1]) + "."
        return f"{last}, {initials}"

    lines = []
    for i, paper in enumerate(papers):
        authors = _author_names(paper)
        year    = paper.get("year", "n.d.")
        title   = (paper.get("title") or "Untitled").strip()
        venue   = (paper.get("venue") or "").strip()
        ids     = paper.get("externalIds") or {}
        doi     = ids.get("DOI", "")
        arxiv   = ids.get("ArXiv", "")
        url     = paper.get("url", "") or ""

        if not authors:
            author_str = "Unknown Author"
        elif len(authors) == 1:
            author_str = fmt_author(authors[0])
        elif len(authors) <= 7:
            fmt = [fmt_author(a) for a in authors]
            author_str = ", ".join(fmt[:-1]) + ", & " + fmt[-1]
        else:
            fmt = [fmt_author(a) for a in authors]
            author_str = ", ".join(fmt[:6]) + ", ... " + fmt[-1]

        ref = f"{author_str} ({year}). {title}."
        if venue:
            ref += f" *{venue}*."
        if doi:
            ref += f" https://doi.org/{doi}"
        elif arxiv:
            ref += f" arXiv:{arxiv}. https://arxiv.org/abs/{arxiv}"
        elif url:
            ref += f" {url}"

        lines.append(f"[{i+1}] {ref}")

    header = "# References\n\n*(APA 7th Edition)*\n\n"
    return header + "\n\n".join(lines)


# ── Section Writers ───────────────────────────────────────────────────────────

async def generate_markdown_review(topic: str, papers: List[Dict], analysis: Dict) -> str:
    """Generate the full literature review in Markdown."""

    themes     = analysis.get("themes", {})
    gaps       = analysis.get("gaps", {})
    comparison = analysis.get("comparison", {})
    papers_text = analysis.get("papers_text_for_writer", "")

    # Build citation reference block
    citation_map = {}
    for i, p in enumerate(papers):
        names = _author_names(p)
        citation_map[i + 1] = {
            "key":     build_citation_key(p),
            "title":   p.get("title", ""),
            "year":    p.get("year", ""),
            "authors": names,
            "venue":   p.get("venue", "") or "",
        }

    ref_block = "\n".join([
        f"[{i}] {', '.join(info['authors'][:3])}{'et al.' if len(info['authors']) > 3 else ''} "
        f"({info['year']}). {info['title']}."
        for i, info in citation_map.items()
    ])

    # Abstract + Introduction
    intro_section = await groq_chat_writer(
        system_prompt=(
            "You are a senior academic researcher writing a publishable literature review paper. "
            "Write in formal academic style. Be thorough, precise, and insightful. "
            "Use inline citations like [1], [2], [1,3] etc."
        ),
        user_prompt=f"""Write an Abstract and Introduction for a literature review on: "{topic}"

Papers covered (indices for citation):
{ref_block}

Research themes identified:
{themes}

Write:
1. **Abstract** (150-200 words): Overview of topic, scope, number of papers reviewed, key findings
2. **1. Introduction** (400-500 words): Background, motivation, why this review matters, scope and structure

Use academic writing. Cite relevant papers with [number] format.""",
        max_tokens=1500,
    )

    # Background
    background_section = await groq_chat_writer(
        system_prompt="You are a senior academic researcher writing a publishable literature review. Use formal academic style and inline citations [N].",
        user_prompt=f"""Topic: "{topic}"

Papers:
{papers_text[:2500]}

Write **2. Background and Preliminaries** (350-450 words):
- Fundamental concepts required to understand this research area
- Historical context and key milestones
- Cite specific papers where appropriate [N]""",
        max_tokens=1000,
    )

    # Thematic sections
    theme_list = themes.get("themes", [])
    thematic_content = []
    for i, theme in enumerate(theme_list[:5]):
        section_num  = i + 3
        theme_name   = theme.get("name", f"Theme {i+1}")
        theme_desc   = theme.get("description", "")
        paper_indices = theme.get("paper_indices", [])
        key_finding  = theme.get("key_finding", "")

        theme_section = await groq_chat_writer(
            system_prompt="You are writing a literature review. Use formal academic style and cite papers as [N].",
            user_prompt=f"""Write section {section_num} for a literature review on "{topic}":

Section title: "{theme_name}"
Description: {theme_desc}
Key finding: {key_finding}
Relevant paper indices: {paper_indices}

Papers:
{papers_text[:2000]}

Write a thorough academic section (300-400 words) covering:
- Main approaches and contributions in this theme
- Comparison of different methods
- Key results and insights
- Cite papers as [N] inline

Start with: **{section_num}. {theme_name}**""",
            max_tokens=900,
        )
        thematic_content.append(theme_section)
        await asyncio.sleep(0.5)

    # Comparative analysis
    comparison_section = await groq_chat_writer(
        system_prompt="You are writing a literature review. Use formal academic style.",
        user_prompt=f"""Topic: "{topic}"

Comparison data:
{comparison}

Research gaps:
{gaps}

Write **{len(theme_list)+3}. Comparative Analysis and Discussion** (400-500 words):
- Systematic comparison of methodologies
- Key similarities and differences
- What the field has learned
- Contradictions or debates in the literature
- Cite papers as [N] inline

Also include a short subsection on **Research Gaps** listing the main open problems.""",
        max_tokens=1000,
    )

    # Conclusion
    conclusion_section = await groq_chat_writer(
        system_prompt="You are writing a literature review conclusion. Be concise and forward-looking.",
        user_prompt=f"""Topic: "{topic}"
Gaps: {gaps.get('gaps', [])}
Future directions: {gaps.get('future_directions', [])}
Consensus: {gaps.get('consensus', [])}

Write **{len(theme_list)+4}. Conclusion and Future Directions** (300-400 words):
- Summary of key findings across the literature
- State of the field assessment
- Most promising future research directions
- Closing remarks on the field's trajectory""",
        max_tokens=800,
    )

    maturity = comparison.get("field_maturity", "developing")

    full_review = f"""# A Survey of {topic}: Methods, Advances, and Future Directions

*Automatically generated literature review covering {len(papers)} papers*

---

{intro_section}

---

{background_section}

---

{"---".join(thematic_content)}

---

{comparison_section}

---

{conclusion_section}

---

## References

{ref_block}

---

*Generated by LitReview AI | Papers reviewed: {len(papers)} | Field maturity: {maturity}*
"""
    return full_review


def markdown_to_latex(topic: str, markdown_text: str, papers: List[Dict]) -> str:
    """Convert the markdown review to a LaTeX document."""
    safe_topic = re.sub(r'[&%_#]', lambda m: '\\' + m.group(), topic)

    body = markdown_text
    body = re.sub(r'^# .+\n',               '', body, flags=re.MULTILINE)
    body = re.sub(r'^\*Automatically.+\*\n','', body, flags=re.MULTILINE)
    body = re.sub(r'^---+$',                '', body, flags=re.MULTILINE)
    body = re.sub(r'^\*Generated by.+\*$',  '', body, flags=re.MULTILINE)
    body = re.sub(r'^## (.+)$',  r'\\section{\1}',    body, flags=re.MULTILINE)
    body = re.sub(r'^### (.+)$', r'\\subsection{\1}', body, flags=re.MULTILINE)
    body = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', body)
    body = re.sub(r'\*(.+?)\*',     r'\\textit{\1}',  body)

    m = re.search(r'## References\n(.*?)$', body, re.DOTALL)
    if m:
        body = body[:m.start()]

    body = re.sub(r'(?<!\\)&', r'\\&', body)
    body = re.sub(r'(?<!\\)%', r'\\%', body)

    abs_match = re.search(r'Abstract[^\n]*\n+(.+?)(?:\\section|\Z)', body, re.DOTALL)
    abstract_text = abs_match.group(1).strip()[:800] if abs_match else \
        f"A comprehensive literature review on {safe_topic}."

    latex_doc = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{parskip}
\usepackage{microtype}
\usepackage{xcolor}
\geometry{margin=1in}
\doublespacing
\hypersetup{colorlinks=true,linkcolor=blue!70!black,citecolor=green!50!black,urlcolor=blue!70!black}

\title{\textbf{A Survey of """ + safe_topic + r"""}: \\
       Methods, Advances, and Future Directions}
\author{LitReview AI \\ \small{AI-Assisted Research Synthesis}}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
""" + abstract_text + r"""
\end{abstract}

\tableofcontents
\newpage

""" + body + r"""

\newpage
\section*{References}
\begin{thebibliography}{99}
"""

    for paper in papers:
        key    = build_citation_key(paper)
        names  = _author_names(paper) or ["Unknown"]
        astr   = ", ".join(names[:4]) + (" et al." if len(names) > 4 else "")
        title  = (paper.get("title") or "Unknown").replace("{", "").replace("}", "")
        year   = paper.get("year", "N/A")
        venue  = paper.get("venue", "") or ""
        ids    = paper.get("externalIds") or {}
        doi    = ids.get("DOI", "")
        arxiv  = ids.get("ArXiv", "")
        url    = paper.get("url", "") or ""

        entry  = f"\\bibitem{{{key}}}\n{astr}.\n\\textit{{{title}}}.\n{venue}, {year}."
        if doi:   entry += f" \\url{{https://doi.org/{doi}}}"
        elif arxiv: entry += f" arXiv:{arxiv}"
        elif url:   entry += f" \\url{{{url}}}"
        latex_doc += entry + "\n\n"

    latex_doc += r"""\end{thebibliography}
\end{document}"""

    return latex_doc


# ── Main Entry Point ─────────────────────────────────────────────────────────

async def run_writer_agent(
    topic: str,
    papers: List[Dict],
    analysis: Dict,
    job_id: str,
    state_manager,
) -> Tuple[str, str, str]:
    """
    Generate literature review in Markdown + LaTeX + APA references.
    Returns: (markdown, latex, apa)
    """
    state_manager.update(job_id, current_agent="✍️ Writer Agent", progress=75)
    state_manager.add_log(job_id, "Generating literature review text (writer key)...")

    state_manager.add_log(job_id, "Writing introduction and background...")
    markdown = await generate_markdown_review(topic, papers, analysis)
    state_manager.update(job_id, progress=90)

    state_manager.add_log(job_id, "Converting to LaTeX format...")
    latex = markdown_to_latex(topic, markdown, papers)
    state_manager.update(job_id, progress=93)

    state_manager.add_log(job_id, "Generating APA 7th edition references...")
    apa = generate_apa_references(papers)
    state_manager.update(job_id, progress=96)

    state_manager.add_log(job_id, "Literature review generation complete! 🎉", "success")
    return markdown, latex, apa
