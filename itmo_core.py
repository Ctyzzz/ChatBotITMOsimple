# itmo_core.py
"""
Core utilities for the ITMO Master Programs assistant.

Provides:
- Fetching & caching of program pages (AI and AI Product) and linked PDFs.
- Text extraction and chunking.
- Simple TF‑IDF search over chunks.
- Answer assembly restricted to loaded sources.
- Heuristic program recommendation and elective suggestion.

This module is used by the CLI, Flask app, and Telegram bots.

Author: chat assistant
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# pdfminer is optional; if absent, PDFs will be skipped gracefully.
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Constants ----------------------------------------------------------------

AI_URL = "https://abit.itmo.ru/program/master/ai"
AI_PRODUCT_URL = "https://abit.itmo.ru/program/master/ai_product"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.9"}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache_itmo")
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Fetching & extraction ----------------------------------------------------

def fetch(url: str, timeout: int = 20) -> Optional[bytes]:
    """HTTP GET with a tiny on-disk cache."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", url)
    path = os.path.join(CACHE_DIR, safe)
    if os.path.exists(path):
        try:
            return open(path, "rb").read()
        except Exception:
            pass
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.content
        with open(path, "wb") as f:
            f.write(data)
        return data
    except Exception:
        return None


def visible_text_from_html(html: bytes) -> str:
    """Extract visible text from HTML, stripping scripts/styles and normalizing whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = text.replace("\r", "")
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def find_pdf_links(html: bytes, base_url: str) -> List[str]:
    """Return up to a handful of absolute PDF links from a page."""
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            links.append(urljoin(base_url, href))
    out, seen = [], set()
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:6]  # keep it small & fast


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a PDF blob using pdfminer (if available)."""
    if pdf_extract_text is None:
        return ""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            path = tmp.name
        txt = pdf_extract_text(path) or ""
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{2,}", "\n", txt)
        return txt.strip()
    except Exception:
        return ""
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# --- Chunking -----------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    url: str
    source: str  # "ai" | "ai_product" | "pdf" | "html"
    start_idx: int
    has_elective_kw: bool = False


def chunk_text(text: str, url: str, source: str, maxlen: int = 700, overlap: int = 150) -> List[Chunk]:
    """Split text into overlapping chunks near sentence boundaries."""
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    joined = "\n".join(paragraphs)

    chunks: List[Chunk] = []
    i = 0
    while i < len(joined):
        window = joined[i:i + maxlen]
        j = i + maxlen
        # extend to end of sentence if possible
        m = re.search(r"[.!?]\s", joined[i:i + maxlen + 120])
        if m:
            end = i + m.end()
            window = joined[i:end]
            j = end
        kws = ["электив", "по выбору", "elective", "module", "модул"]
        has_elective = any(kw in window.lower() for kw in kws)
        chunks.append(Chunk(window, url, source, i, has_elective))
        i = max(j - overlap, i + maxlen)
    return chunks

# --- Index & search -----------------------------------------------------------

class ITMOIndex:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def build(self, urls: List[str]) -> None:
        """Fetch pages & PDFs, build TF‑IDF index."""
        docs: List[Chunk] = []
        for url in urls:
            html = fetch(url)
            if not html:
                continue
            text = visible_text_from_html(html)
            src = "ai" if ("master/ai" in url and "ai_product" not in url) else ("ai_product" if "ai_product" in url else "html")
            docs.extend(chunk_text(text, url, src))

            for purl in find_pdf_links(html, url):
                pdata = fetch(purl)
                if not pdata:
                    continue
                ptext = extract_text_from_pdf_bytes(pdata)
                if ptext.strip():
                    docs.extend(chunk_text(ptext, purl, "pdf"))

        # deduplicate near-identical chunks
        uniq: List[Chunk] = []
        seen = set()
        for ch in docs:
            key = (ch.source, ch.url, ch.text[:140])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(ch)
        self.chunks = uniq

        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, topk: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.vectorizer or self.matrix is None:
            return []
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix)[0]
        order = sims.argsort()[::-1]
        out: List[Tuple[Chunk, float]] = []
        for idx in order[:topk]:
            out.append((self.chunks[idx], float(sims[idx])))
        return out

# --- Answer assembly & helpers ------------------------------------------------

def make_answer_from_chunks(query: str, hits: List[Tuple[Chunk, float]], min_sim: float = 0.18) -> Optional[str]:
    """
    Build a short answer from top chunks. Returns None if top similarity is too low.
    Newlines are REAL newlines; no escaped sequences remain.
    """
    if not hits or hits[0][1] < min_sim:
        return None

    lines: List[str] = []
    sources: List[str] = []

    for ch, sim in hits[:3]:
        # choose 1–2 most overlapping sentences
        sentences = re.split(r"(?<=[.!?])\s+", ch.text)
        q_tokens = set(re.findall(r"[А-Яа-яA-Za-z0-9\-]+", query.lower()))
        scored = []
        for s in sentences:
            stoks = set(re.findall(r"[А-Яа-яA-Za-z0-9\-]+", s.lower()))
            overlap = len(q_tokens & stoks)
            scored.append((overlap, len(s), s))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        take = [s for _, __, s in scored[:2] if s.strip()]
        lines.extend(take)

        if ch.url not in sources:
            sources.append(ch.url)

    answer = " ".join(lines).strip()
    if not answer:
        answer = hits[0][0].text.strip()[:600]

    answer += "\n\nИсточники:\n" + "\n".join(f"- {s}" for s in sources)
    return answer


def is_offtopic(q: str) -> bool:
    """Heuristic filter: True if the question is clearly not about the two programs."""
    topics = [
        "магистр", "магистратур", "программа", "поступлен", "курс", "дисциплин",
        "учебн", "план", "предмет", "экзам", "итмо", "ai", "искусствен",
        "продакт", "управлени", "product", "программирован", "данн", "ml", "машин",
        "нейросет"
    ]
    ql = q.lower()
    return not any(t in ql for t in topics)


def recommend_program(profile: str) -> Tuple[str, Dict[str, int]]:
    """Very simple keyword-based scoring for program fit."""
    ai_kw = {
        "алгоритм", "математ", "вероятност", "статист", "теория", "исслед",
        "python", "c++", "ml", "машин", "deep", "нейросет", "dev", "инженер",
        "data", "cv", "nlp"
    }
    prod_kw = {
        "продукт", "product", "менедж", "roadmap", "стратег", "ux", "ui", "a/b", "ab",
        "маркет", "growth", "монетизац", "аналитик", "unit", "econom", "go-to-market",
        "команд", "коммуникац", "hypothesis"
    }
    text = profile.lower()
    score_ai = sum(k in text for k in ai_kw)
    score_prod = sum(k in text for k in prod_kw)

    if score_ai > score_prod + 1:
        choice = "AI (Искусственный интеллект)"
    elif score_prod > score_ai + 1:
        choice = "AI Product (Управление ИИ-продуктами)"
    else:
        choice = "Зависит от приоритетов: исследовательская/инженерная карьера → AI; product/менеджмент → AI Product."
    return choice, {"ai_score": score_ai, "product_score": score_prod}


def suggest_electives(index: ITMOIndex, interests: str) -> List[str]:
    """Return lines that likely refer to electives and match user's interests keywords."""
    hints: List[str] = []
    q = interests or ""
    hits = index.search(q, topk=12)
    # keep only chunks that mention electives or optional modules
    elect_chunks = [
        (ch, s) for ch, s in hits
        if ch.has_elective_kw or re.search(r"электив|по выбору|elective|module|модул", ch.text, re.I)
    ]
    seen = set()
    for ch, sim in elect_chunks:
        # split by newlines or list bullets
        for line in re.split(r"[\n•\-∙·]\s*", ch.text):
            if re.search(r"электив|по выбору|модул", line, re.I):
                line = line.strip()
                key = line.lower()[:180]
                if len(line) > 12 and key not in seen:
                    seen.add(key)
                    src = "(из AI)" if ch.source == "ai" else ("(из AI Product)" if ch.source == "ai_product" else "(из PDF)")
                    hints.append(f"{line} {src}")
            if len(hints) >= 8:
                break
        if len(hints) >= 8:
            break
    return hints
