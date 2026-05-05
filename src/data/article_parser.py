"""Generic article-body extraction helpers.

This module intentionally starts source-agnostic. It extracts a usable body
from common news-page HTML and gives the fetcher structured parse output. If a
source needs special handling later, source-specific parsers can wrap or
replace ``parse_article_html`` without changing the DB contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

MIN_ARTICLE_CHARS = 200

_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")
_DROP_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "form",
    "header",
    "footer",
    "nav",
    "aside",
)


@dataclass(frozen=True)
class ParsedArticle:
    title: str | None
    summary: str | None
    published_date: str | None
    article_text: str
    cleaned_article_text: str
    parse_status: str
    parse_error: str | None = None


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control-like clutter from text."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def _meta_content(soup: BeautifulSoup, *keys: str) -> str | None:
    for key in keys:
        attr, value = key.split("=", 1)
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            return clean_text(str(tag["content"]))
    return None


def _extract_title(soup: BeautifulSoup) -> str | None:
    title = _meta_content(soup, "property=og:title", "name=twitter:title")
    if title:
        return title
    if soup.title and soup.title.string:
        return clean_text(soup.title.string)
    h1 = soup.find("h1")
    return clean_text(h1.get_text(" ")) if h1 else None


def _extract_summary(soup: BeautifulSoup) -> str | None:
    return _meta_content(
        soup,
        "property=og:description",
        "name=description",
        "name=twitter:description",
    )


def _extract_published_date(soup: BeautifulSoup) -> str | None:
    return _meta_content(
        soup,
        "property=article:published_time",
        "name=pubdate",
        "name=date",
        "itemprop=datePublished",
    )


def _drop_noise(soup: BeautifulSoup) -> None:
    for tag in soup.select(",".join(_DROP_SELECTORS)):
        tag.decompose()


def _paragraphs_from(root: Any) -> list[str]:
    paragraphs: list[str] = []
    for tag in root.find_all(["p", "h2"], recursive=True):
        text = clean_text(tag.get_text(" "))
        if len(text) < 30:
            continue
        lowered = text.lower()
        if lowered.startswith(("abone ol", "son dakika", "reklam")):
            continue
        paragraphs.append(text)
    return paragraphs


def _best_text_container(soup: BeautifulSoup) -> Any:
    candidates = soup.find_all(["article", "main"])
    if not candidates and soup.body is not None:
        candidates = [soup.body]
    if not candidates:
        return soup
    return max(candidates, key=lambda node: len(" ".join(_paragraphs_from(node))))


def parse_article_html(html: str) -> ParsedArticle:
    """Extract body text and metadata from one article HTML document."""
    soup = BeautifulSoup(html or "", "html.parser")
    _drop_noise(soup)

    title = _extract_title(soup)
    summary = _extract_summary(soup)
    published_date = _extract_published_date(soup)

    root = _best_text_container(soup)
    paragraphs = _paragraphs_from(root)
    article_text = "\n\n".join(paragraphs)
    cleaned = clean_text(article_text)

    if len(cleaned) < MIN_ARTICLE_CHARS:
        return ParsedArticle(
            title=title,
            summary=summary,
            published_date=published_date,
            article_text=article_text,
            cleaned_article_text=cleaned,
            parse_status="empty",
            parse_error=f"article text shorter than {MIN_ARTICLE_CHARS} chars",
        )

    return ParsedArticle(
        title=title,
        summary=summary,
        published_date=published_date,
        article_text=article_text,
        cleaned_article_text=cleaned,
        parse_status="parsed",
    )
