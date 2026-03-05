"""
TARKA — GDELT Source Ingestion

Queries the GDELT Project's free API for real-time global events,
converts them into raw text for entity extraction.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

import config

log = logging.getLogger("tarka.gdelt")


async def fetch_gdelt_events(
    keywords: list[str],
    timespan_minutes: int = 60,
    max_articles: int = 10,
    source_country: Optional[str] = None,
) -> list[str]:
    """
    Query GDELT DOC 2.0 API for recent news articles matching
    the given keywords.

    Args:
        keywords:         Search terms (e.g., ["Iran", "strike", "AWS"])
        timespan_minutes: How far back to search (default 60 min)
        max_articles:     Max articles to return
        source_country:   Optional 2-letter country code filter

    Returns:
        List of article text strings ready for entity extraction.
    """
    query = " ".join(keywords)
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_articles),
        "timespan": f"{timespan_minutes}min",
        "format": "json",
        "sort": "DateDesc",
    }

    if source_country:
        params["sourcecountry"] = source_country

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            resp = await client.get(config.GDELT_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        log.error("GDELT API request timed out")
        return []
    except Exception as exc:
        log.error(f"GDELT API request failed: {exc}")
        return []

    articles = data.get("articles", [])
    if not articles:
        log.info(f"No GDELT articles found for query: {query}")
        return []

    results: list[str] = []
    for article in articles[:max_articles]:
        title = article.get("title", "")
        url = article.get("url", "")
        seendate = article.get("seendate", "")
        source_name = article.get("source", "")
        domain = article.get("domain", "")

        # Compose a text block for entity extraction
        text_block = (
            f"[GDELT] {title}\n"
            f"Source: {source_name} ({domain})\n"
            f"Date: {seendate}\n"
            f"URL: {url}"
        )
        results.append(text_block)

    log.info(
        f"Fetched {len(results)} GDELT article(s) for '{query}'"
    )
    return results
