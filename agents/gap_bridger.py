"""
TARKA — Gap Bridger Agent

The critical "Fact D" solver.  When the anomaly detector finds
two entities that SHOULD be connected but aren't, this agent
autonomously:
  1. Queries Qwen3's internal knowledge
  2. If unsure, searches DuckDuckGo for the missing fact
  3. Parses results and adds inferred edges to the graph
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional, Callable

ProgressCallback = Callable[[int, float, str], None]

import config
from agents.llm_client import call_llm
from graph.knowledge_graph import IntelGraph
from graph.models import (
    Entity,
    EntityType,
    GapHypothesis,
    Relationship,
    RelationType,
)

log = logging.getLogger("tarka.gap_bridger")


# ── LLM Prompt for Internal-Knowledge Query ──────────────────

BRIDGE_PROMPT = """
You are a factual knowledge assistant and senior intelligence analyst. I need to determine if there is a highly probable structural, operational, logical, or dependency relationship between two entities.

Entity A: {entity_a}
Entity B: {entity_b}

Context (why I'm asking): {reason}

Answer with ONLY a valid JSON object:
{{
  "has_connection": true/false,
  "confidence": 0.0 to 1.0,
  "relationship_type": "hosts|uses|owned_by|developed_by|deployed_on|operates|related_to|located_in",
  "description": "brief explanation of the connection or logical deduction",
  "bridging_entity": "name of any intermediary entity if applicable, or null",
  "source_is_a": true/false  (true if A relates to B, false if B relates to A)
}}

You may use logical deduction based on the Context provided. If a connection is highly probable given standard corporate/operational structures, set has_connection to true and explain the deduction in the description.
"""

# ── DuckDuckGo Search Prompt ─────────────────────────────────

SEARCH_ANALYSIS_PROMPT = """
Given the following search results about a potential connection between "{entity_a}" and "{entity_b}", extract any factual dependency or relationship.

Search results:
{search_results}

Answer with ONLY a valid JSON object:
{{
  "has_connection": true/false,
  "confidence": 0.0 to 1.0,
  "relationship_type": "hosts|uses|owned_by|developed_by|deployed_on|operates|related_to",
  "description": "explanation citing the search evidence",
  "source_is_a": true/false
}}
"""


async def bridge_gaps(
    gaps: list[GapHypothesis],
    graph: IntelGraph,
    on_progress: Optional[ProgressCallback] = None,
) -> list[GapHypothesis]:
    """
    Attempt to bridge each gap hypothesis by:
    1. Querying Qwen3 internal knowledge
    2. If low confidence → DuckDuckGo search
    3. Add inferred edges on success

    Returns the list of gaps with updated resolution status.
    """
    for gap in gaps:
        if gap.resolved:
            continue

        log.info(
            f"🔍 Bridging gap: '{gap.entity_a_name}' ↔ '{gap.entity_b_name}'"
        )

        # Step 1: Ask the LLM from its internal knowledge
        result = await _query_llm_knowledge(gap, on_progress=on_progress)

        if result and result.get("has_connection") and result.get("confidence", 0) >= 0.5:
            log.info(
                f"  ✓ LLM knows: {result.get('description', '?')} "
                f"(confidence={result.get('confidence', 0):.2f})"
            )
            await _add_bridge_to_graph(
                gap, result, graph,
                confidence=min(result.get("confidence", 0.5), config.GAP_BRIDGE_LLM_CONFIDENCE),
                source="llm_knowledge",
            )
            gap.resolved = True
            gap.bridging_facts.append(result.get("description", ""))
            continue

        # Step 2: LLM unsure → search the web
        log.info("  → LLM uncertain, searching DuckDuckGo...")
        search_result = await _search_and_analyze(gap, on_progress=on_progress)

        if search_result and search_result.get("has_connection") and search_result.get("confidence", 0) >= 0.4:
            log.info(
                f"  ✓ Search found: {search_result.get('description', '?')} "
                f"(confidence={search_result.get('confidence', 0):.2f})"
            )
            await _add_bridge_to_graph(
                gap, search_result, graph,
                confidence=min(search_result.get("confidence", 0.7), config.GAP_BRIDGE_SEARCH_CONFIDENCE),
                source="web_search",
            )
            gap.resolved = True
            gap.bridging_facts.append(search_result.get("description", ""))
        else:
            log.warning(
                f"  ✗ Could not bridge: '{gap.entity_a_name}' ↔ '{gap.entity_b_name}'"
            )

    return gaps


# ── Step 1: LLM Internal Knowledge Query ─────────────────────

async def _query_llm_knowledge(gap: GapHypothesis, on_progress: Optional[ProgressCallback] = None) -> Optional[dict]:
    prompt = BRIDGE_PROMPT.format(
        entity_a=gap.entity_a_name,
        entity_b=gap.entity_b_name,
        reason=gap.reason,
    )
    raw = await call_llm(prompt, on_progress=on_progress)
    return _parse_json(raw)


# ── Step 2: DuckDuckGo Search + LLM Analysis ─────────────────

async def _search_and_analyze(gap: GapHypothesis, on_progress: Optional[ProgressCallback] = None) -> Optional[dict]:
    """Search DuckDuckGo for the connection, then have Qwen3 analyze."""
    queries = [
        f"{gap.entity_a_name} {gap.entity_b_name} relationship",
        f"{gap.entity_a_name} infrastructure dependency {gap.entity_b_name}",
        f"does {gap.entity_a_name} use {gap.entity_b_name}",
    ]

    all_results: list[str] = []
    for query in queries:
        results = await _duckduckgo_search(query)
        all_results.extend(results)
        if len(all_results) >= config.DDG_MAX_RESULTS:
            break

    if not all_results:
        log.warning("  No DuckDuckGo results found, falling back to logical inference...")
        # Fallback: Ask LLM to make its best logical guess based on the reason
        fallback_prompt = f"""
Given that a direct web search failed, use your analytical reasoning to infer if there is a highly probable connection between "{gap.entity_a_name}" and "{gap.entity_b_name}". 
Context for the gap: {gap.reason}

Answer with ONLY a valid JSON object:
{{
  "has_connection": true/false,
  "confidence": 0.0 to 1.0 (keep this low, max 0.6 if inferring),
  "relationship_type": "hosts|uses|owned_by|developed_by|deployed_on|operates|related_to",
  "description": "logical inference explanation",
  "source_is_a": true/false
}}
"""
        raw = await call_llm(fallback_prompt, on_progress=on_progress)
        return _parse_json(raw)

    search_text = "\n\n".join(all_results[: config.DDG_MAX_RESULTS])

    prompt = SEARCH_ANALYSIS_PROMPT.format(
        entity_a=gap.entity_a_name,
        entity_b=gap.entity_b_name,
        search_results=search_text,
    )
    raw = await call_llm(prompt, on_progress=on_progress)
    return _parse_json(raw)


async def _duckduckgo_search(query: str) -> list[str]:
    """Run a DuckDuckGo text search and return result snippets."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=config.DDG_MAX_RESULTS):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                results.append(f"[{title}]({href}): {body}")
        return results
    except Exception as exc:
        log.error(f"DuckDuckGo search failed: {exc}")
        return []


# ── Add Inferred Edge to Graph ────────────────────────────────

async def _add_bridge_to_graph(
    gap: GapHypothesis,
    result: dict,
    graph: IntelGraph,
    confidence: float,
    source: str,
) -> None:
    """Create the inferred edge in the knowledge graph."""
    rtype_str = result.get("relationship_type", "related_to")
    try:
        rtype = RelationType(rtype_str)
    except ValueError:
        rtype = RelationType.RELATED_TO

    # Determine direction
    source_is_a = result.get("source_is_a", True)
    if source_is_a:
        src_id = gap.entity_a_id
        tgt_id = gap.entity_b_id
    else:
        src_id = gap.entity_b_id
        tgt_id = gap.entity_a_id

    # If there's a bridging entity, add it
    bridging_name = result.get("bridging_entity")
    if bridging_name and bridging_name != "null":
        bridge_entity = Entity(
            name=bridging_name,
            entity_type=EntityType.FACT,
            description=f"Bridging entity discovered during gap analysis",
            source=source,
            confidence=confidence,
        )
        bridge_entity = graph.add_entity(bridge_entity)

        # src → bridge → tgt
        rel1 = Relationship(
            source_entity_id=src_id,
            target_entity_id=bridge_entity.id,
            relation_type=rtype,
            evidence=result.get("description", ""),
            source_signal=source,
            confidence=confidence,
            is_inferred=True,
        )
        rel2 = Relationship(
            source_entity_id=bridge_entity.id,
            target_entity_id=tgt_id,
            relation_type=rtype,
            evidence=result.get("description", ""),
            source_signal=source,
            confidence=confidence,
            is_inferred=True,
        )
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)
    else:
        rel = Relationship(
            source_entity_id=src_id,
            target_entity_id=tgt_id,
            relation_type=rtype,
            evidence=result.get("description", ""),
            source_signal=source,
            confidence=confidence,
            is_inferred=True,
        )
        graph.add_relationship(rel)


# ── Ollama + JSON Helpers ─────────────────────────────────────



def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None

    def try_parse(s: str) -> Optional[dict]:
        s = s.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", s)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    parsed = try_parse(text)
    if parsed: return parsed

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        parsed = try_parse(match.group(1))
        if parsed: return parsed

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        parsed = try_parse(match.group(1))
        if parsed: return parsed

    log.error(f"Failed to parse any JSON from LLM output. Raw LLM output:\n{text}")
    return None
