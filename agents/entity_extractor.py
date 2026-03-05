"""
TARKA — Entity & Relationship Extractor Agent

Uses Ollama qwen3:14b to extract structured entities and
relationships from raw OSINT text, then adds them to the
knowledge graph.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional, Callable
ProgressCallback = Callable[[int, float, str], None]

import config
from agents.llm_client import call_llm
from graph.knowledge_graph import IntelGraph
from graph.models import Entity, EntityType, Relationship, RelationType

log = logging.getLogger("tarka.extractor")

# ── Extraction Prompt ─────────────────────────────────────────

EXTRACTION_PROMPT = """
You are an OSINT intelligence analyst. Extract ALL entities and relationships from the following text.

Return ONLY a valid JSON object with this exact structure (no markdown, no commentary):
{
  "entities": [
    {
      "name": "exact name",
      "type": "actor|system|location|event|fact",
      "description": "brief description",
      "timestamp": "ISO 8601 or null"
    }
  ],
  "relationships": [
    {
      "source": "source entity name",
      "target": "target entity name",
      "relation": "attacks|hosts|uses|located_in|retaliates|causes|temporal_correlation|allied_with|opposed_to|operates|owned_by|developed_by|deployed_on|disrupts|affects|related_to",
      "evidence": "exact quote or paraphrase from text"
    }
  ]
}

Entity types:
- actor: nations, organizations, militaries, people
- system: technology platforms, AI models, software, infrastructure
- location: physical places, data centers, military bases, cities
- event: strikes, attacks, outages, deployments, operations
- fact: verified static facts (hosting relationships, ownership, etc.)

Rules:
- Extract EVERY entity mentioned, even minor ones
- Infer implicit relationships (e.g., "X struck Y's base" → X attacks base, base located_in Y)
- Use the most specific relation type available
- Timestamps should be ISO 8601 format when extractable, null otherwise
- Entity names should be canonical (e.g., "United States" not "the US")

TEXT TO ANALYZE:
"""


async def extract_entities_and_relationships(
    text: str,
    graph: IntelGraph,
    source_label: str = "manual",
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Send text to qwen3:14b, parse the structured output, and
    add all entities + relationships to the graph.

    Returns (added_entities, added_relationships).
    """
    prompt = EXTRACTION_PROMPT + text.strip()

    raw_json = await call_llm(
        prompt,
        temperature=0.1,
        max_tokens=4096,
        on_progress=on_progress,
    )
    if not raw_json:
        log.warning("Empty response from Ollama during extraction")
        return [], []

    parsed = _parse_json_response(raw_json)
    if not parsed:
        log.warning("Failed to parse extraction JSON from Ollama response")
        return [], []

    added_entities: list[Entity] = []
    added_relationships: list[Relationship] = []

    # ── Add entities ──────────────────────────────────────────
    raw_entities = parsed.get("entities", [])
    for raw in raw_entities:
        try:
            etype = _parse_entity_type(raw.get("type", ""))
            ts = _parse_timestamp(raw.get("timestamp"))
            entity = Entity(
                name=raw["name"],
                entity_type=etype,
                description=raw.get("description", ""),
                source=source_label,
                timestamp=ts,
                confidence=0.9,  # OSINT-extracted, not confirmed
            )
            added = graph.add_entity(entity)
            added_entities.append(added)
        except Exception as exc:
            log.info(f"Skipping malformed entity: {exc}")

    # ── Add relationships ─────────────────────────────────────
    raw_rels = parsed.get("relationships", [])
    for raw in raw_rels:
        try:
            source_entity = graph.get_entity_by_name(raw["source"])
            target_entity = graph.get_entity_by_name(raw["target"])
            if not source_entity or not target_entity:
                log.info(
                    f"Skipping rel: entity not found "
                    f"({raw['source']} → {raw['target']})"
                )
                continue

            rtype = _parse_relation_type(raw.get("relation", "related_to"))
            rel = Relationship(
                source_entity_id=source_entity.id,
                target_entity_id=target_entity.id,
                relation_type=rtype,
                evidence=raw.get("evidence", ""),
                source_signal=source_label,
                confidence=0.85,
            )
            added = graph.add_relationship(rel)
            added_relationships.append(added)
        except Exception as exc:
            log.info(f"Skipping malformed relationship: {exc}")

    graph.increment_signals()
    return added_entities, added_relationships


# ── Ollama Communication ──────────────────────────────────────



# ── JSON Parsing (robust against malformed LLM output) ────────

def _parse_json_response(text: str) -> Optional[dict]:
    """
    Try multiple strategies to extract valid JSON from LLM output.
    Qwen3 sometimes wraps JSON in markdown code blocks or adds
    commentary.
    """
    if not text:
        return None

    def try_parse(s: str) -> Optional[dict]:
        s = s.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # Fix trailing commas
            cleaned = re.sub(r",\s*([}\]])", r"\1", s)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    # Strategy 1: direct parse
    parsed = try_parse(text)
    if parsed: return parsed

    # Strategy 2: extract from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        parsed = try_parse(match.group(1))
        if parsed: return parsed

    # Strategy 3: find the first { to the last } block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        parsed = try_parse(match.group(1))
        if parsed: return parsed

    log.error(f"Failed to parse any JSON from LLM output. Raw LLM output:\n{text}")
    return None


# ── Type Parsers ──────────────────────────────────────────────

def _parse_entity_type(raw: str) -> EntityType:
    mapping = {
        "actor": EntityType.ACTOR,
        "system": EntityType.SYSTEM,
        "location": EntityType.LOCATION,
        "event": EntityType.EVENT,
        "fact": EntityType.FACT,
    }
    return mapping.get(raw.lower().strip(), EntityType.FACT)


def _parse_relation_type(raw: str) -> RelationType:
    try:
        return RelationType(raw.lower().strip())
    except ValueError:
        return RelationType.RELATED_TO


def _parse_timestamp(raw) -> Optional[datetime]:
    if not raw or raw == "null":
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        # Always return naive datetime to avoid offset-naive vs offset-aware crashes
        return dt.replace(tzinfo=None)
    except (ValueError, TypeError):
        return None
