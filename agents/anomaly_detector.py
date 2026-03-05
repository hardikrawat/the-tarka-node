"""
TARKA — Anomaly Detector Agent

Detects temporal correlations, structural gaps, semantic gaps,
and anomalies in the knowledge graph that suggest missing
connections between isolated OSINT clusters.

Gap detection modes:
  1. Temporal — correlated timestamps in disconnected clusters
  2. Structural — leaf nodes, weak branches, isolated entities
  3. Semantic — LLM-identified logical gaps between entities
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

import config
from agents.llm_client import call_llm
from graph.knowledge_graph import IntelGraph
from graph.models import GapHypothesis

log = logging.getLogger("tarka.anomaly")


async def detect_anomalies(graph: IntelGraph) -> list[GapHypothesis]:
    """
    Full anomaly detection pass:
    1. Find temporally-correlated entity pairs (original)
    2. Detect weak leaf nodes and uncovered entities (structural)
    3. Ask the LLM what connections are logically missing (semantic)

    Returns gap hypotheses for the gap bridger.
    """
    if graph.node_count < 2:
        return []

    # ── Mode 1: Temporal gaps (original logic) ────────────────
    gaps = graph.detect_gaps(window_hours=config.TEMPORAL_WINDOW_HOURS)
    if gaps:
        log.info(f"Temporal detector found {len(gaps)} gap(s)")
        for g in gaps:
            log.info(
                f"  ⚠ TEMPORAL GAP: '{g.entity_a_name}' ↔ '{g.entity_b_name}' "
                f"(correlation={g.correlation_score:.2f})"
            )

    # Isolated timestamped nodes (original)
    isolated_gaps = _detect_isolated_timestamped_nodes(graph)
    gaps.extend(isolated_gaps)
    if isolated_gaps:
        log.info(f"Found {len(isolated_gaps)} isolated timestamped node gap(s)")

    # ── Mode 2: Structural gaps (NEW) ─────────────────────────
    leaf_gaps = _detect_weak_leaf_nodes(graph)
    gaps.extend(leaf_gaps)
    if leaf_gaps:
        log.info(f"Structural detector found {len(leaf_gaps)} weak-leaf gap(s)")
        for g in leaf_gaps:
            log.info(
                f"  ⚠ LEAF GAP: '{g.entity_a_name}' ↔ '{g.entity_b_name}'"
            )

    # ── Mode 3: Semantic gaps (NEW — LLM-powered) ─────────────
    if getattr(config, 'ENABLE_SEMANTIC_GAP_DETECTION', True):
        semantic_gaps = await _detect_semantic_gaps(graph)
        gaps.extend(semantic_gaps)
        if semantic_gaps:
            log.info(f"Semantic detector found {len(semantic_gaps)} logical gap(s)")
            for g in semantic_gaps:
                log.info(
                    f"  ⚠ SEMANTIC GAP: '{g.entity_a_name}' ↔ '{g.entity_b_name}': {g.reason}"
                )

    if not gaps:
        log.info("No gaps detected across all detectors")
    else:
        log.info(f"Total gaps detected: {len(gaps)}")

    # Deduplicate by entity pair
    seen_pairs: set[frozenset[str]] = set()
    unique_gaps: list[GapHypothesis] = []
    for g in gaps:
        pair = frozenset([g.entity_a_id, g.entity_b_id])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_gaps.append(g)

    return unique_gaps


# ── Mode 1: Temporal (original) ───────────────────────────────

def _detect_isolated_timestamped_nodes(
    graph: IntelGraph,
) -> list[GapHypothesis]:
    """
    Find nodes that have timestamps (i.e., they are events or
    time-sensitive) but have no edges at all.
    """
    extras: list[GapHypothesis] = []
    all_entities = graph.get_all_entities()
    timestamped = [e for e in all_entities if e.timestamp is not None]
    connected = [e for e in timestamped if len(graph.get_entity_edges(e.id)) > 0]

    for isolated in timestamped:
        if graph.get_entity_edges(isolated.id):
            continue
        for other in connected:
            if other.id == isolated.id:
                continue
            if other.timestamp and isolated.timestamp:
                try:
                    ts_a = isolated.timestamp.replace(tzinfo=None) if isolated.timestamp.tzinfo else isolated.timestamp
                    ts_b = other.timestamp.replace(tzinfo=None) if other.timestamp.tzinfo else other.timestamp
                    delta_hours = abs((ts_b - ts_a).total_seconds()) / 3600.0
                except Exception:
                    continue
                if delta_hours <= config.TEMPORAL_WINDOW_HOURS:
                    gap = GapHypothesis(
                        entity_a_id=isolated.id,
                        entity_b_id=other.id,
                        entity_a_name=isolated.name,
                        entity_b_name=other.name,
                        reason=(
                            f"'{isolated.name}' is temporally close to "
                            f"'{other.name}' ({delta_hours:.1f}h apart) "
                            f"but has no graph connections."
                        ),
                        correlation_score=max(
                            0.0,
                            1.0 - (delta_hours / config.TEMPORAL_WINDOW_HOURS),
                        ),
                    )
                    extras.append(gap)
                    break

    return extras


# ── Mode 2: Structural gap detection (NEW) ────────────────────

def _detect_weak_leaf_nodes(graph: IntelGraph) -> list[GapHypothesis]:
    """
    Find leaf entities (degree ≤ 1) that have only a weak
    'related_to' connection. These are under-explored branches
    that the gap bridger should investigate.
    """
    gaps: list[GapHypothesis] = []
    leaves = graph.get_leaf_entities()
    central = graph.get_most_central_entity()

    if not central:
        return []

    for leaf in leaves:
        if leaf.id == central.id:
            continue

        edges = graph.get_entity_edges(leaf.id)

        # Isolated node (0 edges) → connect to central
        if len(edges) == 0:
            gaps.append(GapHypothesis(
                entity_a_id=leaf.id,
                entity_b_id=central.id,
                entity_a_name=leaf.name,
                entity_b_name=central.name,
                reason=(
                    f"'{leaf.name}' ({leaf.entity_type.value}) is completely "
                    f"isolated with no graph connections. It should be "
                    f"investigated for a relationship to '{central.name}'."
                ),
                correlation_score=0.7,
            ))
            continue

        # Leaf with only weak edges → flag for deeper investigation
        weak_only = all(
            e.relation_type.value == "related_to" or e.confidence < 0.6
            for e in edges
        )
        if weak_only:
            # Find the entity it's weakly connected to
            edge = edges[0]
            other_id = (
                edge.target_entity_id
                if edge.source_entity_id == leaf.id
                else edge.source_entity_id
            )
            other = graph.get_entity(other_id)
            if not other:
                continue

            gaps.append(GapHypothesis(
                entity_a_id=leaf.id,
                entity_b_id=other.id,
                entity_a_name=leaf.name,
                entity_b_name=other.name,
                reason=(
                    f"'{leaf.name}' has only a weak/generic connection to "
                    f"'{other.name}' (type: {edge.relation_type.value}, "
                    f"confidence: {edge.confidence:.2f}). This relationship "
                    f"needs deeper investigation."
                ),
                correlation_score=0.6,
            ))

    return gaps


# ── Mode 3: Semantic gap detection (NEW — LLM-powered) ────────

SEMANTIC_GAP_PROMPT = """
You are an intelligence analyst reviewing a knowledge graph for logical gaps and missing connections.

Current graph entities and their connections:
{graph_summary}

Analyze this graph and identify entities that SHOULD logically be connected but currently aren't, or whose connections are suspiciously weak. Think about:
- Corporate structures (CEO → company, company → address, company → products)
- Geographic significance (why is X located at Y?)
- Temporal patterns (what happened between event A and event B?)
- Investigative leads (what would an analyst want to verify?)

Return ONLY a valid JSON array of gap objects:
[
  {{
    "entity_a": "exact entity name from the graph",
    "entity_b": "exact entity name from the graph",
    "reason": "why these should be connected or investigated",
    "importance": 0.0 to 1.0
  }}
]

Only include gaps where BOTH entity names exist in the graph above.
Return at most 5 gaps, prioritized by investigative importance.
"""


async def _detect_semantic_gaps(graph: IntelGraph) -> list[GapHypothesis]:
    """
    Use the LLM to identify logical/semantic gaps that structural
    analysis cannot detect. This is the 'missing link' function.
    """
    all_entities = graph.get_all_entities()
    if len(all_entities) < 3:
        return []

    # Build a concise graph summary for the LLM
    lines: list[str] = []
    for e in all_entities:
        edges = graph.get_entity_edges(e.id)
        edge_strs: list[str] = []
        for edge in edges:
            other_id = (
                edge.target_entity_id
                if edge.source_entity_id == e.id
                else edge.source_entity_id
            )
            other = graph.get_entity(other_id)
            other_name = other.name if other else "?"
            direction = "→" if edge.source_entity_id == e.id else "←"
            edge_strs.append(
                f"{direction} {edge.relation_type.value} {direction} {other_name}"
            )
        connections = "; ".join(edge_strs) if edge_strs else "(no connections)"
        lines.append(
            f"- {e.name} [{e.entity_type.value}]: {e.description}. "
            f"Connections: {connections}"
        )

    graph_summary = "\n".join(lines)
    prompt = SEMANTIC_GAP_PROMPT.format(graph_summary=graph_summary)

    raw = await call_llm(prompt)
    parsed = _parse_gap_json(raw)

    if not parsed:
        log.warning("Semantic gap detector: failed to parse LLM response")
        return []

    gaps: list[GapHypothesis] = []
    for item in parsed:
        name_a = item.get("entity_a", "")
        name_b = item.get("entity_b", "")
        entity_a = graph.get_entity_by_name(name_a)
        entity_b = graph.get_entity_by_name(name_b)

        if not entity_a or not entity_b:
            log.info(
                f"Semantic gap skipped: entity not found "
                f"('{name_a}' → '{name_b}')"
            )
            continue

        if entity_a.id == entity_b.id:
            continue

        gaps.append(GapHypothesis(
            entity_a_id=entity_a.id,
            entity_b_id=entity_b.id,
            entity_a_name=entity_a.name,
            entity_b_name=entity_b.name,
            reason=item.get("reason", "LLM-identified logical gap"),
            correlation_score=min(item.get("importance", 0.5), 1.0),
        ))

    return gaps


def _parse_gap_json(text: str) -> Optional[list[dict]]:
    """Parse a JSON array from LLM output (robust)."""
    if not text:
        return None
    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Extract from code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # Find first [ ... ] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None
