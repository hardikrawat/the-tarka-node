"""
TARKA — Theory Generator Agent

Traverses the knowledge graph via multi-hop BFS/DFS, scores
paths by confidence, and synthesizes intelligence theories
using Qwen3.
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

from graph.models import (
    GapHypothesis,
    IntelligenceReport,
    Theory,
)

log = logging.getLogger("tarka.theory")

# ── Theory Synthesis Prompt ───────────────────────────────────

THEORY_PROMPT = """
You are a senior intelligence analyst. Based on the following connected intelligence graph path, synthesize a coherent theory explaining the causal chain.

GRAPH PATH (each hop = one connection):
{path_description}

GAPS THAT WERE AUTONOMOUSLY BRIDGED:
{bridged_gaps}

ALL ENTITIES IN GRAPH:
{entities_summary}

Instructions:
- Explain the complete causal chain clearly
- Distinguish between confirmed facts and inferred connections
- Assign a confidence level (0.0-1.0) to the overall theory
- Note where the theory is strongest/weakest
- Be specific about the intelligence implications

Return ONLY a valid JSON object:
{{
  "title": "short theory title",
  "summary": "2-3 sentence executive summary",
  "detailed_analysis": "full multi-paragraph analysis with evidence citations",
  "confidence": 0.0 to 1.0
}}
"""


async def generate_theories(
    graph: IntelGraph,
    resolved_gaps: list[GapHypothesis],
    on_progress: Optional[ProgressCallback] = None,
) -> IntelligenceReport:
    """
    Full theory generation cycle:
    1. Find all significant multi-hop paths in the graph
    2. Score them by edge confidence
    3. Synthesize intelligence theories via Qwen3
    4. Package into an IntelligenceReport

    Returns an IntelligenceReport with ranked theories.
    """
    theories: list[Theory] = []

    # ── Step 1: Find interesting paths ────────────────────────
    candidate_paths = _find_candidate_paths(graph, resolved_gaps)

    if not candidate_paths:
        # No gap-crossing paths → find diverse general paths
        candidate_paths = _find_diverse_paths(graph)

    log.info(f"Found {len(candidate_paths)} candidate path(s) for theory generation")

    # ── Step 2: Score, rank, and diversify paths ────────────────
    scored_paths = []
    for path in candidate_paths:
        score = graph.score_path(path)
        labels = _path_to_labels(path, graph)
        scored_paths.append((path, labels, score))

    scored_paths.sort(key=lambda x: x[2], reverse=True)

    # Diversify: greedy selection with overlap penalty
    diverse_paths = _diversify_paths(scored_paths, max_paths=5)

    # Ensure peripheral high-importance entities appear in at least one path
    diverse_paths = _ensure_entity_coverage(diverse_paths, graph)

    # ── Step 3: Synthesize theories for diverse paths ──────────
    for path, labels, score in diverse_paths:
        theory = await _synthesize_theory(
            path, labels, score, graph, resolved_gaps, on_progress
        )
        if theory:
            theories.append(theory)

    theories.sort(key=lambda t: t.confidence, reverse=True)

    report = IntelligenceReport(
        theories=theories,
        gaps_found=len(resolved_gaps),
        gaps_resolved=sum(1 for g in resolved_gaps if g.resolved),
        nodes_total=graph.node_count,
        edges_total=graph.edge_count,
        signals_processed=graph.signals_processed,
    )

    log.info(
        f"Generated {len(theories)} theor{'y' if len(theories) == 1 else 'ies'} "
        f"(best confidence: {theories[0].confidence:.2f})" if theories else
        "No theories generated"
    )

    return report


# ── Path Finding ──────────────────────────────────────────────

def _find_candidate_paths(
    graph: IntelGraph,
    gaps: list[GapHypothesis],
) -> list[list[str]]:
    """
    Find paths through the graph that cross the resolved gap
    boundaries.  These are the most interesting paths because
    they connect previously-disconnected intelligence clusters.
    """
    all_paths: list[list[str]] = []

    for gap in gaps:
        if not gap.resolved:
            continue
        paths = graph.find_all_paths(
            gap.entity_a_id, gap.entity_b_id, max_hops=8
        )
        all_paths.extend(paths)
        # Also check reverse
        paths_rev = graph.find_all_paths(
            gap.entity_b_id, gap.entity_a_id, max_hops=8
        )
        all_paths.extend(paths_rev)

    # Deduplicate
    seen = set()
    unique = []
    for p in all_paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def _find_diverse_paths(graph: IntelGraph) -> list[list[str]]:
    """
    Find candidate paths prioritizing diversity over raw length.
    Uses all-pairs BFS but returns a broader set for diversity
    filtering downstream.
    """
    all_entities = graph.get_all_entities()
    if len(all_entities) < 2:
        return []

    all_paths: list[list[str]] = []
    checked = set()
    for a in all_entities:
        for b in all_entities:
            if a.id == b.id:
                continue
            pair_key = frozenset([a.id, b.id])
            if pair_key in checked:
                continue
            checked.add(pair_key)

            paths = graph.find_all_paths(a.id, b.id, max_hops=8)
            for p in paths:
                if len(p) >= 2:
                    all_paths.append(p)

    # Sort by length but keep more candidates for diversity filtering
    all_paths.sort(key=len, reverse=True)
    return all_paths[:30]  # expanded pool for diversity selection


def _diversify_paths(
    scored_paths: list[tuple[list[str], list[str], float]],
    max_paths: int = 5,
) -> list[tuple[list[str], list[str], float]]:
    """
    Greedy maximum-coverage path selection.
    Each new path is penalized by how much it overlaps with
    already-selected paths (Jaccard similarity on node sets).
    """
    if len(scored_paths) <= 1:
        return scored_paths[:max_paths]

    selected: list[tuple[list[str], list[str], float]] = []
    selected_node_sets: list[set[str]] = []

    for path, labels, score in scored_paths:
        if len(selected) >= max_paths:
            break

        path_nodes = set(path)

        if not selected_node_sets:
            # First path is always accepted
            selected.append((path, labels, score))
            selected_node_sets.append(path_nodes)
            continue

        # Jaccard overlap = intersection / union
        max_overlap = 0.0
        for existing_set in selected_node_sets:
            intersection = len(path_nodes & existing_set)
            union = len(path_nodes | existing_set)
            if union > 0:
                jaccard = intersection / union
                max_overlap = max(max_overlap, jaccard)

        # Extremely harsh penalty for any overlap to force true diversity
        # Anything with > 40% overlap is completely rejected
        if max_overlap >= 0.4:
            continue
            
        # For < 40% overlap, still penalize exponentially based on overlap amount
        diversity_score = score * ((1.0 - max_overlap) ** 3)
        
        selected.append((path, labels, diversity_score))
        selected_node_sets.append(path_nodes)

    return selected


def _ensure_entity_coverage(
    paths: list[tuple[list[str], list[str], float]],
    graph: IntelGraph,
) -> list[tuple[list[str], list[str], float]]:
    """
    Check if high-importance entities (actors, locations) are
    missing from all selected paths. If so, force-add a path
    containing that entity.
    """
    from graph.models import EntityType

    # Collect all covered node IDs
    covered_ids: set[str] = set()
    for path, _, _ in paths:
        covered_ids.update(path)

    # High-importance types that should be covered
    important_types = {EntityType.ACTOR, EntityType.LOCATION, EntityType.EVENT}
    uncovered = [
        e for e in graph.get_all_entities()
        if e.entity_type in important_types and e.id not in covered_ids
    ]

    if not uncovered:
        return paths

    log.info(
        f"Entity coverage: {len(uncovered)} important entities missing from paths: "
        f"{[e.name for e in uncovered]}"
    )

    for entity in uncovered:
        # Find a path from this entity to any covered entity
        for covered_id in covered_ids:
            entity_paths = graph.find_all_paths(entity.id, covered_id, max_hops=6)
            if entity_paths:
                best_path = max(entity_paths, key=len)
                labels = _path_to_labels(best_path, graph)
                score = graph.score_path(best_path)
                paths.append((best_path, labels, score))
                covered_ids.update(best_path)
                log.info(f"  + Added coverage path for '{entity.name}'")
                break

    return paths


def _path_to_labels(path: list[str], graph: IntelGraph) -> list[str]:
    """Convert node IDs to human-readable labels with edge info."""
    # Pre-fetch all relationships once (avoid O(n²) re-fetching)
    all_rels = graph.get_all_relationships()
    edge_lookup: dict[tuple[str, str], list] = {}
    for r in all_rels:
        edge_lookup.setdefault((r.source_entity_id, r.target_entity_id), []).append(r)
        edge_lookup.setdefault((r.target_entity_id, r.source_entity_id), []).append(r)

    labels = []
    for i, node_id in enumerate(path):
        entity = graph.get_entity(node_id)
        name = entity.name if entity else node_id

        if i < len(path) - 1:
            next_id = path[i + 1]
            edge_label = "→"
            rels = edge_lookup.get((node_id, next_id), [])
            if rels:
                r = rels[0]
                edge_label = f"--[{r.relation_type.value}]-->"
                if r.is_inferred:
                    edge_label += " (INFERRED)"
            labels.append(f"{name} {edge_label}")
        else:
            labels.append(name)

    return labels


# ── Theory Synthesis ──────────────────────────────────────────

async def _synthesize_theory(
    path: list[str],
    labels: list[str],
    path_score: float,
    graph: IntelGraph,
    gaps: list[GapHypothesis],
    on_progress: Optional[ProgressCallback] = None,
) -> Optional[Theory]:
    """Have Qwen3 synthesize an intelligence theory from a path."""
    path_desc = "\n".join(f"  {i+1}. {label}" for i, label in enumerate(labels))

    bridged_desc = "\n".join(
        f"  - '{g.entity_a_name}' ↔ '{g.entity_b_name}': {'; '.join(g.bridging_facts)}"
        for g in gaps if g.resolved
    ) or "  (none)"

    entities_summary = "\n".join(
        f"  - {e.name} ({e.entity_type.value}): {e.description}"
        for e in graph.get_all_entities()
    )

    prompt = THEORY_PROMPT.format(
        path_description=path_desc,
        bridged_gaps=bridged_desc,
        entities_summary=entities_summary,
    )

    raw = await call_llm(prompt, think=config.OLLAMA_THINK_FOR_THEORY, on_progress=on_progress)
    parsed = _parse_json(raw)

    if not parsed:
        log.warning("Failed to parse theory synthesis response")
        return None

    gap_ids = [g.id for g in gaps if g.resolved]

    return Theory(
        title=parsed.get("title", "Untitled Theory"),
        summary=parsed.get("summary", ""),
        detailed_analysis=parsed.get("detailed_analysis", ""),
        confidence=min(parsed.get("confidence", 0.5), path_score),
        path=path,
        path_labels=labels,
        evidence_chain=[
            f"Path score: {path_score:.3f}",
            f"Bridged gaps: {len(gap_ids)}",
        ],
        gaps_bridged=gap_ids,
    )


# ── Helpers ───────────────────────────────────────────────────



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
