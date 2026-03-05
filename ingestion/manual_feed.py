"""
TARKA — Manual Feed Ingestion

Processes raw OSINT text input from the TUI or API,
runs entity extraction, and adds results to the graph.
"""

from __future__ import annotations

import logging

from agents.entity_extractor import extract_entities_and_relationships
from graph.knowledge_graph import IntelGraph
from graph.models import Entity, OSINTSignal, Relationship
from typing import Optional, Callable
ProgressCallback = Callable[[int, float, str], None]

log = logging.getLogger("tarka.feed")


async def process_manual_feed(
    text: str,
    graph: IntelGraph,
    source: str = "manual",
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Process a single raw OSINT text signal:
    1. Create an OSINTSignal record
    2. Run entity & relationship extraction via Qwen3
    3. Return the extracted entities and relationships

    Args:
        text:   Raw OSINT text (news article, tweet, report, etc.)
        graph:  The knowledge graph to add to
        source: Label for the source ("manual", "api", etc.)

    Returns:
        Tuple of (added_entities, added_relationships)
    """
    if not text.strip():
        log.warning("Empty text submitted, skipping")
        return [], []

    signal = OSINTSignal(text=text, source=source)
    log.info(
        f"Processing OSINT signal [{signal.id}] from '{source}' "
        f"({len(text)} chars)"
    )

    entities, relationships = await extract_entities_and_relationships(
        text=text,
        graph=graph,
        source_label=f"{source}:{signal.id}",
        on_progress=on_progress,
    )

    log.info(
        f"Extracted {len(entities)} entities and {len(relationships)} "
        f"relationships from signal [{signal.id}]"
    )

    return entities, relationships


async def process_batch_feed(
    texts: list[str],
    graph: IntelGraph,
    source: str = "api",
) -> dict:
    """
    Process multiple OSINT signals in sequence.
    Returns a summary dict.
    """
    total_entities = 0
    total_rels = 0

    for i, text in enumerate(texts):
        entities, rels = await process_manual_feed(
            text=text, graph=graph, source=f"{source}_batch_{i}"
        )
        total_entities += len(entities)
        total_rels += len(rels)

    return {
        "signals_processed": len(texts),
        "entities_extracted": total_entities,
        "relationships_extracted": total_rels,
        "graph_nodes": graph.node_count,
        "graph_edges": graph.edge_count,
    }
