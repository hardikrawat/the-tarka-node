"""
TARKA — REST API Routes

FastAPI endpoints exposed for external OSINT tools to submit
data, query the graph, and trigger analysis.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.anomaly_detector import detect_anomalies
from agents.gap_bridger import bridge_gaps
from agents.theory_generator import generate_theories
from graph.knowledge_graph import IntelGraph
from ingestion.gdelt_source import fetch_gdelt_events
from ingestion.manual_feed import process_batch_feed, process_manual_feed

log = logging.getLogger("tarka.api")


# ── Request / Response schemas ────────────────────────────────

class FeedRequest(BaseModel):
    text: str
    source: str = "api"

class BatchFeedRequest(BaseModel):
    texts: list[str]
    source: str = "api"

class GDELTRequest(BaseModel):
    keywords: list[str]
    timespan_minutes: int = 60
    max_articles: int = 10
    source_country: Optional[str] = None

class AnalyzeRequest(BaseModel):
    pass  # no body needed, triggers analysis on current graph


# ── Route Factory ─────────────────────────────────────────────

def create_api(graph: IntelGraph) -> FastAPI:
    """Create and return the FastAPI app with all routes wired to the graph."""

    api = FastAPI(
        title="TARKA API",
        description="Agentic Graph-RAG OSINT Intelligence API",
        version="0.1.0",
    )

    # ── Feed Endpoints ────────────────────────────────────────

    @api.post("/api/feed")
    async def feed_osint(req: FeedRequest):
        """Submit a single OSINT text signal."""
        entities, rels = await process_manual_feed(
            text=req.text, graph=graph, source=req.source
        )
        return {
            "status": "processed",
            "entities_added": len(entities),
            "relationships_added": len(rels),
            "graph_nodes": graph.node_count,
            "graph_edges": graph.edge_count,
        }

    @api.post("/api/feed/batch")
    async def feed_batch(req: BatchFeedRequest):
        """Submit multiple OSINT signals at once."""
        result = await process_batch_feed(
            texts=req.texts, graph=graph, source=req.source
        )
        return {"status": "processed", **result}

    @api.post("/api/feed/gdelt")
    async def feed_gdelt(req: GDELTRequest):
        """Fetch from GDELT and process into the graph."""
        articles = await fetch_gdelt_events(
            keywords=req.keywords,
            timespan_minutes=req.timespan_minutes,
            max_articles=req.max_articles,
            source_country=req.source_country,
        )
        if not articles:
            return {"status": "no_results", "articles_found": 0}

        result = await process_batch_feed(
            texts=articles, graph=graph, source="gdelt"
        )
        return {"status": "processed", **result}

    # ── Graph Endpoints ───────────────────────────────────────

    @api.get("/api/graph")
    async def get_graph():
        """Get the full knowledge graph as JSON."""
        return graph.to_dict()

    @api.get("/api/graph/node/{node_id}")
    async def get_node(node_id: str):
        """Get details for a specific entity node."""
        entity = graph.get_entity(node_id)
        if not entity:
            raise HTTPException(404, f"Entity '{node_id}' not found")
        edges = graph.get_entity_edges(node_id)
        return {
            "entity": entity.model_dump(),
            "edges": [e.model_dump() for e in edges],
        }

    # ── Analysis Endpoints ────────────────────────────────────

    @api.post("/api/analyze")
    async def run_analysis():
        """
        Trigger the full reasoning pipeline:
        1. Anomaly detection → find gaps
        2. Gap bridging → autonomous search
        3. Theory generation → multi-hop synthesis
        """
        if graph.node_count < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 entities in the graph to analyze",
            }

        # Step 1: Detect anomalies
        gaps = await detect_anomalies(graph)

        # Step 2: Bridge gaps
        resolved_gaps = await bridge_gaps(gaps, graph)

        # Step 3: Generate theories
        report = await generate_theories(graph, resolved_gaps)

        return {
            "status": "complete",
            "report": report.model_dump(),
        }

    @api.get("/api/theories")
    async def list_theories():
        """
        Quick analysis: detect, bridge, theorize, return theories only.
        """
        if graph.node_count < 2:
            return {"theories": []}

        gaps = await detect_anomalies(graph)
        resolved = await bridge_gaps(gaps, graph)
        report = await generate_theories(graph, resolved)

        return {
            "theories": [t.model_dump() for t in report.theories],
            "count": len(report.theories),
        }

    # ── Status ────────────────────────────────────────────────

    @api.get("/api/status")
    async def status():
        return {
            "status": "online",
            "graph_nodes": graph.node_count,
            "graph_edges": graph.edge_count,
            "signals_processed": graph.signals_processed,
        }

    return api
