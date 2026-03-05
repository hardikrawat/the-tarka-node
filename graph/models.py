"""
TARKA — Data Models

Pydantic models for the knowledge graph entities, relationships,
gap hypotheses, theories, and intelligence reports.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────

class EntityType(str, Enum):
    ACTOR = "actor"           # Nations, organizations, people
    SYSTEM = "system"         # Technology platforms, AI models, infra
    LOCATION = "location"     # Physical places, data centers, bases
    EVENT = "event"           # Strikes, outages, launches
    FACT = "fact"             # Verified static facts (hosting, ownership)


class RelationType(str, Enum):
    ATTACKS = "attacks"
    HOSTS = "hosts"
    USES = "uses"
    LOCATED_IN = "located_in"
    RETALIATES = "retaliates"
    CAUSES = "causes"
    TEMPORAL_CORRELATION = "temporal_correlation"
    ALLIED_WITH = "allied_with"
    OPPOSED_TO = "opposed_to"
    OPERATES = "operates"
    OWNED_BY = "owned_by"
    DEVELOPED_BY = "developed_by"
    DEPLOYED_ON = "deployed_on"
    DISRUPTS = "disrupts"
    AFFECTS = "affects"
    RELATED_TO = "related_to"


# ── Core Models ───────────────────────────────────────────────

class Entity(BaseModel):
    """A node in the knowledge graph."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    entity_type: EntityType
    description: str = ""
    source: str = ""              # which OSINT signal introduced this entity
    timestamp: Optional[datetime] = None
    confidence: float = 1.0       # 0.0 – 1.0
    metadata: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    """An edge in the knowledge graph."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    evidence: str = ""            # text excerpt that supports this edge
    source_signal: str = ""       # which OSINT signal introduced this edge
    confidence: float = 1.0
    is_inferred: bool = False     # True if created by gap-bridger, not raw OSINT
    timestamp: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class GapHypothesis(BaseModel):
    """
    A structural gap identified by the anomaly detector.
    Two entities/clusters are temporally correlated but have no
    connecting path in the graph.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    entity_a_id: str
    entity_b_id: str
    entity_a_name: str
    entity_b_name: str
    reason: str                   # why the system thinks these should be linked
    correlation_score: float      # strength of temporal/contextual correlation
    resolved: bool = False
    bridging_facts: list[str] = Field(default_factory=list)


class Theory(BaseModel):
    """
    A synthesized intelligence theory derived from multi-hop
    graph traversal after gap-bridging.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    summary: str
    detailed_analysis: str
    confidence: float
    path: list[str] = Field(default_factory=list)        # ordered node IDs
    path_labels: list[str] = Field(default_factory=list)  # human-readable path
    evidence_chain: list[str] = Field(default_factory=list)
    gaps_bridged: list[str] = Field(default_factory=list)  # gap hypothesis IDs
    generated_at: datetime = Field(default_factory=datetime.now)


class IntelligenceReport(BaseModel):
    """Full report output from a reasoning cycle."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    theories: list[Theory] = Field(default_factory=list)
    gaps_found: int = 0
    gaps_resolved: int = 0
    nodes_total: int = 0
    edges_total: int = 0
    signals_processed: int = 0
    generated_at: datetime = Field(default_factory=datetime.now)


class OSINTSignal(BaseModel):
    """A raw OSINT input — either manual or API-submitted."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str
    source: str = "manual"        # "manual", "api", "gdelt"
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)
