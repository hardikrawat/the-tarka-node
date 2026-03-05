"""
TARKA — Knowledge Graph Engine

NetworkX-backed directed graph that stores OSINT entities and
relationships with confidence scoring, temporal correlation
detection, disconnected-cluster analysis, and multi-hop path
finding.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx

from graph.models import (
    Entity,
    EntityType,
    GapHypothesis,
    Relationship,
    RelationType,
)


class IntelGraph:
    """Core intelligence knowledge graph."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}
        self._signals_count: int = 0

    # ── Properties ────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def signals_processed(self) -> int:
        return self._signals_count

    def increment_signals(self) -> None:
        self._signals_count += 1

    # ── Entity (Node) Operations ──────────────────────────────

    def add_entity(self, entity: Entity) -> Entity:
        """Add an entity node to the graph. Returns the (possibly deduplicated) entity."""
        existing = self._find_duplicate_entity(entity.name, entity.entity_type)
        if existing:
            # Merge: keep higher confidence, append metadata
            if entity.confidence > existing.confidence:
                existing.confidence = entity.confidence
            if entity.description and not existing.description:
                existing.description = entity.description
            existing.metadata.update(entity.metadata)
            self._graph.nodes[existing.id].update(self._entity_attrs(existing))
            return existing

        self._entities[entity.id] = entity
        self._graph.add_node(entity.id, **self._entity_attrs(entity))
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        name_lower = name.lower().strip()
        for e in self._entities.values():
            if e.name.lower().strip() == name_lower:
                return e
        return None

    def get_all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def remove_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        if self._graph.has_node(entity_id):
            self._graph.remove_node(entity_id)

    # ── Relationship (Edge) Operations ────────────────────────

    def add_relationship(self, rel: Relationship) -> Relationship:
        """Add a directed edge between two entities."""
        if rel.source_entity_id not in self._entities:
            raise ValueError(f"Source entity {rel.source_entity_id} not in graph")
        if rel.target_entity_id not in self._entities:
            raise ValueError(f"Target entity {rel.target_entity_id} not in graph")

        # Check for duplicate edge
        existing = self._find_duplicate_relationship(
            rel.source_entity_id, rel.target_entity_id, rel.relation_type
        )
        if existing:
            if rel.confidence > existing.confidence:
                existing.confidence = rel.confidence
            if rel.evidence and existing.evidence:
                existing.evidence += f" | {rel.evidence}"
            elif rel.evidence:
                existing.evidence = rel.evidence
            self._graph[existing.source_entity_id][existing.target_entity_id].update(
                self._rel_attrs(existing)
            )
            return existing

        self._relationships[rel.id] = rel
        self._graph.add_edge(
            rel.source_entity_id,
            rel.target_entity_id,
            **self._rel_attrs(rel),
        )
        return rel

    def get_relationship(self, rel_id: str) -> Optional[Relationship]:
        return self._relationships.get(rel_id)

    def get_all_relationships(self) -> list[Relationship]:
        return list(self._relationships.values())

    def get_entity_edges(self, entity_id: str) -> list[Relationship]:
        """Get all relationships involving an entity (in or out)."""
        result = []
        for r in self._relationships.values():
            if r.source_entity_id == entity_id or r.target_entity_id == entity_id:
                result.append(r)
        return result

    # ── Path Finding ──────────────────────────────────────────

    def find_all_paths(
        self, source_id: str, target_id: str, max_hops: int = 6
    ) -> list[list[str]]:
        """Find all simple paths between two entities up to max_hops."""
        if source_id not in self._graph or target_id not in self._graph:
            return []
        try:
            paths = list(
                nx.all_simple_paths(
                    self._graph, source_id, target_id, cutoff=max_hops
                )
            )
            # Also check undirected paths (relationships may not be perfectly directed)
            undirected = self._graph.to_undirected()
            undirected_paths = list(
                nx.all_simple_paths(undirected, source_id, target_id, cutoff=max_hops)
            )
            # Merge and deduplicate
            all_paths = paths + undirected_paths
            seen = set()
            unique = []
            for p in all_paths:
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            return unique
        except nx.NetworkXError:
            return []

    def has_path(self, source_id: str, target_id: str) -> bool:
        """Check if any path exists between two entities (undirected)."""
        undirected = self._graph.to_undirected()
        try:
            return nx.has_path(undirected, source_id, target_id)
        except nx.NodeNotFound:
            return False

    def score_path(self, path: list[str]) -> float:
        """
        Score a path by multiplying edge confidences.
        Longer paths get a slight penalty.
        """
        if len(path) < 2:
            return 0.0
        score = 1.0
        for i in range(len(path) - 1):
            edge_data = self._graph.get_edge_data(path[i], path[i + 1])
            if edge_data is None:
                # Try reverse direction
                edge_data = self._graph.get_edge_data(path[i + 1], path[i])
            if edge_data:
                score *= edge_data.get("confidence", 0.5)
            else:
                score *= 0.1  # penalty for unlinked hop
        # Hop penalty: longer paths are less certain
        hop_penalty = 1.0 / (1.0 + 0.1 * (len(path) - 2))
        return score * hop_penalty

    # ── Graph Analysis Utilities ────────────────────────────────

    def get_leaf_entities(self) -> list[Entity]:
        """Return entities with degree ≤ 1 (only one edge, or isolated)."""
        leaves: list[Entity] = []
        for entity_id, entity in self._entities.items():
            degree = self._graph.degree(entity_id)  # in + out for DiGraph
            if degree <= 1:
                leaves.append(entity)
        return leaves

    def get_most_central_entity(self) -> Optional[Entity]:
        """Return the entity with the highest combined degree (most connections)."""
        if not self._entities:
            return None
        best_id = max(self._entities.keys(), key=lambda nid: self._graph.degree(nid))
        return self._entities.get(best_id)

    def get_entity_neighborhood(
        self, entity_id: str, hops: int = 1
    ) -> list[Entity]:
        """Return all entities within N hops of a given entity (undirected)."""
        if entity_id not in self._graph:
            return []
        undirected = self._graph.to_undirected()
        visited: set[str] = {entity_id}
        frontier: set[str] = {entity_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in undirected.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        # Return entities (exclude the starting entity itself)
        return [
            self._entities[nid]
            for nid in visited
            if nid != entity_id and nid in self._entities
        ]

    # ── Cluster & Anomaly Analysis ────────────────────────────

    def detect_disconnected_clusters(self) -> list[set[str]]:
        """
        Find weakly-connected components.
        Each component is a set of node IDs.
        """
        undirected = self._graph.to_undirected()
        return [set(c) for c in nx.connected_components(undirected)]

    def find_temporal_correlations(
        self, window_hours: float = 24.0
    ) -> list[tuple[Entity, Entity, float]]:
        """
        Find pairs of entities whose timestamps fall within the
        given window.  Returns (entity_a, entity_b, hours_apart).
        """
        timed_entities = [
            e for e in self._entities.values() if e.timestamp is not None
        ]
        correlations: list[tuple[Entity, Entity, float]] = []
        window = timedelta(hours=window_hours)

        for a, b in itertools.combinations(timed_entities, 2):
            assert a.timestamp and b.timestamp  # guarded above
            try:
                ts_a = a.timestamp.replace(tzinfo=None) if a.timestamp.tzinfo else a.timestamp
                ts_b = b.timestamp.replace(tzinfo=None) if b.timestamp.tzinfo else b.timestamp
                delta = abs(ts_a - ts_b)
            except Exception:
                continue
            if delta <= window:
                hours_apart = delta.total_seconds() / 3600.0
                correlations.append((a, b, hours_apart))

        correlations.sort(key=lambda x: x[2])
        return correlations

    def detect_gaps(self, window_hours: float = 24.0) -> list[GapHypothesis]:
        """
        Core anomaly detection:
        Find temporally-correlated entity pairs that belong to
        different disconnected clusters (i.e., no graph path
        between them).
        """
        correlations = self.find_temporal_correlations(window_hours)
        clusters = self.detect_disconnected_clusters()
        cluster_map: dict[str, int] = {}
        for idx, cluster in enumerate(clusters):
            for node_id in cluster:
                cluster_map[node_id] = idx

        gaps: list[GapHypothesis] = []
        for entity_a, entity_b, hours_apart in correlations:
            cluster_a = cluster_map.get(entity_a.id, -1)
            cluster_b = cluster_map.get(entity_b.id, -1)

            if cluster_a != cluster_b and cluster_a >= 0 and cluster_b >= 0:
                # These are correlated in time but disconnected in the graph
                correlation_score = max(0.0, 1.0 - (hours_apart / window_hours))
                gap = GapHypothesis(
                    entity_a_id=entity_a.id,
                    entity_b_id=entity_b.id,
                    entity_a_name=entity_a.name,
                    entity_b_name=entity_b.name,
                    reason=(
                        f"'{entity_a.name}' and '{entity_b.name}' occurred within "
                        f"{hours_apart:.1f}h of each other but have no connecting "
                        f"path in the knowledge graph."
                    ),
                    correlation_score=correlation_score,
                )
                gaps.append(gap)

        gaps.sort(key=lambda g: g.correlation_score, reverse=True)
        return gaps

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Export graph as a JSON-serializable dict for API / TUI."""
        nodes = []
        for e in self._entities.values():
            nodes.append({
                "id": e.id,
                "name": e.name,
                "type": e.entity_type.value,
                "description": e.description,
                "confidence": e.confidence,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "source": e.source,
            })
        edges = []
        for r in self._relationships.values():
            src = self._entities.get(r.source_entity_id)
            tgt = self._entities.get(r.target_entity_id)
            edges.append({
                "id": r.id,
                "source": r.source_entity_id,
                "target": r.target_entity_id,
                "source_name": src.name if src else "?",
                "target_name": tgt.name if tgt else "?",
                "relation": r.relation_type.value,
                "evidence": r.evidence,
                "confidence": r.confidence,
                "is_inferred": r.is_inferred,
            })
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }

    def to_ascii(self) -> str:
        """Render graph as colored ASCII for the TUI."""
        if not self._entities:
            return "  (empty graph)"

        type_icons = {
            EntityType.ACTOR: "👤",
            EntityType.SYSTEM: "⚙️",
            EntityType.LOCATION: "📍",
            EntityType.EVENT: "⚡",
            EntityType.FACT: "📌",
        }

        lines: list[str] = []
        lines.append("╔══ KNOWLEDGE GRAPH ══════════════════╗")
        lines.append(f"║  Nodes: {self.node_count}  │  Edges: {self.edge_count}       ║")
        lines.append("╠═════════════════════════════════════╣")

        # Group entities by type
        by_type: dict[EntityType, list[Entity]] = defaultdict(list)
        for e in self._entities.values():
            by_type[e.entity_type].append(e)

        for etype in EntityType:
            entities = by_type.get(etype, [])
            if not entities:
                continue
            icon = type_icons.get(etype, "•")
            lines.append(f"║ {icon} {etype.value.upper()}S:")
            for e in entities:
                edges_out = [
                    r for r in self._relationships.values()
                    if r.source_entity_id == e.id
                ]
                conf_str = f"[{e.confidence:.0%}]"
                lines.append(f"║   ├─ {e.name} {conf_str}")
                for edge in edges_out:
                    target = self._entities.get(edge.target_entity_id)
                    tname = target.name if target else "?"
                    rel_label = edge.relation_type.value.replace("_", " ")
                    inferred = " (inferred)" if edge.is_inferred else ""
                    lines.append(
                        f"║   │  └→ {rel_label} → {tname}{inferred}"
                    )

        lines.append("╚═════════════════════════════════════╝")
        return "\n".join(lines)

    # ── Internal Helpers ──────────────────────────────────────

    def _find_duplicate_entity(
        self, name: str, entity_type: EntityType
    ) -> Optional[Entity]:
        name_lower = name.lower().strip()
        for e in self._entities.values():
            if (
                e.name.lower().strip() == name_lower
                and e.entity_type == entity_type
            ):
                return e
        return None

    def _find_duplicate_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationType,
    ) -> Optional[Relationship]:
        for r in self._relationships.values():
            if (
                r.source_entity_id == source_id
                and r.target_entity_id == target_id
                and r.relation_type == rel_type
            ):
                return r
        return None

    @staticmethod
    def _entity_attrs(entity: Entity) -> dict:
        return {
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "confidence": entity.confidence,
            "description": entity.description,
        }

    @staticmethod
    def _rel_attrs(rel: Relationship) -> dict:
        return {
            "rel_id": rel.id,
            "relation_type": rel.relation_type.value,
            "confidence": rel.confidence,
            "is_inferred": rel.is_inferred,
            "evidence": rel.evidence,
        }
