"""
TARKA — Custom TUI Widgets

Production-ready widgets for the intelligence dashboard:
- GraphViewer: ASCII knowledge graph rendering
- TheoryPanel: Intelligence report display
- LogPanel: Always-visible scrollable system log
- OllamaActivity: Real-time AI reasoning display
- MetricsRow: Live intelligence gauges
- PipelineIndicator: Multi-step analysis progress
"""

from __future__ import annotations

import time
from datetime import datetime
from collections import deque

from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from textual.widgets import Static, RichLog
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive

from graph.knowledge_graph import IntelGraph
from graph.models import EntityType, IntelligenceReport, Theory


# ── Color Scheme ──────────────────────────────────────────────

ENTITY_COLORS = {
    EntityType.ACTOR:     "#4ade80",   # green
    EntityType.SYSTEM:    "#60a5fa",   # blue
    EntityType.LOCATION:  "#f97316",   # orange
    EntityType.EVENT:     "#f43f5e",   # red
    EntityType.FACT:      "#a78bfa",   # purple
}

ENTITY_ICONS = {
    EntityType.ACTOR:     "👤",
    EntityType.SYSTEM:    "⚙️ ",
    EntityType.LOCATION:  "📍",
    EntityType.EVENT:     "⚡",
    EntityType.FACT:      "📌",
}


# ══════════════════════════════════════════════════════════════
# MetricsRow — Live Intelligence Gauges
# ══════════════════════════════════════════════════════════════

class MetricsRow(Horizontal):
    """Horizontal bar with live-updating intelligence gauges."""

    _entities_history: deque = deque(maxlen=60)
    _last_entity_count: int = 0
    _start_time: float = 0.0

    def compose(self):
        yield Static(id="metric-nodes", classes="metric-cell")
        yield Static(id="metric-edges", classes="metric-cell")
        yield Static(id="metric-confidence", classes="metric-cell")
        yield Static(id="metric-tps", classes="metric-cell")
        yield Static(id="metric-load", classes="metric-cell")
        yield Static(id="metric-uptime", classes="metric-cell")

    def on_mount(self) -> None:
        self._start_time = time.monotonic()
        self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        """Called every second to update gauges."""
        try:
            app = self.app
            graph = getattr(app, "_graph", None)
            if not graph:
                return

            # Node/Edge counts with sparkline bar
            nc = graph.node_count
            ec = graph.edge_count

            # Track entity growth rate
            self._entities_history.append(nc)
            rate = 0.0
            if len(self._entities_history) >= 2:
                rate = (self._entities_history[-1] - self._entities_history[0]) / max(len(self._entities_history), 1)

            # Average confidence across entities
            entities = graph.get_all_entities()
            avg_conf = sum(e.confidence for e in entities) / max(len(entities), 1) if entities else 0.0

            # Uptime
            uptime = time.monotonic() - self._start_time
            mins = int(uptime) // 60
            secs = int(uptime) % 60

            # Sparkline-style confidence bar
            conf_pct = int(avg_conf * 100)
            bar_len = 8
            filled = int(avg_conf * bar_len)
            conf_bar = "█" * filled + "░" * (bar_len - filled)
            conf_color = "#4ade80" if conf_pct >= 70 else "#ffd93d" if conf_pct >= 40 else "#f43f5e"

            # Get LLM performance from activity widget if active
            tps_str = "0.0"
            tps_color = "#4a6080"
            load_bar = "░░░░░░░░"
            load_color = "#4a6080"
            
            try:
                activity = app.query_one("#ollama-activity", OllamaActivity)
                if activity.running:
                    # Calculate TPS
                    if activity._tokens > 0 and activity._elapsed > 0:
                        tps = activity._tokens / activity._elapsed
                        tps_str = f"{tps:.1f}"
                        tps_color = "#00ffaa"
                    
                    # Inference load visualization (pulsing bar)
                    load_idx = int(time.time() * 4) % 8
                    load_bar = "█" * load_idx + "▒" + "░" * (7 - load_idx)
                    load_color = "#00d4ff"
                else:
                    tps_str = "IDLE"
                    tps_color = "#6680aa"
            except:
                pass

            # Update cells
            self.query_one("#metric-nodes", Static).update(
                Text.from_markup(f"[bold #00d4ff]⬡ NODES[/]  [bold white]{nc}[/]")
            )
            self.query_one("#metric-edges", Static).update(
                Text.from_markup(f"[bold #a78bfa]⬡ EDGES[/]  [bold white]{ec}[/]")
            )
            self.query_one("#metric-confidence", Static).update(
                Text.from_markup(f"[bold {conf_color}]◉ CONF[/]  [{conf_color}]{conf_bar}[/] [white]{conf_pct}%[/]")
            )

            self.query_one("#metric-tps", Static).update(
                Text.from_markup(f"[bold #00ffaa]⚡ TPS[/]  [{tps_color}]{tps_str}[/]")
            )

            self.query_one("#metric-load", Static).update(
                Text.from_markup(f"[bold #00d4ff]⎆ LOAD[/]  [{load_color}]{load_bar}[/]")
            )

            self.query_one("#metric-uptime", Static).update(
                Text.from_markup(f"[bold #ff6b6b]⏱ UP[/]  [white]{mins:02d}:{secs:02d}[/]")
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# PipelineIndicator — Analysis Step Progress
# ══════════════════════════════════════════════════════════════

class PipelineIndicator(Static):
    """Shows the current analysis pipeline step with animated indicators."""

    STEPS = [
        ("🔍", "ANOMALY DETECT", "#ff6b6b"),
        ("🌉", "GAP BRIDGING",   "#ffd93d"),
        ("🧠", "THEORY GEN",     "#4ade80"),
    ]

    _active_step: int = -1
    _spinner_idx: int = 0
    SPINNER = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.12, self._tick, pause=True)

    def _tick(self) -> None:
        self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER)
        self._render_pipeline()

    def set_step(self, step: int) -> None:
        """Set the active pipeline step (0-2). -1 = idle."""
        self._active_step = step
        if step >= 0:
            self._timer.resume()
        else:
            self._timer.pause()
        self._render_pipeline()

    def complete(self) -> None:
        """Mark all steps complete."""
        self._active_step = len(self.STEPS)
        self._timer.pause()
        self._render_pipeline()

    def reset(self) -> None:
        """Reset to idle."""
        self._active_step = -1
        self._timer.pause()
        self._render_pipeline()

    def _render_pipeline(self) -> None:
        t = Text()
        t.append("  ", style="")
        for i, (icon, name, color) in enumerate(self.STEPS):
            if i < self._active_step:
                # Completed
                t.append(f" ✓ {name} ", style=f"bold {color}")
            elif i == self._active_step:
                # Active — animated spinner
                spin = self.SPINNER[self._spinner_idx]
                t.append(f" {spin} {name} ", style=f"bold reverse {color}")
            else:
                # Pending
                t.append(f" ○ {name} ", style="dim #4a6080")
            if i < len(self.STEPS) - 1:
                connector_style = f"bold {color}" if i < self._active_step else "dim #2a4060"
                t.append(" → ", style=connector_style)

        if self._active_step >= len(self.STEPS):
            t.append("  ✅ COMPLETE", style="bold #4ade80")
        elif self._active_step < 0:
            t.append("  │ Press Ctrl+A to analyze", style="dim #4a6080")

        self.update(t)


# ══════════════════════════════════════════════════════════════
# GraphViewer — Knowledge Graph ASCII Renderer
# ══════════════════════════════════════════════════════════════

class GraphViewer(Static):
    """Renders the knowledge graph as a rich ASCII tree view."""

    def __init__(self, graph: IntelGraph, **kwargs) -> None:
        super().__init__(**kwargs)
        self._graph = graph

    def render(self) -> Text:
        if self._graph.node_count == 0:
            t = Text()
            t.append("\n  ┌─────────────────────────────┐\n", style="dim #2a4060")
            t.append("  │                             │\n", style="dim #2a4060")
            t.append("  │   ", style="dim #2a4060")
            t.append("NO INTELLIGENCE DATA", style="bold #4a6080")
            t.append("    │\n", style="dim #2a4060")
            t.append("  │                             │\n", style="dim #2a4060")
            t.append("  │  ", style="dim #2a4060")
            t.append("Feed OSINT signals to", style="#4a6080")
            t.append("   │\n", style="dim #2a4060")
            t.append("  │  ", style="dim #2a4060")
            t.append("populate the graph.", style="#4a6080")
            t.append("     │\n", style="dim #2a4060")
            t.append("  │                             │\n", style="dim #2a4060")
            t.append("  │  ", style="dim #2a4060")
            t.append("Ctrl+F", style="bold #00d4ff")
            t.append(" → Feed input", style="#4a6080")
            t.append("      │\n", style="dim #2a4060")
            t.append("  │                             │\n", style="dim #2a4060")
            t.append("  └─────────────────────────────┘\n", style="dim #2a4060")
            return t

        return self._render_graph()

    def _render_graph(self) -> Text:
        t = Text()

        t.append("╔══════════════════════════════════════╗\n", style="bold #1a3060")
        t.append("║  ", style="bold #1a3060")
        t.append("INTELLIGENCE GRAPH", style="bold #ff6b6b")
        stats = f"  N:{self._graph.node_count} E:{self._graph.edge_count}"
        t.append(stats, style="#6680aa")
        padding = 38 - 20 - len(stats)
        t.append(" " * max(0, padding) + "║\n", style="bold #1a3060")
        t.append("╠══════════════════════════════════════╣\n", style="bold #1a3060")

        # Group entities by type
        from collections import defaultdict
        by_type: dict[EntityType, list] = defaultdict(list)
        for e in self._graph.get_all_entities():
            by_type[e.entity_type].append(e)

        for etype in EntityType:
            entities = by_type.get(etype, [])
            if not entities:
                continue

            icon = ENTITY_ICONS.get(etype, "•")
            color = ENTITY_COLORS.get(etype, "#808080")

            t.append(f"║ {icon} ", style="bold #1a3060")
            t.append(f"{etype.value.upper()}S", style=f"bold {color}")
            t.append("\n")

            for i, entity in enumerate(entities):
                is_last = i == len(entities) - 1
                branch = "└─" if is_last else "├─"
                conf = f" [{entity.confidence:.0%}]"

                t.append(f"║   {branch} ", style="#2a4060")
                t.append(entity.name, style=f"bold {color}")
                t.append(conf, style="#4a6080")
                t.append("\n")

                # Show outgoing edges
                edges = [
                    r for r in self._graph.get_all_relationships()
                    if r.source_entity_id == entity.id
                ]
                for j, edge in enumerate(edges):
                    target = self._graph.get_entity(edge.target_entity_id)
                    tname = target.name if target else "?"
                    tcolor = ENTITY_COLORS.get(target.entity_type, "#808080") if target else "#808080"

                    connector = "   " if is_last else "│  "
                    edge_branch = "└→" if j == len(edges) - 1 else "├→"
                    rel_label = edge.relation_type.value.replace("_", " ")

                    t.append(f"║   {connector} {edge_branch} ", style="#2a4060")
                    t.append(rel_label, style="italic #ffd93d")
                    t.append(" → ", style="#4a6080")
                    t.append(tname, style=f"{tcolor}")
                    if edge.is_inferred:
                        t.append(" ⚡INFERRED", style="bold #ff6b6b")
                    t.append("\n")

        t.append("╚══════════════════════════════════════╝\n", style="bold #1a3060")
        return t

    def refresh_graph(self) -> None:
        self.update(self.render())


# ══════════════════════════════════════════════════════════════
# TheoryPanel — Intelligence Reports
# ══════════════════════════════════════════════════════════════

class TheoryPanel(Static):
    """Renders intelligence theories and reports with real-time streaming."""

    _partial_theory: str = ""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._report: IntelligenceReport | None = None
        self._theories: list[Theory] = []

    def append_partial(self, content: str) -> None:
        """Stream new theory content into the view."""
        self._partial_theory += content
        self.update(self.render())

    def clear_partial(self) -> None:
        self._partial_theory = ""
        self.update(self.render())

    def render(self) -> Text:
        if not self._theories and not self._partial_theory:
            t = Text()
            t.append("\n  ┌─────────────────────────────┐\n", style="dim #2a4060")
            t.append("  │                             │\n", style="dim #2a4060")
            t.append("  │   ", style="dim #2a4060")
            t.append("NO THEORIES YET", style="bold #4a6080")
            t.append("        │\n", style="dim #2a4060")
            t.append("  │                             │\n", style="dim #2a4060")
            t.append("  │  ", style="dim #2a4060")
            t.append("Feed OSINT data, then", style="#4a6080")
            t.append("  │\n", style="dim #2a4060")
            t.append("  │  ", style="dim #2a4060")
            t.append("press ", style="#4a6080")
            t.append("Ctrl+A", style="bold #ffd93d")
            t.append(" to analyze.", style="#4a6080")
            t.append("│\n", style="dim #2a4060")
            t.append("  │                             │\n", style="dim #2a4060")
            t.append("  └─────────────────────────────┘\n", style="dim #2a4060")
            return t

        return self._render_theories()

    def _render_theories(self) -> Text:
        t = Text()

        # Render partial streaming theory if it exists
        if self._partial_theory:
            t.append("╔══ 🧠 AI REASONING ──────────────────╗\n", style="bold #a78bfa")
            # Wrap partial content
            words = self._partial_theory.split()
            line = "║ "
            for word in words:
                if len(line) + len(word) > 38:
                    t.append(line, style="#c0c8d8")
                    t.append("\n")
                    line = "║ "
                line += word + " "
            if line.strip("║ "):
                t.append(line, style="#c0c8d8")
                t.append("\n")
            t.append("╚" + "═" * 38 + "╝\n\n", style="#a78bfa")

        if self._report:
            t.append("┌─ ANALYSIS SUMMARY ─────────────────┐\n", style="#1a3060")
            t.append(f"│ Gaps found:    {self._report.gaps_found}\n", style="#6680aa")
            t.append(f"│ Gaps resolved: {self._report.gaps_resolved}\n", style="#6680aa")
            t.append(f"│ Theories:      {len(self._theories)}\n", style="#6680aa")
            t.append("└────────────────────────────────────┘\n\n", style="#1a3060")

        for i, theory in enumerate(self._theories):
            # Confidence bar
            conf_pct = int(theory.confidence * 100)
            bar_filled = int(theory.confidence * 20)
            bar_empty = 20 - bar_filled
            conf_color = "#4ade80" if conf_pct >= 70 else "#ffd93d" if conf_pct >= 40 else "#f43f5e"

            t.append(f"╔══ THEORY {i + 1} ", style=f"bold {conf_color}")
            t.append(f"[{conf_pct}%] ", style=f"bold {conf_color}")
            t.append("█" * bar_filled, style=conf_color)
            t.append("░" * bar_empty, style="#1a3060")
            t.append(" ══╗\n", style=f"bold {conf_color}")

            t.append("║ ", style=f"{conf_color}")
            t.append(theory.title, style=f"bold white")
            t.append("\n")

            t.append("╠────────────────────────────────────╣\n", style="#1a3060")

            # Summary
            t.append("║ ", style="#1a3060")
            t.append(theory.summary, style="#c0c8d8")
            t.append("\n")

            # Evidence path
            if theory.path_labels:
                t.append("╠── CAUSAL CHAIN ────────────────────╣\n", style="#1a3060")
                for label in theory.path_labels:
                    t.append("║  ", style="#1a3060")
                    t.append(f"→ {label}", style="#60a5fa")
                    t.append("\n")

            # Detailed analysis
            if theory.detailed_analysis:
                t.append("╠── ANALYSIS ────────────────────────╣\n", style="#1a3060")
                # Wrap long text
                words = theory.detailed_analysis.split()
                line = "║ "
                for word in words:
                    if len(line) + len(word) > 38:
                        t.append(line, style="#8899bb")
                        t.append("\n")
                        line = "║ "
                    line += word + " "
                if line.strip("║ "):
                    t.append(line, style="#8899bb")
                    t.append("\n")

            t.append("╚════════════════════════════════════╝\n\n", style=f"{conf_color}")

        return t

    def set_report(self, report: IntelligenceReport) -> None:
        self._report = report
        self._theories = report.theories
        self.update(self.render())

    def clear(self) -> None:
        self._report = None
        self._theories = []
        self.update(self.render())


# ══════════════════════════════════════════════════════════════
# LogPanel — System Activity Log
# ══════════════════════════════════════════════════════════════

class LogPanel(RichLog):
    """Scrollable log output panel with clipboard export support."""

    def __init__(self, **kwargs) -> None:
        super().__init__(markup=True, wrap=True, **kwargs)
        self._lines: list[str] = []

    def add_log(self, message: str, level: str = "info") -> None:
        color_map = {
            "info": "#6680aa",
            "warning": "#ffd93d",
            "error": "#f43f5e",
            "success": "#4ade80",
        }
        color = color_map.get(level, "#6680aa")
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self._lines.append(line)
        if len(self._lines) > 500:
            self._lines = self._lines[-500:]
        from rich.markup import escape
        self.write(f"[{color}][{escape(ts)}] {escape(message)}[/]")


# ══════════════════════════════════════════════════════════════
# OllamaActivity — Real-time AI Reasoning Display
# ══════════════════════════════════════════════════════════════

class OllamaActivity(Vertical):
    """Real-time Ollama activity indicator with thinking-mode support."""

    phase = reactive("")
    running = reactive(False)
    _thinking = False
    _tokens = 0
    _elapsed = 0.0

    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    THINK_FRAMES = [
        "🧠 ·····",
        "🧠 ●····",
        "🧠 ●●···",
        "🧠 ●●●··",
        "🧠 ●●●●·",
        "🧠 ●●●●●",
        "🧠 ·●●●●",
        "🧠 ··●●●",
        "🧠 ···●●",
        "🧠 ····●",
    ]

    def compose(self):
        yield Static(id="ollama-header")
        yield RichLog(id="ollama-stream", highlight=True, wrap=True, markup=True)

    def on_mount(self) -> None:
        self.update_timer = self.set_interval(0.1, self.tick_spinner, pause=True)

    def tick_spinner(self) -> None:
        self.spinner_idx = (self.spinner_idx + 1) % len(self.SPINNER)
        self._update_render()

    def watch_phase(self, _: str, __: str) -> None:
        self._update_render()

    def watch_tokens_text(self, _: str, __: str) -> None:
        self._update_render()

    def watch_elapsed_text(self, _: str, __: str) -> None:
        self._update_render()

    def _update_render(self) -> None:
        header = self.query_one("#ollama-header", Static)
        if not self.running:
            header.update("🧠 AI REASONING (IDLE)")
            return

        t = Text()
        if self._thinking:
            # Thinking mode — animated brain pulse
            frame = self.THINK_FRAMES[self.spinner_idx % len(self.THINK_FRAMES)]
            t.append(f"{frame} ", style="bold #a78bfa")
            t.append("THINKING │ ", style="bold #a78bfa")
            t.append(f"{self.elapsed_text}", style="#ff6b6b")
        else:
            spin = self.SPINNER[self.spinner_idx]
            t.append("🧠 ", style="#00d4ff")
            t.append(f"{spin} ", style="bold #00ffaa")
            t.append(f"{self.phase} │ ", style="bold white")
            t.append(f"{self.tokens_text} │ ", style="#ffd93d")
            t.append(f"{self.elapsed_text}", style="#ff6b6b")
        header.update(t)

    def start(self, phase: str) -> None:
        self.phase = phase
        self._tokens = 0
        self._elapsed = 0.0
        self.tokens_text = "0 tokens"
        self.elapsed_text = "0.0s"
        self.spinner_idx = 0
        self.running = True
        self._thinking = False
        stream = self.query_one("#ollama-stream", RichLog)
        stream.clear()
        stream.write(f"[bold #00d4ff]── {phase.upper()} ──[/]")
        self.update_timer.resume()
        self.add_class("active")
        self._update_render()

    def update_progress(self, tokens: int, elapsed: float, token_text: str = "") -> None:
        self._tokens = tokens
        self._elapsed = elapsed
        self.elapsed_text = f"{elapsed:.1f}s"
        if token_text:
            # Got real content — switch from thinking to streaming
            self._thinking = False
            self.tokens_text = f"{tokens} tokens"
            from rich.markup import escape
            stream = self.query_one("#ollama-stream", RichLog)
            stream.write(escape(token_text))
        elif tokens == 0:
            # Heartbeat during thinking phase — no content yet
            self._thinking = True
            self.tokens_text = "thinking..."

    def stop(self) -> None:
        self.running = False
        self._thinking = False
        self.update_timer.pause()
        self.remove_class("active")
        self._update_render()
        stream = self.query_one("#ollama-stream", RichLog)
        stream.write("\n[bold #00ffaa]── COMPLETE ──[/]")
