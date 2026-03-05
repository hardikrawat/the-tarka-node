"""
TARKA — Main TUI Application

Production-ready Textual terminal interface with:
- ASCII splash logo on startup
- Three-panel layout: Feed | Graph | Reports
- Live metrics row with intelligence gauges
- Pipeline step indicator for analysis progress
- Real-time AI reasoning display
- Always-visible system log panel
- Non-blocking LLM calls via run_worker()
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Footer, Header, Static, TextArea, Button, Label
from textual.worker import Worker, WorkerState

from graph.knowledge_graph import IntelGraph
from tui.widgets import (
    GraphViewer, TheoryPanel, LogPanel, OllamaActivity,
    MetricsRow, PipelineIndicator,
)

# ── ASCII Logo ────────────────────────────────────────────────

def _build_logo() -> str:
    import config
    provider = getattr(config, 'LLM_PROVIDER', 'ollama')
    if provider == "gemini":
        model = getattr(config, 'GEMINI_MODEL', 'gemini-2.5-flash')
        badge = f"🔵 gemini │ {model}"
    else:
        model = getattr(config, 'OLLAMA_MODEL', 'qwen3:14b')
        badge = f"🟢 ollama │ {model} │ local"
    return f"[bold #00d4ff]████ TARKA ████[/] [dim #4a6080]── Agentic Graph-RAG OSINT Intelligence ──[/] [dim #2a4060]v0.1.0 │ {badge}[/]"


class TarkaApp(App):
    """TARKA Intelligence Dashboard"""

    TITLE = "TARKA"
    SUB_TITLE = "Agentic Graph-RAG OSINT Intelligence"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+f", "focus_feed", "Feed OSINT", show=True),
        Binding("ctrl+a", "run_analysis", "Analyze", show=True),
        Binding("ctrl+g", "refresh_graph", "Refresh", show=True),
        Binding("ctrl+p", "toggle_provider", "Provider", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(self, graph: IntelGraph, **kwargs):
        super().__init__(**kwargs)
        self._graph = graph
        self._processing = False
        self._log_handler = TUILogHandler(self)

    def compose(self) -> ComposeResult:
        yield Static(_build_logo(), id="header-bar")

        # Live metrics row
        yield MetricsRow(id="metrics-row")

        with Horizontal(id="main-container"):
            # Left Panel — OSINT Feed
            with Vertical(id="feed-panel"):
                yield Static("📡 OSINT FEED", id="feed-title")
                yield TextArea(id="feed-input")
                with Horizontal(id="feed-buttons"):
                    yield Button("⚡ Submit", id="btn-submit", classes="feed-btn")
                    yield Button("📡 GDELT", id="btn-gdelt", classes="feed-btn")
                    yield Button("📋 Copy Logs", id="btn-copy-logs", classes="feed-btn")

            # Center Panel — Graph (scrollable)
            with Vertical(id="graph-panel"):
                yield Static("🔗 KNOWLEDGE GRAPH", id="graph-title")
                with ScrollableContainer(id="graph-scroll"):
                    yield GraphViewer(self._graph, id="graph-view")

            # Right Panel — Reports (scrollable)
            with Vertical(id="report-panel"):
                yield Static("🧠 INTELLIGENCE", id="report-title")
                with ScrollableContainer(id="report-scroll"):
                    yield TheoryPanel(id="report-view")

        # Pipeline step indicator
        yield PipelineIndicator(id="pipeline-indicator")

        with Horizontal(id="activity-container"):
            yield LogPanel(id="log-panel")
            yield OllamaActivity(id="ollama-activity")

        with Horizontal(id="status-bar"):
            yield Static("Ready", id="status-left")
            yield Static("Nodes: 0 │ Edges: 0 │ Signals: 0", id="status-right")

        yield Footer()

    
    def get_progress_callback(self, phase: str):
        try:
            activity = self.query_one("#ollama-activity", OllamaActivity)
            activity.start(phase)
            def callback(tokens: int, elapsed: float, token: str):
                # Use call_from_thread to safely update TUI from worker thread
                self.call_from_thread(activity.update_progress, tokens, elapsed, token)
                if phase.lower().startswith("generating"):
                    try:
                        theory_panel = self.query_one("#report-view", TheoryPanel)
                        self.call_from_thread(theory_panel.append_partial, token)
                    except:
                        pass
            return callback
        except Exception:
            return None

    def stop_progress(self):
        try:
            self.query_one("#ollama-activity", OllamaActivity).stop()
        except:
            pass

    def on_mount(self) -> None:
        """Set up logging to capture agent output in the TUI."""
        # Configure root logger to send to TUI
        root = logging.getLogger("tarka")
        root.setLevel(logging.INFO)
        root.addHandler(self._log_handler)
        self._update_status_bar()
        self.log_message("TARKA initialized — ready for OSINT input", "success")

    def log_message(self, msg: str, level: str = "info") -> None:
        try:
            log_panel = self.query_one("#log-panel", LogPanel)
            log_panel.add_log(msg, level)
        except Exception:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-submit":
            self._launch_submit_feed()
        elif event.button.id == "btn-gdelt":
            self._launch_fetch_gdelt()
        elif event.button.id == "btn-copy-logs":
            self.action_copy_logs()

    # ── Actions ───────────────────────────────────────────────

    def action_copy_logs(self) -> None:
        try:
            log_panel = self.query_one("#log-panel", LogPanel)
            text_to_copy = "\n".join(log_panel._lines)
            
            import subprocess
            from sys import platform
            
            if platform == "darwin":
                p = subprocess.Popen("pbcopy", env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
                p.communicate(text_to_copy.encode("utf-8"))
            elif platform.startswith("win"):
                p = subprocess.Popen("clip", stdin=subprocess.PIPE)
                p.communicate(text_to_copy.encode("utf-16le"))
            else:
                p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
                p.communicate(text_to_copy.encode("utf-8"))
                
            self.log_message("✓ Logs copied to clipboard", "success")
        except Exception as exc:
            self.log_message(f"✗ Failed to copy logs: {exc}", "error")

    def action_focus_feed(self) -> None:
        try:
            feed = self.query_one("#feed-input", TextArea)
            feed.focus()
        except Exception:
            pass

    async def action_run_analysis(self) -> None:
        self._launch_analysis()

    def action_refresh_graph(self) -> None:
        self._refresh_graph_view()

    def action_toggle_provider(self) -> None:
        """Toggle between Ollama and Gemini LLM providers."""
        import config
        if self._processing:
            self.log_message("⏳ Cannot switch provider during processing", "warning")
            return
        if config.LLM_PROVIDER == "ollama":
            if not getattr(config, 'GEMINI_API_KEY', ''):
                self.log_message("✗ No GEMINI_API_KEY set. Add it to .env or use --gemini-key", "error")
                return
            config.LLM_PROVIDER = "gemini"
            self.log_message(f"🔵 Switched to Gemini Cloud ({config.GEMINI_MODEL})", "success")
        else:
            config.LLM_PROVIDER = "ollama"
            self.log_message(f"🟢 Switched to Ollama Local ({config.OLLAMA_MODEL})", "success")
        # Update header
        try:
            self.query_one("#header-bar", Static).update(_build_logo())
        except:
            pass
        self._update_provider_style()

    def _update_provider_style(self) -> None:
        import config
        if config.LLM_PROVIDER == "gemini":
            self.add_class("gemini-mode")
            self.remove_class("ollama-mode")
        else:
            self.add_class("ollama-mode")
            self.remove_class("gemini-mode")

    # ── Worker Launchers (non-blocking) ───────────────────────
    # These kick off background workers so the TUI stays interactive

    def _launch_submit_feed(self) -> None:
        if self._processing:
            self.log_message("⏳ Already processing, please wait...", "warning")
            return
        try:
            feed_input = self.query_one("#feed-input", TextArea)
            text = feed_input.text.strip()
        except Exception:
            return
        if not text:
            self.log_message("Empty input — paste OSINT text first", "warning")
            return
        self._processing = True
        self._update_app_style()
        self.run_worker(self._do_submit_feed(text), exclusive=True, thread=True)

    def _launch_fetch_gdelt(self) -> None:
        if self._processing:
            self.log_message("⏳ Already processing, please wait...", "warning")
            return
        try:
            feed_input = self.query_one("#feed-input", TextArea)
            keywords_text = feed_input.text.strip()
        except Exception:
            return
        if not keywords_text:
            self.log_message("Enter keywords in feed box, then press GDELT", "warning")
            return
        self._processing = True
        self._update_app_style()
        self.run_worker(self._do_fetch_gdelt(keywords_text), exclusive=True, thread=True)

    def _launch_analysis(self) -> None:
        if self._processing:
            self.log_message("⏳ Already processing, please wait...", "warning")
            return
        if self._graph.node_count < 2:
            self.log_message("Need at least 2 entities — feed more OSINT data first", "warning")
            return
        self._processing = True
        self._update_app_style()
        self.run_worker(self._do_full_analysis(), exclusive=True, thread=True)

    def _update_app_style(self) -> None:
        """Update app styles based on current state."""
        if self._processing:
            self.add_class("processing-mode")
        else:
            self.remove_class("processing-mode")

    # ── Worker Coroutines (run in background thread) ──────────

    async def _do_submit_feed(self, text: str) -> None:
        self.call_from_thread(self._set_status, "Processing OSINT signal...")
        self.call_from_thread(self.log_message, f"Processing signal ({len(text)} chars)...")

        try:
            from ingestion.manual_feed import process_manual_feed
            entities, rels = await process_manual_feed(
                text=text, graph=self._graph, source="manual",
                on_progress=self.get_progress_callback("Extracting entities...")
            )
            self.call_from_thread(
                self.log_message,
                f"✓ Extracted {len(entities)} entities, {len(rels)} relationships",
                "success",
            )
            self.call_from_thread(self._refresh_graph_view)
            self.call_from_thread(self._update_status_bar)
            # Clear feed input
            try:
                feed_input = self.query_one("#feed-input", TextArea)
                self.call_from_thread(feed_input.clear)
            except:
                pass
        except Exception as exc:
            self.call_from_thread(self.log_message, f"✗ Feed processing failed: {exc}", "error")
        finally:
            self._processing = False
            self.call_from_thread(self.stop_progress)
            self.call_from_thread(self._set_status, "Ready")
            self.call_from_thread(self._update_app_style)

    async def _do_fetch_gdelt(self, keywords_text: str) -> None:
        keywords = [k.strip() for k in keywords_text.split(",")]
        self.call_from_thread(self._set_status, f"Fetching GDELT: {', '.join(keywords)}...")
        self.call_from_thread(self.log_message, f"Querying GDELT for: {keywords}")

        try:
            from ingestion.gdelt_source import fetch_gdelt_events
            from ingestion.manual_feed import process_manual_feed

            articles = await fetch_gdelt_events(keywords=keywords)
            self.call_from_thread(self.log_message, f"Fetched {len(articles)} articles from GDELT")

            for i, article in enumerate(articles):
                self.call_from_thread(self.log_message, f"Processing article {i + 1}/{len(articles)}...")
                await process_manual_feed(
                    text=article, graph=self._graph, source="gdelt",
                    on_progress=self.get_progress_callback(f"Article {i+1}/{len(articles)}")
                )
                self.call_from_thread(self.stop_progress)

            self.call_from_thread(self.log_message, "✓ GDELT ingestion complete", "success")
            try:
                feed_input = self.query_one("#feed-input", TextArea)
                self.call_from_thread(feed_input.clear)
            except:
                pass
            self.call_from_thread(self._refresh_graph_view)
            self.call_from_thread(self._update_status_bar)
        except Exception as exc:
            self.call_from_thread(self.log_message, f"✗ GDELT fetch failed: {exc}", "error")
        finally:
            self._processing = False
            self.call_from_thread(self.stop_progress)
            self.call_from_thread(self._set_status, "Ready")

    async def _do_full_analysis(self) -> None:
        self.call_from_thread(self._set_status, "🔍 Running analysis pipeline...")
        self.call_from_thread(self.log_message, "═══ ANALYSIS PIPELINE STARTED ═══", "info")

        # Get the pipeline indicator
        try:
            pipeline = self.query_one("#pipeline-indicator", PipelineIndicator)
        except Exception:
            pipeline = None

        try:
            # Step 1: Anomaly detection
            if pipeline:
                self.call_from_thread(pipeline.set_step, 0)
            self.call_from_thread(self.log_message, "Step 1/3: Detecting anomalies & structural gaps...")
            self.call_from_thread(self._set_status, "Step 1/3: Anomaly detection...")
            try:
                activity = self.query_one('#ollama-activity', OllamaActivity)
                self.call_from_thread(activity.start, 'Detecting anomalies...')
            except:
                pass
            from agents.anomaly_detector import detect_anomalies
            gaps = await detect_anomalies(self._graph)
            self.call_from_thread(self.stop_progress)
            self.call_from_thread(
                self.log_message,
                f"Found {len(gaps)} structural gap(s)",
                "success" if gaps else "info",
            )

            # Step 2: Gap bridging (always run — structural/semantic gaps may exist)
            if pipeline:
                self.call_from_thread(pipeline.set_step, 1)
            self.call_from_thread(self.log_message, "Step 2/3: Bridging gaps (LLM + web search)...")
            self.call_from_thread(self._set_status, "Step 2/3: Bridging gaps...")
            if gaps:
                from agents.gap_bridger import bridge_gaps
                resolved = await bridge_gaps(gaps, self._graph, on_progress=self.get_progress_callback("Bridging gaps..."))
                self.call_from_thread(self.stop_progress)
                resolved_count = sum(1 for g in resolved if g.resolved)
                self.call_from_thread(
                    self.log_message,
                    f"Bridged {resolved_count}/{len(gaps)} gaps",
                    "success" if resolved_count > 0 else "warning",
                )
                self.call_from_thread(self._refresh_graph_view)
            else:
                resolved = []
                self.call_from_thread(self.log_message, "No gaps to bridge — all entities well-connected", "info")

            # Step 3: Theory generation
            if pipeline:
                self.call_from_thread(pipeline.set_step, 2)
            self.call_from_thread(self.log_message, "Step 3/3: Generating intelligence theories...")
            self.call_from_thread(self._set_status, "Step 3/3: Theory generation...")
            try:
                theory_panel = self.query_one("#report-view", TheoryPanel)
                self.call_from_thread(theory_panel.clear_partial)
            except:
                pass
            from agents.theory_generator import generate_theories
            report = await generate_theories(self._graph, resolved, on_progress=self.get_progress_callback("Generating theories..."))
            self.call_from_thread(self.stop_progress)
            try:
                theory_panel = self.query_one("#report-view", TheoryPanel)
                self.call_from_thread(theory_panel.clear_partial)
            except:
                pass

            # Mark pipeline complete
            if pipeline:
                self.call_from_thread(pipeline.complete)

            # Display results
            try:
                theory_panel = self.query_one("#report-view", TheoryPanel)
                self.call_from_thread(theory_panel.set_report, report)
            except Exception:
                pass

            # Export theories to .txt file
            if report.theories:
                theory_path = self._export_theories(report)
                if theory_path:
                    self.call_from_thread(self.log_message, f"📄 Theories saved to: {theory_path}", "success")

            self.call_from_thread(
                self.log_message,
                f"═══ ANALYSIS COMPLETE: {len(report.theories)} theor"
                f"{'y' if len(report.theories) == 1 else 'ies'} generated ═══",
                "success",
            )
            self.call_from_thread(self._update_status_bar)

        except Exception as exc:
            self.call_from_thread(self.log_message, f"✗ Analysis failed: {exc}", "error")
        finally:
            self._processing = False
            self.call_from_thread(self.stop_progress)
            if pipeline:
                self.call_from_thread(pipeline.reset)
            self.call_from_thread(self._set_status, "Ready")

    # ── Theory Export ─────────────────────────────────────────

    def _export_theories(self, report) -> str | None:
        """Export theories to a timestamped .txt file and return the path."""
        try:
            from datetime import datetime
            from pathlib import Path

            output_dir = Path("output/theories")
            output_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"theory_{ts}.txt"

            lines = []
            lines.append("=" * 60)
            lines.append("  TARKA — Intelligence Theory Report")
            lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  Graph: {self._graph.node_count} nodes, {self._graph.edge_count} edges")
            lines.append(f"  Gaps found: {report.gaps_found} | Resolved: {report.gaps_resolved}")
            lines.append("=" * 60)
            lines.append("")

            for i, theory in enumerate(report.theories):
                conf_pct = int(theory.confidence * 100)
                lines.append(f"{'─' * 60}")
                lines.append(f"  THEORY {i + 1}: {theory.title}")
                lines.append(f"  Confidence: {conf_pct}%")
                lines.append(f"{'─' * 60}")
                lines.append("")

                lines.append("  SUMMARY:")
                lines.append(f"  {theory.summary}")
                lines.append("")

                if theory.path_labels:
                    lines.append("  CAUSAL CHAIN:")
                    for label in theory.path_labels:
                        lines.append(f"    → {label}")
                    lines.append("")

                if theory.detailed_analysis:
                    lines.append("  DETAILED ANALYSIS:")
                    lines.append(f"  {theory.detailed_analysis}")
                    lines.append("")

                if theory.evidence_chain:
                    lines.append("  EVIDENCE:")
                    for ev in theory.evidence_chain:
                        lines.append(f"    • {ev}")
                    lines.append("")

                lines.append("")

            lines.append("=" * 60)
            lines.append("  END OF REPORT")
            lines.append("=" * 60)

            filepath.write_text("\n".join(lines), encoding="utf-8")
            return str(filepath.resolve())

        except Exception as exc:
            self.call_from_thread(self.log_message, f"✗ Failed to export theories: {exc}", "error")
            return None

    # ── UI Helpers ────────────────────────────────────────────

    def _refresh_graph_view(self) -> None:
        try:
            gv = self.query_one("#graph-view", GraphViewer)
            gv.refresh_graph()
        except Exception:
            pass

    def _update_status_bar(self) -> None:
        try:
            import config
            provider_badge = "🔵 Gemini" if config.LLM_PROVIDER == "gemini" else "🟢 Ollama"
            status_right = self.query_one("#status-right", Static)
            status_right.update(
                f"{provider_badge} │ "
                f"Nodes: {self._graph.node_count} │ "
                f"Edges: {self._graph.edge_count} │ "
                f"Signals: {self._graph.signals_processed}"
            )
        except Exception:
            pass

    def _set_status(self, msg: str) -> None:
        try:
            status_left = self.query_one("#status-left", Static)
            status_left.update(msg)
        except Exception:
            pass


class TUILogHandler(logging.Handler):
    """Bridges Python logging into the TUI log panel."""

    def __init__(self, app: TarkaApp):
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        level_map = {
            "DEBUG": "info",
            "INFO": "info",
            "WARNING": "warning",
            "ERROR": "error",
            "CRITICAL": "error",
        }
        level = level_map.get(record.levelname, "info")
        try:
            self._app.log_message(record.getMessage(), level)
        except Exception:
            pass
