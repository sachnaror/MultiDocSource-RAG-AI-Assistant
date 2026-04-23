from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from api_client import APIClient


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.client = APIClient()
        self.selected_file: Path | None = None
        self.current_job_id: str | None = None
        self.chat_history: list[str] = []

        self.setWindowTitle("Multi-Format AI Knowledge Assistant")
        self.resize(1500, 900)
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_job)
        self._check_backend_connectivity()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        splitter = QSplitter()
        root_layout.addWidget(splitter)

        left = self._build_left_panel()
        middle = self._build_middle_panel()
        right = self._build_right_panel()

        splitter.addWidget(left)
        splitter.addWidget(middle)
        splitter.addWidget(right)
        splitter.setSizes([360, 600, 540])

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Ingestion Console")
        title.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        source_box = QGroupBox("Upload Sources")
        source_layout = QFormLayout(source_box)

        self.source_name_input = QLineEdit()
        self.source_name_input.setPlaceholderText("e.g. policy.pdf / sales.xlsx / inventory-api")

        self.source_type_input = QLineEdit("pdf")
        self.source_type_input.setPlaceholderText("pdf or excel")

        self.file_path_label = QLabel("No file selected")
        self.pick_file_btn = QPushButton("Pick PDF/Excel File")
        self.pick_file_btn.clicked.connect(self._pick_file)

        self.ingest_file_btn = QPushButton("Start File Ingestion")
        self.ingest_file_btn.clicked.connect(self._ingest_file)

        self.api_url_input = QLineEdit()
        self.api_url_input.setPlaceholderText("https://api.example.com/data")
        self.ingest_api_btn = QPushButton("Ingest API Source")
        self.ingest_api_btn.clicked.connect(self._ingest_api)

        source_layout.addRow("Source Name", self.source_name_input)
        source_layout.addRow("Source Type", self.source_type_input)
        source_layout.addRow(self.pick_file_btn)
        source_layout.addRow("Selected", self.file_path_label)
        source_layout.addRow(self.ingest_file_btn)
        source_layout.addRow(QLabel("API Endpoint"))
        source_layout.addRow(self.api_url_input)
        source_layout.addRow(self.ingest_api_btn)

        layout.addWidget(source_box)

        job_box = QGroupBox("Job Status")
        job_layout = QVBoxLayout(job_box)
        self.job_status_text = QTextEdit()
        self.job_status_text.setReadOnly(True)
        self.job_status_text.setPlaceholderText("Async ingestion status will appear here...")
        job_layout.addWidget(self.job_status_text)
        layout.addWidget(job_box)

        src_box = QGroupBox("Indexed Sources")
        src_layout = QVBoxLayout(src_box)
        self.sources_list = QListWidget()
        self.refresh_sources_btn = QPushButton("Refresh Sources")
        self.refresh_sources_btn.clicked.connect(self._refresh_sources)
        src_layout.addWidget(self.sources_list)
        src_layout.addWidget(self.refresh_sources_btn)

        layout.addWidget(src_box)
        layout.addStretch(1)
        return panel

    def _build_middle_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("RAG Chat Workspace")
        title.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        query_box = QGroupBox("Ask Questions Across PDF + Excel + API")
        query_layout = QFormLayout(query_box)

        self.top_k_input = QSpinBox()
        self.top_k_input.setRange(1, 10)
        self.top_k_input.setValue(5)

        self.question_input = QPlainTextEdit()
        self.question_input.setPlaceholderText("Ask a factual / analytical / summary question...")
        self.question_input.setFixedHeight(120)

        self.ask_btn = QPushButton("Run Query")
        self.ask_btn.clicked.connect(self._run_query)

        query_layout.addRow("Top K", self.top_k_input)
        query_layout.addRow("Question", self.question_input)
        query_layout.addRow(self.ask_btn)

        layout.addWidget(query_box)

        answer_box = QGroupBox("Answer (Right-Hand Response Panel Output)")
        answer_layout = QVBoxLayout(answer_box)
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("Generated answer will be rendered here")
        answer_layout.addWidget(self.answer_output)

        layout.addWidget(answer_box, stretch=1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Metrics + Debug Dashboard")
        title.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        perf_box = QGroupBox("Retrieval & Performance")
        perf_layout = QGridLayout(perf_box)
        self.metric_topk = QLabel("Top K: -")
        self.metric_scores = QLabel("Scores: -")
        self.metric_avg = QLabel("Avg Similarity: -")
        self.metric_latency = QLabel("Latency: -")
        self.metric_tokens = QLabel("Tokens: -")
        self.metric_confidence = QLabel("Confidence: -")
        perf_layout.addWidget(self.metric_topk, 0, 0)
        perf_layout.addWidget(self.metric_scores, 1, 0)
        perf_layout.addWidget(self.metric_avg, 2, 0)
        perf_layout.addWidget(self.metric_latency, 3, 0)
        perf_layout.addWidget(self.metric_tokens, 4, 0)
        perf_layout.addWidget(self.metric_confidence, 5, 0)
        layout.addWidget(perf_box)

        embed_box = QGroupBox("Embedding Insights")
        embed_layout = QVBoxLayout(embed_box)
        self.embedding_stats = QLabel("Model: -\nVector Dimension: -\nAvg Embed Time: -")
        embed_layout.addWidget(self.embedding_stats)
        layout.addWidget(embed_box)

        src_box = QGroupBox("Source Attribution")
        src_layout = QVBoxLayout(src_box)
        self.source_attribution_output = QTextEdit()
        self.source_attribution_output.setReadOnly(True)
        src_layout.addWidget(self.source_attribution_output)
        layout.addWidget(src_box)

        debug_box = QGroupBox("Debug Panel")
        debug_layout = QVBoxLayout(debug_box)
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        debug_layout.addWidget(self.debug_output)
        layout.addWidget(debug_box, stretch=1)

        return panel

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select source file", "", "Documents (*.pdf *.xlsx *.xls)")
        if path:
            self.selected_file = Path(path)
            self.file_path_label.setText(path)

    def _ingest_file(self) -> None:
        if not self.selected_file:
            QMessageBox.warning(self, "Missing file", "Select a PDF or Excel file first.")
            return

        source_name = self.source_name_input.text().strip() or self.selected_file.name
        source_type = self.source_type_input.text().strip().lower() or "pdf"

        try:
            result = self.client.ingest_file(self.selected_file, source_name, source_type)
            self.current_job_id = result["job_id"]
            self.job_status_text.append(f"[queued] {result}")
            self.timer.start(1500)
        except Exception as exc:
            QMessageBox.critical(self, "Ingestion failed", str(exc))

    def _ingest_api(self) -> None:
        source_name = self.source_name_input.text().strip() or "api-source"
        url = self.api_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Missing API URL", "Enter API endpoint URL.")
            return

        try:
            result = self.client.ingest_api(source_name, url)
            self.current_job_id = result["job_id"]
            self.job_status_text.append(f"[queued] {result}")
            self.timer.start(1500)
        except Exception as exc:
            QMessageBox.critical(self, "API ingestion failed", str(exc))

    def _poll_job(self) -> None:
        if not self.current_job_id:
            return

        try:
            state = self.client.get_job(self.current_job_id)
            self.job_status_text.append(f"[{state['status']}] {state['message']}")
            if state["status"] in {"completed", "failed"}:
                self.timer.stop()
                self._refresh_sources()
        except Exception as exc:
            self.timer.stop()
            self.job_status_text.append(f"[error] {exc}")

    def _run_query(self) -> None:
        question = self.question_input.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "Question required", "Enter a question before querying.")
            return

        try:
            result = self.client.query(
                question=question,
                top_k=self.top_k_input.value(),
                chat_history=self.chat_history[-5:],
            )
            self.answer_output.setPlainText(result["answer"])
            self._render_metrics(result)
            self.chat_history.append(question)
        except Exception as exc:
            QMessageBox.critical(self, "Query failed", str(exc))

    def _render_metrics(self, result: dict) -> None:
        retrieval = result["retrieval_metrics"]
        perf = result["query_performance"]
        tokens = result["token_usage"]
        emb = result["embedding_insights"]

        self.metric_topk.setText(f"Top K: {retrieval['top_k']}")
        self.metric_scores.setText("Scores: " + ", ".join(str(s) for s in retrieval["similarity_scores"]))
        self.metric_avg.setText(f"Avg Similarity: {retrieval['avg_similarity_score']}")
        self.metric_latency.setText(
            f"Retrieval: {perf['retrieval_time_ms']}ms | LLM: {perf['llm_response_time_ms']}ms | Total: {perf['total_latency_ms']}ms"
        )
        self.metric_tokens.setText(
            f"Tokens -> in: {tokens['input_tokens']}, out: {tokens['output_tokens']}, total: {tokens['total_tokens']}"
        )
        self.metric_confidence.setText(
            (
                f"Confidence: {result['confidence_score']}% | Query Type: {result['query_type']} | "
                f"Mode: {result.get('applied_query_mode', '-')} | Style: {result.get('applied_response_style', '-')}"
            )
        )

        self.embedding_stats.setText(
            f"Model: {emb['model']}\nVector Dimension: {emb['vector_dimension']}\nAvg Embed Time/Chunk: {emb['avg_embedding_time_ms']} ms"
        )

        sources = []
        for source in result["source_attribution"]:
            sources.append(
                f"- {source['source_name']} ({source['source_type']}) | page/row: {source['page_or_row']} | chunk: {source['chunk_id']} | sim: {source['similarity']}"
            )
        self.source_attribution_output.setPlainText("\n".join(sources) if sources else "No sources")

        debug = result["debug"]
        self.debug_output.setPlainText(
            f"Raw Prompt:\n{debug['raw_prompt']}\n\nRetrieved Context:\n{debug['retrieved_context']}"
        )

    def _refresh_sources(self) -> None:
        self.sources_list.clear()
        try:
            sources = self.client.sources()
            for src in sources:
                self.sources_list.addItem(
                    f"{src['source_name']} [{src['source_type']}] chunks={src['chunk_count']}"
                )
        except Exception as exc:
            self.sources_list.addItem(f"Unable to load sources: {exc}")

    def _check_backend_connectivity(self) -> None:
        try:
            self.client.health()
        except Exception:
            QMessageBox.warning(
                self,
                "Backend Not Running",
                (
                    "Cannot reach FastAPI backend at http://127.0.0.1:8000.\n\n"
                    "Start backend first:\n"
                    "1) source .venv/bin/activate\n"
                    "2) ./scripts/run_backend.sh"
                ),
            )


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
