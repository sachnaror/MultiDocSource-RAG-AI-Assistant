from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class APIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def ingest_file(self, file_path: Path, source_name: str, source_type: str) -> dict[str, Any]:
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data = {"source_name": source_name, "source_type": source_type}
            response = requests.post(f"{self.base_url}/v1/ingest/file", files=files, data=data, timeout=60)
        response.raise_for_status()
        return response.json()

    def ingest_api(self, source_name: str, url: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/v1/ingest/api",
            json={"source_name": source_name, "url": url, "headers": {}},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/v1/jobs/{job_id}", timeout=20)
        response.raise_for_status()
        return response.json()

    def query(
        self,
        question: str,
        top_k: int = 5,
        chat_history: list[str] | None = None,
        query_mode: str = "auto",
        response_style: str = "concise",
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "top_k": top_k,
            "chat_history": chat_history or [],
            "query_mode": query_mode,
            "response_style": response_style,
        }
        response = requests.post(f"{self.base_url}/v1/query", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def dashboard(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/v1/dashboard", timeout=20)
        response.raise_for_status()
        return response.json()

    def sources(self) -> list[dict[str, Any]]:
        response = requests.get(f"{self.base_url}/v1/sources", timeout=20)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()
