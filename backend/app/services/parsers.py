from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pypdf import PdfReader


class ParserService:
    @staticmethod
    def _normalize_col_name(name: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(name).strip().lower())
        cleaned = re.sub(r"[^a-z0-9 _-]", "", cleaned)
        return cleaned

    def _normalize_pdf_text(self, text: str) -> str:
        normalized = text.replace("\n", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Fix OCR/extraction artifacts like "p y t h o n" -> "python".
        def _join_spaced_letters(match: re.Match[str]) -> str:
            return match.group(0).replace(" ", "")

        for _ in range(3):
            normalized = re.sub(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b", _join_spaced_letters, normalized)

        normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
        return normalized

    async def parse_pdf(self, path: Path) -> list[dict[str, Any]]:
        reader = PdfReader(str(path))
        records: list[dict[str, Any]] = []

        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self._normalize_pdf_text(text)
            records.append(
                {
                    "content": text.strip(),
                    "locator": f"Page {idx}",
                    "page_or_row": str(idx),
                    "metadata": {"page": idx},
                }
            )
        return records

    async def parse_excel(self, path: Path) -> list[dict[str, Any]]:
        workbook = pd.read_excel(path, sheet_name=None)
        records: list[dict[str, Any]] = []

        for sheet_name, df in workbook.items():
            df = df.fillna("")
            normalized_columns = [self._normalize_col_name(col) for col in df.columns]
            for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
                row_data: dict[str, str] = {}
                values: list[str] = []
                for raw_col, norm_col in zip(df.columns, normalized_columns):
                    value = str(row[raw_col]).strip()
                    if not value:
                        continue
                    row_data[norm_col] = value
                    values.append(f"{raw_col}: {value}")

                content = " | ".join(values).strip()
                if not content:
                    continue

                # Keep explicit table semantics in content so retrieval understands row/column intent.
                table_prefix = f"Sheet: {sheet_name} | Row: {row_idx} | "
                records.append(
                    {
                        "content": f"{table_prefix}{content}",
                        "locator": f"Sheet {sheet_name}, Row {row_idx}",
                        "page_or_row": f"{sheet_name}:{row_idx}",
                        "metadata": {
                            "sheet": sheet_name,
                            "row": row_idx,
                            "columns": normalized_columns,
                            "row_data": row_data,
                            "record_type": "table_row",
                        },
                    }
                )
        return records

    async def parse_api(self, url: str, headers: dict[str, str]) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()

        records: list[dict[str, Any]] = []
        if isinstance(payload, list):
            iterable = payload
        else:
            iterable = [payload]

        for idx, item in enumerate(iterable, start=1):
            content = json.dumps(item, ensure_ascii=True)
            records.append(
                {
                    "content": content,
                    "locator": f"API Record {idx}",
                    "page_or_row": str(idx),
                    "metadata": {"record": idx},
                }
            )
        return records
