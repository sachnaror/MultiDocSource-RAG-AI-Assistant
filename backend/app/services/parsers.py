from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pypdf import PdfReader


class ParserService:
    async def parse_pdf(self, path: Path) -> list[dict[str, Any]]:
        reader = PdfReader(str(path))
        records: list[dict[str, Any]] = []

        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
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
            for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
                values = [f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()]
                content = " | ".join(values).strip()
                if not content:
                    continue
                records.append(
                    {
                        "content": content,
                        "locator": f"Sheet {sheet_name}, Row {row_idx}",
                        "page_or_row": f"{sheet_name}:{row_idx}",
                        "metadata": {"sheet": sheet_name, "row": row_idx},
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
