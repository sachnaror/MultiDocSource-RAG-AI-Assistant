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

    def _split_table_cells(self, line: str) -> list[str]:
        raw = line.strip()
        if not raw:
            return []
        if "|" in raw:
            cells = [c.strip() for c in raw.split("|")]
            cells = [c for c in cells if c]
            return cells
        cells = [c.strip() for c in re.split(r"\s{2,}|\t+", raw) if c.strip()]
        return cells

    def _looks_like_header(self, cells: list[str]) -> bool:
        if len(cells) < 2:
            return False
        alpha_like = 0
        for cell in cells:
            token = cell.strip()
            if re.search(r"[A-Za-z]", token):
                alpha_like += 1
        return alpha_like >= max(2, len(cells) - 1)

    def _extract_pdf_table_records(self, raw_text: str, page_idx: int) -> list[dict[str, Any]]:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        tokenized: list[tuple[int, list[str]]] = []
        for line_idx, line in enumerate(lines):
            cells = self._split_table_cells(line)
            if len(cells) >= 2:
                tokenized.append((line_idx, cells))

        records: list[dict[str, Any]] = []
        i = 0
        while i < len(tokenized):
            _, header_cells = tokenized[i]
            if not self._looks_like_header(header_cells):
                i += 1
                continue

            header = [self._normalize_col_name(c) for c in header_cells]
            if len(set(header)) < 2:
                i += 1
                continue

            row_num = 0
            j = i + 1
            while j < len(tokenized):
                _, row_cells = tokenized[j]
                # Stop when another likely header starts.
                if self._looks_like_header(row_cells) and len(row_cells) == len(header):
                    break
                if len(row_cells) < 2:
                    break

                # Align row with header width.
                row_cells = row_cells[: len(header)] + ([""] * max(0, len(header) - len(row_cells)))
                row_data: dict[str, str] = {}
                parts: list[str] = []
                for col_name, value in zip(header, row_cells):
                    clean_value = str(value).strip()
                    if not col_name or not clean_value:
                        continue
                    row_data[col_name] = clean_value
                    parts.append(f"{col_name}: {clean_value}")

                if row_data:
                    row_num += 1
                    records.append(
                        {
                            "content": (
                                f"Page: {page_idx} | TableRow: {row_num} | "
                                + " | ".join(parts)
                            ),
                            "locator": f"Page {page_idx}, Table row {row_num}",
                            "page_or_row": f"{page_idx}:{row_num}",
                            "metadata": {
                                "page": page_idx,
                                "sheet": f"page_{page_idx}",
                                "row": row_num,
                                "columns": header,
                                "row_data": row_data,
                                "record_type": "table_row",
                            },
                        }
                    )
                j += 1

            # Advance to next potential block.
            i = j if j > i + 1 else i + 1

        return records

    def _extract_pdf_kv_records(self, raw_text: str, page_idx: int) -> list[dict[str, Any]]:
        normalized = self._normalize_pdf_text(raw_text)
        if not normalized:
            return []

        # Capture table/form-like "Field: Value" spans on each page.
        # Keep label detection strict to avoid chopping values that contain
        # title-cased phrases (for example long addresses).
        label_boundary = r"(?:[A-Z][A-Za-z0-9()/_&.'-]*\s*){1,3}"
        pattern = re.compile(
            rf"([A-Z][A-Za-z0-9 ()/&._'-]{{2,70}}?)\s*:\s*(.+?)(?=\s+{label_boundary}:|$)"
        )

        records: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        row_num = 0

        for key_raw, value_raw in pattern.findall(normalized):
            key = re.sub(r"\s+", " ", key_raw).strip(" -.;,")
            value = re.sub(r"\s+", " ", value_raw).strip(" -.;,")
            if not key or not value:
                continue
            if len(value) < 2 or len(value) > 400:
                continue

            key_norm = self._normalize_col_name(key)
            if not key_norm:
                continue

            # Address values in scanned/tabular PDFs often get prematurely
            # split by title-cased locality tokens. Expand via dedicated span.
            if "address" in key_norm:
                address_match = re.search(
                    r"address\s*:\s*(.+?)(?=\s+(?:regency|province|purpose|objectives?)\s*:|$)",
                    normalized,
                    flags=re.I,
                )
                if address_match:
                    expanded = re.sub(r"\s+", " ", address_match.group(1)).strip(" -.;,")
                    if len(expanded) > len(value):
                        value = expanded

            dedupe_key = (key_norm, value.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            row_num += 1
            records.append(
                {
                    "content": f"Page: {page_idx} | Field: {key} | Value: {value}",
                    "locator": f"Page {page_idx}, Field row {row_num}",
                    "page_or_row": f"{page_idx}:{row_num}",
                    "metadata": {
                        "page": page_idx,
                        "sheet": f"page_{page_idx}",
                        "row": row_num,
                        "columns": [key_norm],
                        "row_data": {key_norm: value},
                        "record_type": "table_row",
                    },
                }
            )

        return records

    async def parse_pdf(self, path: Path) -> list[dict[str, Any]]:
        reader = PdfReader(str(path))
        records: list[dict[str, Any]] = []

        for idx, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            table_records = self._extract_pdf_table_records(raw_text, idx)
            records.extend(table_records)
            kv_records = self._extract_pdf_kv_records(raw_text, idx)
            records.extend(kv_records)

            text = self._normalize_pdf_text(raw_text)
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
