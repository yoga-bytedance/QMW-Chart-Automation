#!/usr/bin/env python3
# Requirements:
#   pip install matplotlib numpy pillow requests
# Runtime dependency:
#   mdp-cli must be available on PATH when fetching MDP-backed report IDs.
"""
1. Read and validate QMW template JSON files.
2. Resolve MDP report IDs from embedded template URLs.
3. Fetch raw datasets through `mdp-cli` (or direct JSON URLs when available).
4. Transform fetched data into the grain structure expected by QMW charts.
5. Build legacy or multi-subplot chart specs.
6. Render high-resolution static images in PNG, JPEG, or SVG.
7. Validate saved output for common viewer compatibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass


import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.transforms import blended_transform_factory
from scipy.interpolate import PchipInterpolator

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image


LOGGER = logging.getLogger("qmw_chart_workflow")

REVENUE_MEASURE_HINTS = ("daily average", "daily revenue")
YOY_MEASURE_HINT = "percentage difference"
GRAIN_KEYS = ("q", "m", "w")
WEEK_LABEL_RE = re.compile(
    r".*?W\d+(?:\s*\((\d{2}/\d{2})\s*~\s*\d{2}/\d{2}\))?.*"
)
QM_YEAR_PREFIX_RE = re.compile(r"^(\d{2})(\d{2})-(.+)$")
REPORT_URL_RE = re.compile(r"/report/edit/(\d+)")


class TemplateValidationError(ValueError):
    """Raised when a QMW template is malformed."""


class DataFetchError(RuntimeError):
    """Raised when a remote report cannot be retrieved."""


class RenderValidationError(RuntimeError):
    """Raised when a generated image is invalid."""


def slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")
    return value[:100] or "chart"


def _empty_grain() -> Dict[str, Any]:
    return {
        "x_labels": [],
        "series_names": [],
        "series_values": [],
        "yoy_series_names": [],
        "yoy_series_values": [],
        "overall_revenue": None,
        "overall_yoy": None,
        "measures": {},
    }


def _ensure_dict(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TemplateValidationError(f"{context} must be an object")
    return value


def _ensure_string_list(value: Any, context: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TemplateValidationError(f"{context} must be a list of strings")
    return list(value)


def _validate_format_block(block: Any, context: str) -> Dict[str, Any]:
    block = _ensure_dict(block, context)
    raw = block.get("raw")
    if raw is not None and not isinstance(raw, str):
        raise TemplateValidationError(f"{context}.raw must be a string when provided")
    validated = {"raw": raw or ""}
    for label in ("Q", "M", "W"):
        validated[label] = _ensure_string_list(block.get(label, []), f"{context}.{label}")
    return validated


def _extract_report_id_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise TemplateValidationError(f"Invalid URL: {url}")
    match = REPORT_URL_RE.search(parsed.path)
    if match:
        return match.group(1)
    query_match = re.search(r"(?:^|[?&])id=(\d+)(?:&|$)", parsed.query)
    if query_match:
        return query_match.group(1)
    return None


def _normalize_source_ref(
    ref: Any, chart_title: str, grain_key: str
) -> Dict[str, Optional[str]]:
    if ref in (None, {}):
        return {"report_id": None, "url": None, "fetch_mode": None}
    ref = _ensure_dict(ref, f"chart[{chart_title!r}].{grain_key}")
    url = ref.get("url") or ref.get("data_url")
    report_id = ref.get("report_id")

    if url is not None and not isinstance(url, str):
        raise TemplateValidationError(
            f"chart[{chart_title!r}].{grain_key}.url must be a string"
        )
    if report_id is not None and not isinstance(report_id, str):
        raise TemplateValidationError(
            f"chart[{chart_title!r}].{grain_key}.report_id must be a string"
        )

    extracted = _extract_report_id_from_url(url) if url else None
    if report_id and extracted and report_id != extracted:
        raise TemplateValidationError(
            f"chart[{chart_title!r}].{grain_key} has mismatched report_id/url: "
            f"{report_id} != {extracted}"
        )
    resolved_report_id = report_id or extracted
    if not resolved_report_id and not url:
        raise TemplateValidationError(
            f"chart[{chart_title!r}].{grain_key} must define report_id or url"
        )
    fetch_mode = "mdp" if resolved_report_id else "http"
    return {
        "report_id": resolved_report_id,
        "url": url,
        "fetch_mode": fetch_mode,
    }


def _validate_subplot_format(
    value: Any, chart_title: str
) -> Optional[List[Dict[str, Any]]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TemplateValidationError(
            f"chart[{chart_title!r}].subplot_format must be a list or null"
        )
    normalized = []
    for index, item in enumerate(value, start=1):
        item = _ensure_dict(item, f"chart[{chart_title!r}].subplot_format[{index}]")
        name = item.get("name")
        chart_type = item.get("type")
        title = item.get("title")
        if not isinstance(name, str) or not name.strip():
            raise TemplateValidationError(
                f"chart[{chart_title!r}].subplot_format[{index}].name is required"
            )
        if chart_type not in ("stackplot", "line"):
            raise TemplateValidationError(
                f"chart[{chart_title!r}].subplot_format[{index}].type "
                "must be 'stackplot' or 'line'"
            )
        if title is not None and not isinstance(title, str):
            raise TemplateValidationError(
                f"chart[{chart_title!r}].subplot_format[{index}].title must be a string"
            )
        fmt = item.get("format")
        if fmt is not None:
            fmt = _ensure_dict(
                fmt, f"chart[{chart_title!r}].subplot_format[{index}].format"
            )
            label_kind = fmt.get("label_kind")
            unit = fmt.get("unit")
            if label_kind not in (None, "number", "percentage"):
                raise TemplateValidationError(
                    f"chart[{chart_title!r}].subplot_format[{index}].format."
                    "label_kind must be number, percentage, or null"
                )
            if unit not in (None, "K", "M", "None", "Full", "raw"):
                raise TemplateValidationError(
                    f"chart[{chart_title!r}].subplot_format[{index}].format.unit "
                    "must be K, M, None, Full, raw, or null"
                )
        normalized.append(
            {
                "name": name,
                "type": chart_type,
                "title": title,
                "format": fmt,
            }
        )
    return normalized


def validate_template_document(payload: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    if "format" not in payload:
        raise TemplateValidationError(f"{source_path} is missing top-level 'format'")
    if "charts" not in payload:
        raise TemplateValidationError(f"{source_path} is missing top-level 'charts'")

    fmt = _validate_format_block(payload["format"], f"{source_path}.format")
    charts = payload["charts"]
    if not isinstance(charts, list):
        raise TemplateValidationError(f"{source_path}.charts must be a list")

    normalized_charts = []
    for index, chart in enumerate(charts):
        chart = _ensure_dict(chart, f"{source_path}.charts[{index}]")
        title = chart.get("title")
        source = chart.get("source")
        if not isinstance(title, str) or not title.strip():
            raise TemplateValidationError(
                f"{source_path}.charts[{index}].title must be a non-empty string"
            )
        if source is not None and not isinstance(source, str):
            raise TemplateValidationError(
                f"{source_path}.charts[{index}].source must be a string when provided"
            )
        unit = chart.get("unit")
        if unit not in (None, "K", "M", "None", "Full"):
            raise TemplateValidationError(
                f"{source_path}.charts[{index}].unit must be K, M, None, Full, or null"
            )
        mapping = chart.get("mapping", {})
        if not isinstance(mapping, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()
        ):
            raise TemplateValidationError(
                f"{source_path}.charts[{index}].mapping must be an object of strings"
            )
        fixed_bottom = chart.get("fixed_bottom")
        if fixed_bottom is not None:
            if isinstance(fixed_bottom, list):
                if any(not isinstance(item, str) for item in fixed_bottom):
                    raise TemplateValidationError(
                        f"{source_path}.charts[{index}].fixed_bottom must contain strings"
                    )
            elif not isinstance(fixed_bottom, str):
                raise TemplateValidationError(
                    f"{source_path}.charts[{index}].fixed_bottom must be string, list, or null"
                )
        custom_color = chart.get("custom_color", {})
        if not isinstance(custom_color, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in custom_color.items()
        ):
            raise TemplateValidationError(
                f"{source_path}.charts[{index}].custom_color must be an object of strings"
            )
        subplot_format = _validate_subplot_format(chart.get("subplot_format"), title)
        period_format = chart.get("period_format")
        if period_format is not None:
            period_format = _validate_format_block(
                period_format, f"{source_path}.charts[{index}].period_format"
            )

        normalized_chart = {
            "title": title,
            "unit": unit,
            "source": source or "",
            "mapping": dict(mapping),
            "fixed_bottom": fixed_bottom,
            "custom_color": dict(custom_color),
            "subplot_format": subplot_format,
            "period_format": period_format,
        }
        for grain_key in GRAIN_KEYS:
            if grain_key not in chart:
                raise TemplateValidationError(
                    f"{source_path}.charts[{index}] is missing '{grain_key}'"
                )
            normalized_chart[grain_key] = _normalize_source_ref(
                chart.get(grain_key), title, grain_key
            )
        normalized_charts.append(normalized_chart)

    return {"format": fmt, "charts": normalized_charts, "source_path": str(source_path)}


def load_template_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TemplateValidationError(f"Invalid JSON in {path}: {exc}") from exc
    LOGGER.info("Loaded template: %s", path)
    return validate_template_document(payload, path)


def extract_rows(data: Any) -> List[Dict[str, Any]]:
    rows = None
    if isinstance(data, dict):
        inner = data.get("data", data)
        if isinstance(inner, dict):
            if "rows" in inner: rows = inner["rows"]
            elif "data" in inner: rows = inner["data"]
            elif "results" in inner: rows = inner["results"]
    elif isinstance(data, list):
        rows = data

    if not isinstance(rows, list):
        raise DataFetchError("Fetched data does not contain a rows/data/results list")
    if rows and not isinstance(rows[0], dict):
        raise DataFetchError("Fetched rows must contain objects")
    return rows


def validate_fetched_dataset(data: Any) -> None:
    if isinstance(data, dict) and data.get("status") == "error":
        err = data.get("error") or {}
        raise DataFetchError(
            "Upstream returned an error payload: %s" % err.get("message", str(err))
        )
    extract_rows(data)


def parse_json_from_text(text: str) -> Any:
    stripped = text.strip()
    json_start = stripped.find("{")
    if json_start < 0:
        json_start = stripped.find("[")
    if json_start < 0:
        raise DataFetchError("Response did not contain JSON content")
    try:
        return json.loads(stripped[json_start:])
    except json.JSONDecodeError as exc:
        raise DataFetchError(f"Response contained invalid JSON: {exc}") from exc


class DataFetcher:
    """Fetch QMW data sources with retry and timeout management."""

    def __init__(
        self,
        mdp_cli: str = "mdp-cli",
        retries: int = 3,
        timeout: int = 120,
        retry_backoff_seconds: float = 1.5,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.mdp_cli = mdp_cli
        self.retries = max(1, retries)
        self.timeout = timeout
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()

    def fetch(self, source: Dict[str, Optional[str]]) -> Any:
        if not source.get("fetch_mode"):
            return None

        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                LOGGER.info(
                    "Fetching %s source (%s), attempt %s/%s",
                    source["fetch_mode"],
                    source.get("report_id") or source.get("url"),
                    attempt,
                    self.retries,
                )
                if source["fetch_mode"] == "mdp":
                    payload = self._fetch_via_mdp(source["report_id"])
                else:
                    payload = self._fetch_via_http(source["url"])
                validate_fetched_dataset(payload)
                return payload
            except (DataFetchError, requests.RequestException, subprocess.TimeoutExpired) as exc:
                last_error = exc
                if attempt >= self.retries or not self._should_retry(exc):
                    break
                sleep_for = self.retry_backoff_seconds * attempt
                LOGGER.warning("Fetch failed (%s). Retrying in %.1fs", exc, sleep_for)
                time.sleep(sleep_for)

        raise DataFetchError(
            "Failed to fetch source %s after %s attempts: %s"
            % (source.get("report_id") or source.get("url"), self.retries, last_error)
        )

    def _should_retry(self, exc: BaseException) -> bool:
        if isinstance(exc, subprocess.TimeoutExpired):
            return True
        if isinstance(exc, requests.Timeout):
            return True
        if isinstance(exc, requests.ConnectionError):
            return True
        if isinstance(exc, requests.HTTPError):
            response = exc.response
            return bool(response is not None and response.status_code >= 500)
        if isinstance(exc, DataFetchError):
            message = str(exc).lower()
            return any(
                token in message
                for token in ("timeout", "tempor", "connection", "502", "503", "504")
            )
        return False

    def _fetch_via_mdp(self, report_id: Optional[str]) -> Any:
        if not report_id:
            raise DataFetchError("Missing report_id for mdp-backed fetch")
        command = [
            self.mdp_cli,
            "ac",
            "report",
            "query-data",
            "--id",
            report_id,
            "-j",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise DataFetchError(
                "mdp-cli was not found on PATH; install it or pass --mdp-cli"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise DataFetchError(
                "mdp-cli exited with code %s: %s" % (result.returncode, stderr[:500])
            )
        return parse_json_from_text(result.stdout)

    def _fetch_via_http(self, url: Optional[str]) -> Any:
        if not url:
            raise DataFetchError("Missing URL for HTTP fetch")
        response = self.session.get(
            url,
            timeout=self.timeout,
            headers={"Accept": "application/json, text/plain;q=0.9, */*;q=0.8"},
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            LOGGER.debug("HTTP fetch failed body: %s", response.text[:500])
            raise
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type.lower():
            return response.json()
        return parse_json_from_text(response.text)


def transform_mdp_data(data: Any) -> Dict[str, Any]:
    """Transform mdp-cli query-data output into grain dict format."""
    if data is None:
        return _empty_grain()

    if isinstance(data, dict) and data.get("status") == "error":
        err = data.get("error") or {}
        raise DataFetchError("mdp-cli returned error: %s" % err.get("message", str(err)))

    rows = None
    if isinstance(data, dict):
        inner = data.get("data", data)
        if isinstance(inner, dict):
            rows = inner.get("rows") or inner.get("data") or inner.get("results")
    if rows is None and isinstance(data, list):
        rows = data
    if not rows or not isinstance(rows, list):
        return _empty_grain()
    first_row = rows[0]
    if not isinstance(first_row, dict):
        return _empty_grain()

    if isinstance(data, dict):
        inner = data.get("data", data)
        if isinstance(inner, dict) and isinstance(inner.get("columns"), list):
            all_keys = inner["columns"]
        else:
            all_keys = list(first_row.keys())
    else:
        all_keys = list(first_row.keys())

    time_keys = []
    series_keys = []
    metric_keys = []
    for key in all_keys:
        sample_vals = set()
        for row in rows:
            value = row.get(key)
            if value is not None:
                sample_vals.add(str(value))
        is_time = any(
            re.match(r"^\d{4}-[QMW]\d", str(value))
            or re.match(r"^\d{4}-\d{2}$", str(value))
            for value in sample_vals
        )
        is_numeric = False
        has_non_none = False
        for row in rows:
            value = row.get(key)
            if value is not None:
                has_non_none = True
                if isinstance(value, (int, float)):
                    is_numeric = True
                    break
                try:
                    float(value)
                    is_numeric = True
                    break
                except (ValueError, TypeError):
                    pass
        if not has_non_none:
            continue
        if is_time:
            time_keys.append(key)
        elif is_numeric:
            metric_keys.append(key)
        else:
            series_keys.append(key)

    if not time_keys:
        for key in all_keys:
            lowered = key.lower()
            if any(hint in lowered for hint in ("quarter", "month", "week", "date", "period", "time")):
                if key not in time_keys:
                    time_keys.append(key)
    if not time_keys:
        return _empty_grain()

    time_key = time_keys[0]
    series_key = series_keys[0] if series_keys else None

    revenue_metrics = []
    yoy_metrics = []
    other_metrics = []
    for key in metric_keys:
        lowered = key.lower()
        if YOY_MEASURE_HINT in lowered or "yoy" in lowered:
            yoy_metrics.append(key)
        elif any(hint in lowered for hint in REVENUE_MEASURE_HINTS):
            revenue_metrics.append(key)
        else:
            other_metrics.append(key)
    if not revenue_metrics and other_metrics:
        revenue_metrics = [other_metrics.pop(0)]
    if not yoy_metrics:
        for index, key in enumerate(list(other_metrics)):
            lowered = key.lower()
            if "yoy" in lowered or "percentage" in lowered or "diff" in lowered:
                yoy_metrics.append(key)
                other_metrics.pop(index)
                break

    revenue_metric = revenue_metrics[0] if revenue_metrics else None
    yoy_metric = yoy_metrics[0] if yoy_metrics else None

    x_labels = []
    x_seen = set()
    for row in rows:
        value = str(row.get(time_key, ""))
        if value and value not in x_seen:
            x_labels.append(value)
            x_seen.add(value)

    series_names = []
    if series_key:
        series_seen = set()
        for row in rows:
            value = str(row.get(series_key, ""))
            if value and value not in series_seen:
                series_names.append(value)
                series_seen.add(value)
    else:
        series_names = ["Total"]

    revenue_lookup = {}
    yoy_lookup = {}
    for row in rows:
        period = str(row.get(time_key, ""))
        series = str(row.get(series_key, "Total")) if series_key else "Total"
        if revenue_metric:
            value = row.get(revenue_metric)
            if value is not None:
                try:
                    revenue_lookup[(period, series)] = float(value)
                except (ValueError, TypeError):
                    revenue_lookup[(period, series)] = None
        if yoy_metric:
            value = row.get(yoy_metric)
            if value is not None:
                try:
                    yoy_value = float(value)
                    lowered = yoy_metric.lower()
                    if "yoy(" in lowered or lowered.startswith("yoy"):
                        yoy_value = yoy_value * 100
                    elif abs(yoy_value) < 50 and (
                        "percentage" in lowered or "diff" in lowered
                    ):
                        yoy_value = (yoy_value - 1) * 100
                    yoy_lookup[(period, series)] = yoy_value
                except (ValueError, TypeError):
                    yoy_lookup[(period, series)] = None

    sum_label = "Sum"
    has_sum = sum_label in series_names
    if has_sum:
        series_names = [name for name in series_names if name != sum_label]

    series_values = [
        [revenue_lookup.get((label, series_name)) for label in x_labels]
        for series_name in series_names
    ]
    yoy_series_values = [
        [yoy_lookup.get((label, series_name)) for label in x_labels]
        for series_name in series_names
    ]

    overall_revenue = []
    overall_yoy = []
    for label in x_labels:
        if has_sum:
            overall_revenue.append(revenue_lookup.get((label, sum_label)))
            overall_yoy.append(yoy_lookup.get((label, sum_label)))
        else:
            revenue_sum = sum(revenue_lookup.get((label, name), 0) or 0 for name in series_names)
            overall_revenue.append(revenue_sum if revenue_sum != 0 else None)
            numerator = 0.0
            denominator = 0.0
            for name in series_names:
                revenue_value = revenue_lookup.get((label, name))
                yoy_value = yoy_lookup.get((label, name))
                if revenue_value is not None and yoy_value is not None:
                    numerator += float(revenue_value) * float(yoy_value)
                    denominator += float(revenue_value)
            overall_yoy.append(numerator / denominator if denominator > 0 else None)

    measures_all = {}
    for metric_key in metric_keys:
        lowered = metric_key.lower()
        metric_lookup = {}
        for row in rows:
            period = str(row.get(time_key, ""))
            series = str(row.get(series_key, "Total")) if series_key else "Total"
            value = row.get(metric_key)
            if value is not None:
                try:
                    numeric_value = float(value)
                    if "yoy(" in lowered or lowered.startswith("yoy"):
                        numeric_value = numeric_value * 100
                    elif abs(numeric_value) < 50 and (
                        "percentage" in lowered or "diff" in lowered
                    ):
                        numeric_value = (numeric_value - 1) * 100
                    metric_lookup[(period, series)] = numeric_value
                except (ValueError, TypeError):
                    metric_lookup[(period, series)] = None
        metric_series_values = [
            [metric_lookup.get((label, name)) for label in x_labels]
            for name in series_names
        ]
        metric_overall = []
        for label in x_labels:
            values = [metric_lookup.get((label, name)) for name in series_names]
            non_none = [value for value in values if value is not None]
            metric_overall.append(sum(non_none) if non_none else None)
        measures_all[metric_key] = {
            "series_names": list(series_names),
            "series_values": metric_series_values,
            "overall": metric_overall,
        }

    return {
        "x_labels": x_labels,
        "series_names": series_names,
        "series_values": series_values,
        "yoy_series_names": list(series_names),
        "yoy_series_values": yoy_series_values,
        "overall_revenue": overall_revenue if any(v is not None for v in overall_revenue) else None,
        "overall_yoy": overall_yoy if any(v is not None for v in overall_yoy) else None,
        "measures": measures_all,
    }


def apply_format_filter(grain: Dict[str, Any], tokens: Optional[Sequence[str]]) -> Dict[str, Any]:
    if tokens is None:
        return grain
    if not tokens:
        return _empty_grain()

    labels = grain.get("x_labels", [])
    keep_indices = []
    keep_labels = []
    for token in tokens:
        for index, label in enumerate(labels):
            bare_label = re.sub(r"\s*\(.*\)\s*$", "", label).strip()
            bare_token = re.sub(r"\s*\(.*\)\s*$", "", token).strip()
            if label == token or bare_label == token or bare_label == bare_token:
                keep_indices.append(index)
                keep_labels.append(labels[index])
                break

    def slice_row(row: Sequence[Any]) -> List[Any]:
        return [row[index] for index in keep_indices]

    out = dict(grain)
    out["x_labels"] = keep_labels
    out["series_values"] = [slice_row(row) for row in grain.get("series_values", [])]
    out["yoy_series_values"] = [
        slice_row(row) for row in grain.get("yoy_series_values", [])
    ]
    if grain.get("overall_revenue") is not None:
        out["overall_revenue"] = slice_row(grain["overall_revenue"])
    if grain.get("overall_yoy") is not None:
        out["overall_yoy"] = slice_row(grain["overall_yoy"])
    if grain.get("measures"):
        measures = {}
        for name, measure in grain["measures"].items():
            measures[name] = {
                "series_names": list(measure.get("series_names") or []),
                "series_values": [
                    slice_row(row) for row in (measure.get("series_values") or [])
                ],
                "overall": (
                    slice_row(measure["overall"])
                    if measure.get("overall") is not None
                    else None
                ),
            }
        out["measures"] = measures
    return out


def shorten_week_label(label: str) -> str:
    match = WEEK_LABEL_RE.fullmatch(label)
    if not match:
        return label
    first_date = match.group(1)
    if first_date:
        return "W%s" % first_date
    return re.sub(r"^\d{4}-", "", label)


def shorten_qm_label(label: str) -> str:
    match = QM_YEAR_PREFIX_RE.match(label)
    if not match:
        return label
    return "%s-%s" % (match.group(2), match.group(3))


def _resolve_chart_unit(chart: Dict[str, Any], unit_override: Optional[str]) -> str:
    title = chart.get("title", "")
    if unit_override:
        return unit_override
    if chart.get("unit"):
        return chart["unit"]
    if "$K" in title or "[$k" in title.lower():
        return "K"
    if "$M" in title or "[$m" in title.lower():
        return "M"
    return "K"


def _find_measure(
    grain: Dict[str, Any], needle: str, excluded_keys: Optional[Iterable[str]] = None
) -> Optional[Tuple[str, Dict[str, Any]]]:
    measures = grain.get("measures") or {}
    needle_lower = needle.strip().lower()
    excluded = set(excluded_keys or [])
    matches = [
        (key, value)
        for key, value in measures.items()
        if key not in excluded and needle_lower in key.lower()
    ]
    if not matches and "percentage difference" in needle_lower:
        yoy_matches = [
            (key, value)
            for key, value in measures.items()
            if key not in excluded and key.lower().startswith("yoy(")
        ]
        if len(yoy_matches) == 1:
            matches = yoy_matches
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        prefix_hits = [
            (key, value)
            for key, value in matches
            if key.lower().startswith(needle_lower) or key.lower() == needle_lower
        ]
        if len(prefix_hits) == 1:
            return prefix_hits[0]
    return None


def _measure_match_diagnostic(
    grain: Dict[str, Any], needle: str, excluded_keys: Optional[Iterable[str]] = None
) -> str:
    measures = grain.get("measures") or {}
    needle_lower = needle.strip().lower()
    excluded = set(excluded_keys or [])
    matches = [
        key
        for key in measures
        if key not in excluded and needle_lower in key.lower()
    ]
    if not matches:
        available = [key for key in measures if key not in excluded]
        return "no measure matches %r; available: %s" % (needle, available)
    return "directive %r matches multiple measures: %s" % (needle, matches)


def _is_percentage_difference_directive(directive: Dict[str, Any]) -> bool:
    return "percentage difference" in (directive.get("name") or "").strip().lower()


def spec_from_parsed_chart_subplots(
    chart: Dict[str, Any],
    unit_override: Optional[str] = None,
    source_override: Optional[str] = None,
) -> Dict[str, Any]:
    directives = chart["subplot_format"]
    mapping = chart.get("mapping") or {}

    def remap(name: str) -> str:
        return mapping.get(name, name)

    q = chart.get("q") or {}
    m = chart.get("m") or {}
    w = chart.get("w") or {}

    q_count = len(q.get("x_labels") or [])
    m_count = len(m.get("x_labels") or [])
    w_count = len(w.get("x_labels") or [])
    display_q = [shorten_qm_label(label) for label in (q.get("x_labels") or [])]
    display_m = [shorten_qm_label(label) for label in (m.get("x_labels") or [])]
    display_w = [shorten_week_label(label) for label in (w.get("x_labels") or [])]
    x_labels = display_q + display_m + display_w
    total = q_count + m_count + w_count
    chart_unit = _resolve_chart_unit(chart, unit_override)

    fixed_bottom_raw = chart.get("fixed_bottom")
    if isinstance(fixed_bottom_raw, str):
        fixed_bottom_list_raw = [fixed_bottom_raw]
    elif isinstance(fixed_bottom_raw, list):
        fixed_bottom_list_raw = [str(item) for item in fixed_bottom_raw if item]
    else:
        fixed_bottom_list_raw = []
    seen_fixed = set()
    fixed_bottom_list = []
    for name in fixed_bottom_list_raw:
        remapped = remap(name)
        if remapped and remapped not in seen_fixed:
            fixed_bottom_list.append(remapped)
            seen_fixed.add(remapped)

    custom_color = {remap(name): color for name, color in (chart.get("custom_color") or {}).items()}
    claimed_per_grain = {"q": set(), "m": set(), "w": set()}
    indexed_directives = list(enumerate(directives, start=1))
    resolution_order = [
        item for item in indexed_directives if _is_percentage_difference_directive(item[1])
    ] + [
        item for item in indexed_directives if not _is_percentage_difference_directive(item[1])
    ]
    resolved = {}

    for index, directive in resolution_order:
        needle = directive["name"]
        per_grain = {"q": None, "m": None, "w": None}
        for key, grain in (("q", q), ("m", m), ("w", w)):
            if not (grain.get("x_labels") or []):
                continue
            hit = _find_measure(grain, needle, excluded_keys=claimed_per_grain[key])
            if hit is None:
                raise ValueError(
                    "Chart %r subplot %s (%r): %s"
                    % (
                        chart.get("title"),
                        index,
                        needle,
                        _measure_match_diagnostic(
                            grain, needle, excluded_keys=claimed_per_grain[key]
                        ),
                    )
                )
            per_grain[key] = hit
            claimed_per_grain[key].add(hit[0])
        resolved[index] = per_grain

    subplots = []
    for index, directive in indexed_directives:
        needle = directive["name"]
        subplot_type = directive["type"]
        fmt_override = directive.get("format") or {}
        override_label = fmt_override.get("label_kind")
        override_unit = fmt_override.get("unit")
        per_grain = resolved[index]

        union_names = []
        seen_names = {}
        for key in ("q", "m", "w"):
            hit = per_grain[key]
            if hit is None:
                continue
            _, measure_data = hit
            for name in measure_data.get("series_names") or []:
                remapped = remap(name)
                if remapped not in seen_names:
                    seen_names[remapped] = len(union_names)
                    union_names.append(remapped)

        values_rows = [[None] * total for _ in union_names]
        offset = 0
        for key, count in (("q", q_count), ("m", m_count), ("w", w_count)):
            hit = per_grain[key]
            if hit is not None:
                _, measure_data = hit
                src_names = measure_data.get("series_names") or []
                src_values = measure_data.get("series_values") or []
                for local_index, name in enumerate(src_names):
                    remapped = remap(name)
                    if remapped not in seen_names:
                        continue
                    global_index = seen_names[remapped]
                    for time_index in range(count):
                        if time_index < len(src_values[local_index]):
                            value = src_values[local_index][time_index]
                            values_rows[global_index][offset + time_index] = (
                                None if value is None else float(value)
                            )
            offset += count

        if override_label is not None:
            label_kind = override_label
        else:
            label_kind = "number" if subplot_type == "stackplot" else "percentage"
        if label_kind == "percentage":
            value_format = "pct"
        else:
            if override_unit is not None:
                unit_choice = override_unit
            else:
                if any(token in needle.lower() for token in ("revenue", "gmv")):
                    unit_choice = chart_unit
                else:
                    unit_choice = "None"
            value_format = {"K": "K", "M": "M", "Full": "raw", "raw": "raw"}.get(
                unit_choice, "raw"
            )

        overall_values = None
        include_overall = False
        if subplot_type == "line":
            first_measure_name = ""
            for key in ("q", "m", "w"):
                hit = per_grain[key]
                if hit:
                    first_measure_name = hit[0].lower()
                    break
            looks_yoy = "yoy" in first_measure_name or "yoy" in needle.lower()
            if looks_yoy and len(union_names) >= 2:
                include_overall = True
                overall_values = [None] * total
                offset = 0
                for key, count in (("q", q_count), ("m", m_count), ("w", w_count)):
                    grain = {"q": q, "m": m, "w": w}[key]
                    if count == 0:
                        offset += count
                        continue
                    revenue_data = None
                    for measure_name, measure_data in (grain.get("measures") or {}).items():
                        lowered = measure_name.lower()
                        if (
                            ("revenue" in lowered or "gmv" in lowered)
                            and "yoy" not in lowered
                            and "percentage difference" not in lowered
                        ):
                            revenue_data = measure_data
                            break
                    for time_index in range(count):
                        if revenue_data is not None:
                            numerator = 0.0
                            denominator = 0.0
                            had_any = False
                            src_names = revenue_data.get("series_names") or []
                            src_values = revenue_data.get("series_values") or []
                            for series_index, src_name in enumerate(src_names):
                                remapped = remap(src_name)
                                if remapped not in seen_names:
                                    continue
                                global_index = seen_names[remapped]
                                revenue_value = (
                                    src_values[series_index][time_index]
                                    if time_index < len(src_values[series_index])
                                    else None
                                )
                                yoy_value = values_rows[global_index][offset + time_index]
                                if revenue_value is None or yoy_value is None:
                                    continue
                                numerator += float(revenue_value) * float(yoy_value)
                                denominator += float(revenue_value)
                                had_any = True
                            if had_any and denominator:
                                overall_values[offset + time_index] = numerator / denominator
                        else:
                            values = [
                                row[offset + time_index]
                                for row in values_rows
                                if row[offset + time_index] is not None
                            ]
                            if values:
                                overall_values[offset + time_index] = sum(values) / len(values)
                    offset += count

        if subplot_type == "stackplot":
            totals = {
                name: sum(value for value in values_rows[idx] if value is not None)
                for idx, name in enumerate(union_names)
            }
            sorted_names = sorted(union_names, key=lambda name: totals[name], reverse=True)
            pinned_present = [name for name in fixed_bottom_list if name in sorted_names]
            for name in pinned_present:
                sorted_names.remove(name)
            sorted_names = pinned_present + sorted_names
            name_index = {name: idx for idx, name in enumerate(union_names)}
            union_names = sorted_names
            values_rows = [values_rows[name_index[name]] for name in union_names]

        subplots.append(
            {
                "name": needle,
                "title_override": directive.get("title"),
                "type": subplot_type,
                "names": union_names,
                "values": values_rows,
                "value_format": value_format,
                "include_overall": include_overall,
                "overall_values": overall_values,
            }
        )

    return {
        "title": chart.get("title", ""),
        "source": source_override if source_override is not None else chart.get("source"),
        "q_count": q_count,
        "m_count": m_count,
        "w_count": w_count,
        "x_labels": x_labels,
        "subplots": subplots,
        "custom_color": custom_color,
    }


def spec_from_parsed_chart(
    chart: Dict[str, Any],
    unit_override: Optional[str] = None,
    source_override: Optional[str] = None,
) -> Dict[str, Any]:
    q = chart.get("q") or _empty_grain()
    m = chart.get("m") or _empty_grain()
    w = chart.get("w") or _empty_grain()
    mapping = chart.get("mapping") or {}

    def remap(name: str) -> str:
        return mapping.get(name, name)

    q_count = len(q.get("x_labels") or [])
    m_count = len(m.get("x_labels") or [])
    w_count = len(w.get("x_labels") or [])
    x_labels = (
        [shorten_qm_label(label) for label in q.get("x_labels") or []]
        + [shorten_qm_label(label) for label in m.get("x_labels") or []]
        + [shorten_week_label(label) for label in w.get("x_labels") or []]
    )

    left_names = []
    seen_left = {}
    for grain in (q, m, w):
        for name in grain.get("series_names") or []:
            remapped = remap(name)
            if remapped not in seen_left:
                seen_left[remapped] = len(left_names)
                left_names.append(remapped)
    left_values = [
        [None] * (q_count + m_count + w_count) for _ in left_names
    ]
    offset = 0
    for grain, count in ((q, q_count), (m, m_count), (w, w_count)):
        for local_index, name in enumerate(grain.get("series_names") or []):
            global_index = seen_left[remap(name)]
            series_values = grain["series_values"][local_index]
            for time_index in range(count):
                if time_index < len(series_values):
                    value = series_values[time_index]
                    left_values[global_index][offset + time_index] = (
                        None if value is None else float(value)
                    )
        offset += count

    right_names = []
    seen_right = {}
    for grain in (q, m, w):
        for name in grain.get("yoy_series_names") or []:
            remapped = remap(name)
            if remapped not in seen_right:
                seen_right[remapped] = len(right_names)
                right_names.append(remapped)
    include_overall = len(right_names) >= 2
    if include_overall and "Overall" not in seen_right:
        seen_right["Overall"] = len(right_names)
        right_names.append("Overall")

    right_values = [
        [None] * (q_count + m_count + w_count) for _ in right_names
    ]
    offset = 0
    for grain, count in ((q, q_count), (m, m_count), (w, w_count)):
        for local_index, name in enumerate(grain.get("yoy_series_names") or []):
            global_index = seen_right[remap(name)]
            series_values = grain["yoy_series_values"][local_index]
            for time_index in range(count):
                if time_index < len(series_values):
                    value = series_values[time_index]
                    right_values[global_index][offset + time_index] = (
                        None if value is None else float(value)
                    )
        if include_overall:
            overall = grain.get("overall_yoy") or [None] * count
            global_index = seen_right["Overall"]
            for time_index in range(count):
                if time_index < len(overall):
                    value = overall[time_index]
                    right_values[global_index][offset + time_index] = (
                        None if value is None else float(value)
                    )
        offset += count

    fixed_bottom_raw = chart.get("fixed_bottom")
    if isinstance(fixed_bottom_raw, str):
        fixed_bottom_list_raw = [fixed_bottom_raw]
    elif isinstance(fixed_bottom_raw, list):
        fixed_bottom_list_raw = [str(item) for item in fixed_bottom_raw if item]
    else:
        fixed_bottom_list_raw = []
    fixed_bottom_list = []
    seen_fixed = set()
    for name in fixed_bottom_list_raw:
        remapped = remap(name)
        if remapped and remapped not in seen_fixed:
            fixed_bottom_list.append(remapped)
            seen_fixed.add(remapped)

    left_totals = {
        name: sum(value for value in left_values[idx] if value is not None)
        for idx, name in enumerate(left_names)
    }
    sorted_left = sorted(left_names, key=lambda name: left_totals[name], reverse=True)
    pinned_present = [name for name in fixed_bottom_list if name in sorted_left]
    for name in pinned_present:
        sorted_left.remove(name)
    sorted_left = pinned_present + sorted_left
    left_index = {name: idx for idx, name in enumerate(left_names)}
    left_names = sorted_left
    left_values = [left_values[left_index[name]] for name in left_names]

    visual_top_to_bottom = list(reversed(left_names))
    right_index = {name: idx for idx, name in enumerate(right_names)}
    new_right_order = []
    for name in visual_top_to_bottom:
        if name in right_index:
            new_right_order.append(name)
    for name in right_names:
        if name not in new_right_order:
            new_right_order.append(name)
    right_names = new_right_order
    right_values = [right_values[right_index[name]] for name in right_names]

    custom_color = {remap(name): color for name, color in (chart.get("custom_color") or {}).items()}
    return {
        "title": chart.get("title", ""),
        "source": source_override if source_override is not None else chart.get("source"),
        "unit": _resolve_chart_unit(chart, unit_override),
        "q_count": q_count,
        "m_count": m_count,
        "w_count": w_count,
        "x_labels": x_labels,
        "left_names": left_names,
        "left_values": left_values,
        "right_names": right_names,
        "right_values": right_values,
        "custom_color": custom_color,
    }


def build_chart_spec(
    chart: Dict[str, Any],
    unit_override: Optional[str] = None,
    source_override: Optional[str] = None,
) -> Dict[str, Any]:
    if chart.get("subplot_format"):
        return spec_from_parsed_chart_subplots(
            chart, unit_override=unit_override, source_override=source_override
        )
    return spec_from_parsed_chart(
        chart, unit_override=unit_override, source_override=source_override
    )


_FONT_DIR = Path(__file__).resolve().parent / "fonts"

# Common Linux locations for the Noto Sans CJK TrueType Collection. We
# explicitly register these because matplotlib's built-in font scan
# misses TTC sub-faces on many distros — without this, Chinese/CJK
# glyphs render as the dreaded missing-glyph "tofu" boxes even though
# the font is technically installed system-wide.
#
# NOTE about TTC and matplotlib: the .ttc file packages multiple
# language-specific faces (JP/SC/TC/KR/HK), but matplotlib only
# registers the FIRST face from the collection (typically "Noto Sans
# CJK JP"). The good news: that first face's glyph table covers the
# entire unified CJK Ideograph block, so it renders Simplified Chinese
# correctly even though the registered name says "JP". Listing
# "Noto Sans CJK JP" in the font.sans-serif chain is therefore the
# canonical workaround.
_SYSTEM_CJK_TTCS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _register_bundled_fonts() -> None:
    """Add any OTF/TTF in the bundled fonts/ folder + system CJK fonts.

    Idempotent — calling twice doesn't double-register because the
    underlying ttflist dedupes by file path.
    """
    # Bundled Latin fonts (URW Gothic).
    if _FONT_DIR.is_dir():
        for path in sorted(_FONT_DIR.glob("*.[ot]tf")):
            try:
                font_manager.fontManager.addfont(str(path))
            except Exception:
                # Don't let a corrupt font abort chart rendering.
                pass

    # System CJK TTCs — explicitly registered so Chinese/Japanese/Korean
    # text in series names, doc titles, etc. renders correctly instead
    # of falling through to "tofu" boxes.
    for ttc_path in _SYSTEM_CJK_TTCS:
        if Path(ttc_path).is_file():
            try:
                font_manager.fontManager.addfont(ttc_path)
            except Exception:
                pass


_register_bundled_fonts()

# Make sure unicode minus signs render in the chosen font (otherwise
# matplotlib swaps in a non-ASCII U+2212 from STIX, which can also miss
# the font chain).
plt.rcParams["axes.unicode_minus"] = False

# Global font chain. Order matters: first hit wins per glyph.
#
# Why "Noto Sans CJK JP" appears in the chain even though we want
# Simplified Chinese: matplotlib only registers the FIRST face from a
# .ttc file, which on Debian/Ubuntu is the JP face. That face's glyph
# coverage includes the unified CJK Ideograph block and renders
# Simplified Chinese fine. Listing the SC name first is harmless (it
# silently fails) but the JP entry is what actually delivers CJK
# coverage on Linux. Don't remove it.
# IMPORTANT: matplotlib does per-glyph font fallback via ``font.family``
# (a LIST of family names), NOT via ``font.sans-serif``. The latter is
# only a single-font selector used to resolve the "sans-serif" generic
# family name to ONE concrete font. If we want a string like
# "Marlene 测试中文 郝佳佳" to render with Latin chars in URW Gothic and
# Han chars in Noto Sans CJK, we MUST list both families directly in
# ``font.family`` — otherwise mpl picks the first font (URW Gothic) and
# silently drops every glyph the font lacks (the user-reported bug).
#
# Order matters: matplotlib walks the list per glyph, taking the first
# font that has that codepoint. We put Latin fonts first so English
# stays in URW Gothic / Century Gothic, then list every plausible CJK
# font name so we get coverage on Linux/macOS/Windows.
plt.rcParams["font.family"] = [
    # Latin preference
    "Century Gothic",      # if installed (Windows/macOS), use the real thing
    "URW Gothic",          # bundled OFL clone — visually identical letterforms
    # CJK preference — list ALL plausible names so glyph fallback works
    # whatever the OS. Matplotlib silently skips names that don't resolve.
    "Microsoft YaHei",     # Windows native
    "PingFang SC",         # macOS native
    "Noto Sans CJK SC",    # rarely registered standalone, but try first
    "Noto Sans CJK JP",    # the face matplotlib actually loads from the
                            # Linux TTC; covers all unified CJK ideographs
    "WenQuanYi Zen Hei",   # alternate distro fallback
    "Source Han Sans SC",  # Adobe alt
    # Last-resort generic fallbacks
    "DejaVu Sans",
    "sans-serif",
]
# Keep ``font.sans-serif`` populated too for any code path (or third-
# party lib) that resolves the generic "sans-serif" family name. Same
# list — harmless duplication.
plt.rcParams["font.sans-serif"] = list(plt.rcParams["font.family"])
# Don't let matplotlib try to render Latin text in a unicode-mathematical
# style — keep it consistent with the chosen sans-serif family.
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "URW Gothic"
plt.rcParams["mathtext.it"] = "URW Gothic:italic"
plt.rcParams["mathtext.bf"] = "URW Gothic:bold"
# Suppress matplotlib's "findfont: Generic family 'sans-serif' not found"
# warnings that would otherwise spam stderr if the system is missing one
# of the names higher up in the chain — the chain itself handles fallback.
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


PALETTE = [
    # --- Main palette (brand colors, used bottom -> top in stacked order) ---
    "#0774C4",  # 1. Blue       (bottom-most series)
    "#48AA01",  # 2. Green
    "#FF7305",  # 3. Orange
    "#FEC101",  # 4. Yellow
    "#077C6F",  # 5. Teal
    "#A21098",  # 6. Magenta-purple
    # --- Reserve palette (used only if a chart has more than 6 series) ---
    # Hand-picked to harmonize with the main 6: saturated, similar value,
    # evenly distributed across the hue wheel so neighbors stay visually
    # distinct when stacked together.
    "#D81B60",  # 7.  Pink
    "#039BE5",  # 8.  Sky cyan-blue
    "#7CB342",  # 9.  Fresh green (lighter than #48AA01)
    "#5E35B1",  # 10. Deep violet
    "#00897B",  # 11. Teal-green
    "#F4511E",  # 12. Coral red
    "#6D4C41",  # 13. Warm brown
    "#546E7A",  # 14. Slate
    "#C0CA33",  # 15. Lime-yellow
]


@dataclass
class Series:
    name: str
    values: list[float]
    color: str | None = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_value(val: float, unit: str) -> str:
    """Format a numeric value for chart display.

    Per-chart Format directive (set in the Lark doc below the chart title):
        K     -> divide by 1,000, no decimals, no suffix
                 (e.g. 1,503,000 -> "1,503";  789,800 -> "790")
        M     -> divide by 1,000,000, no decimals, no suffix
                 (e.g. 2,150,000 -> "2";   12,500,000 -> "13")
        None  -> raw integer, comma-separated (e.g. "1,503,247")

    Legacy aliases ``Full`` and empty string behave the same as ``None``.
    """
    if unit == "K":
        return f"{val/1000:,.0f}"
    if unit == "M":
        return f"{val/1_000_000:,.0f}"
    return f"{val:,.0f}"


def _resolve_y_overlaps(items: list[dict], min_gap: float) -> None:
    """Push overlapping labels apart along Y while preserving the mean."""
    if not items:
        return
    items.sort(key=lambda x: x["adj_y"])
    for j in range(1, len(items)):
        if items[j]["adj_y"] - items[j - 1]["adj_y"] < min_gap:
            items[j]["adj_y"] = items[j - 1]["adj_y"] + min_gap
    orig_mean = np.mean([p["orig_y"] for p in items])
    adj_mean = np.mean([p["adj_y"] for p in items])
    shift = adj_mean - orig_mean
    for p in items:
        p["adj_y"] -= shift


# ---------------------------------------------------------------------------
# Color assignment — stable across grains
# ---------------------------------------------------------------------------

def assign_colors(names: list[str], palette: list[str] | None = None,
                   fixed: dict[str, str] | None = None) -> list[Series]:
    palette = palette or PALETTE
    fixed = fixed or {}
    out = []
    slot = 0
    for n in names:
        if n in fixed:
            color = fixed[n]
        else:
            color = palette[slot % len(palette)]
            slot += 1
        out.append(Series(name=n, values=[], color=color))
    return out


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _safe_float(v, null_as_zero=False) -> float:
    if v is None:
        return 0.0 if null_as_zero else np.nan
    if isinstance(v, str) and v.lower().strip() == "null":
        return 0.0 if null_as_zero else np.nan
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0 if null_as_zero else np.nan


def _plot_stacked_area(ax, x, labels, series_list, q_count, m_count, w_count, unit):
    if not series_list:
        ax.set_xlim(-0.5, len(x) + 2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold", color="#444444")
        ax.yaxis.set_visible(False)
        ax.tick_params(axis="x", length=0, pad=14)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        _decorate_sections(ax, q_count, m_count, w_count)
        return

    values_matrix = np.array([[_safe_float(v, null_as_zero=True) for v in s.values] for s in series_list], dtype=float)
    colors = [s.color for s in series_list]
    names = [s.name for s in series_list]

    ax.stackplot(x, values_matrix, colors=colors, alpha=1.0, linewidth=0, edgecolor="none")
    cumulative = np.cumsum(values_matrix, axis=0)
    lower_bounds = np.vstack([np.zeros(len(x)), cumulative[:-1]])
    col_totals = cumulative[-1]

    # NO white separator lines between stacked series — colors should touch
    # directly so each segment butts up against the next without any gap.
    # (Earlier versions drew thin white lines here; this was removed per the
    # reference design which keeps the stack as one continuous painted area.)

    # Show every per-cell label regardless of size. When labels overlap,
    # the larger (more influential) value should win — i.e. be drawn LAST so
    # it ends up on top in z-order.
    #
    # Two separate ranking rules are applied:
    #
    #   * Interior columns (everything except the rightmost W column):
    #     rank ascending by value; larger value gets higher zorder.
    #
    #   * Last (rightmost W) column ONLY: rank ascending by
    #     (prev_value - last_value), i.e. the period-over-period drop
    #     from the second-to-last column to the last column. The series
    #     with the LARGEST DROP gets the HIGHEST zorder so its label
    #     sits on top when overlaps occur. Rises (negative drop) sink
    #     to the bottom.
    last_col_idx = len(x) - 1
    interior_jobs = []   # (value, idx, xi, ypos, text, color)
    last_col_jobs = []   # (drop, idx, xi, ypos, text, color)
    # Minimum y position for labels — prevents them from colliding with x-axis ticks.
    min_label_y = col_totals.max() * 0.04
    for idx in range(len(series_list)):
        values = values_matrix[idx]
        bottoms = lower_bounds[idx]
        for xi_idx, (xi, value, bottom) in enumerate(zip(x, values, bottoms)):
            if value <= 0:
                continue
            pct = (value / col_totals[xi_idx]) * 100 if col_totals[xi_idx] > 0 else 0
            ypos = bottom + value / 2
            if ypos < min_label_y:
                ypos = min_label_y
            payload = (
                idx, xi, ypos,
                f"{format_value(value, unit)}\n({pct:.0f}%)",
                colors[idx],
            )
            if xi_idx == last_col_idx and last_col_idx >= 1:
                prev_value = float(values[last_col_idx - 1])
                drop = prev_value - float(value)
                last_col_jobs.append((drop, *payload))
            else:
                interior_jobs.append((float(value), *payload))

    # Interior columns: ascending by value (largest on top).
    interior_jobs.sort(key=lambda t: t[0])
    for rank, (_v, idx, xi, ypos, text, color) in enumerate(interior_jobs):
        ax.text(
            xi, ypos, text,
            ha="center", va="center", fontsize=18, color="white",
            fontweight="black",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                      edgecolor="none", alpha=1.0),
            zorder=5 + rank,
        )
    # Last column: ascending by (prev - last) so biggest drop ends up
    # with the highest zorder. Base rank above the interior pool so any
    # overlap between last-column and interior-column labels is decided
    # by the last-column ranking rule (last column always wins ties).
    # Nudge labels slightly left so they sit fully inside the stacked area.
    last_col_jobs.sort(key=lambda t: t[0])
    base = 5 + len(interior_jobs)
    for rank, (_d, idx, xi, ypos, text, color) in enumerate(last_col_jobs):
        ax.text(
            xi - 0.28, ypos, text,
            ha="center", va="center", fontsize=18, color="white",
            fontweight="black",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                      edgecolor="none", alpha=1.0),
            zorder=base + rank,
        )

    # Find the rightmost column that has data in ANY series so all
    # right-edge labels align to the same x-coordinate (end of subplot).
    global_last_valid = 0
    for idx in range(len(series_list)):
        raw_valid = [i for i, v in enumerate(series_list[idx].values)
                     if not np.isnan(_safe_float(v, null_as_zero=False))]
        if raw_valid:
            global_last_valid = max(global_last_valid, raw_valid[-1])

    name_points = []
    for idx, s in enumerate(series_list):
        # Anchor the right-edge label to the GLOBAL last column (not the
        # series-specific last column) so all labels align at the subplot edge.
        # The y position is still based on the series' value at that column
        # (which may be 0/null, so we fall back to the series' last-valid y).
        raw_valid = [i for i, v in enumerate(s.values)
                     if not np.isnan(_safe_float(v, null_as_zero=False))]
        if not raw_valid:
            continue
        series_last_valid = raw_valid[-1]
        # y position: use the value at global_last_valid if it exists,
        # otherwise fall back to the series' own last valid position
        if global_last_valid < len(s.values):
            y_pos = lower_bounds[idx, global_last_valid] + values_matrix[idx, global_last_valid] / 2
        else:
            y_pos = lower_bounds[idx, series_last_valid] + values_matrix[idx, series_last_valid] / 2
        name_points.append({"orig_y": y_pos, "adj_y": y_pos,
                             "text": names[idx], "color": colors[idx],
                             "x_offset": 0.2, "x_pos": x[global_last_valid]})
    _resolve_y_overlaps(name_points, col_totals.max() * 0.05)
    for p in name_points:
        ax.text(p["x_pos"] + p["x_offset"], p["adj_y"], p["text"], fontsize=20,
                color=p["color"], va="center", fontweight="bold")

    for xi, total in zip(x, col_totals):
        # Lift the total label well above the stack so it doesn't visually
        # collide with the topmost band's value/percent label. 5% of the
        # global column max gives a clear gap at the chosen 18pt size.
        ax.text(xi, total + col_totals.max() * 0.05, format_value(total, unit),
                ha="center", va="bottom", fontsize=18, color="#222222",
                fontweight="bold")

    ax.set_xlim(-0.5, len(x) + 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold",
                       color="#444444")
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=14)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    _decorate_sections(ax, q_count, m_count, w_count)


def _plot_smooth_line(ax, x, y, line_width, line_color, zorder):
    """Plot a smooth line handling NaNs by splitting into valid segments."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = ~np.isnan(y_arr)
    
    if not valid.any():
        return
        
    # Split into contiguous valid segments
    segments = []
    current_x = []
    current_y = []
    for xi, yi, is_valid in zip(x_arr, y_arr, valid):
        if is_valid:
            current_x.append(xi)
            current_y.append(yi)
        else:
            if current_x:
                segments.append((current_x, current_y))
                current_x = []
                current_y = []
    if current_x:
        segments.append((current_x, current_y))
        
    for seg_x, seg_y in segments:
        if len(seg_x) >= 2:
            xs_smooth = np.linspace(seg_x[0], seg_x[-1], max(60, len(seg_x) * 20))
            spl = PchipInterpolator(seg_x, seg_y)
            ys_smooth = spl(xs_smooth)
            ax.plot(xs_smooth, ys_smooth, linewidth=line_width, color=line_color, zorder=zorder)
        elif len(seg_x) == 1:
            ax.plot(seg_x, seg_y, marker='o', color=line_color, markersize=5, zorder=zorder)

def _plot_yoy_lines(ax, x, labels, series_list, q_count, m_count, w_count):
    """YoY line chart with outlier-aware per-series compression.

    Design goal for charts like Chart 6:
      * Keep NORMAL series (roughly <= 800%) visually expressive
      * Compress only the EXTREME outlier series (e.g. Gaming at 10k%~90k%)
      * Preserve the original UI / labels / smoothing as much as possible

    Strategy:
      * Detect a large outlier threshold from the non-overall series distribution
      * Series whose max(|v|) is above that threshold get stronger log compression
      * Normal series get a much lighter transform so they don't collapse into
        almost-flat lines
    """
    import math

    if not series_list or len(x) == 0:
        _plot_yoy_lines_shared(ax, x, labels, series_list, q_count, m_count, w_count)
        return

    n_cols = len(x)
    n_series = len(series_list)

    all_vals = []
    abs_vals = []
    for s in series_list:
        for v in s.values:
            fv = _safe_float(v)
            if np.isnan(fv):
                continue
            all_vals.append(fv)
            abs_vals.append(abs(fv))
    if not all_vals:
        return

    PAD_LO, PAD_HI = 0.04, 0.96
    
    # Quantile mapping (Histogram equalization)
    valid_vals = [v for v in all_vals if not np.isnan(v)]
    sorted_vals = sorted(valid_vals)
    import bisect
    def T_any(v: float) -> float:
        if np.isnan(v): return np.nan
        idx = bisect.bisect_left(sorted_vals, v)
        return float(idx) / max(1, len(sorted_vals) - 1)
    
    transforms = [T_any for _ in series_list]
    
    Ts = []
    for i in range(n_cols):
        row = []
        for j in range(n_series):
            v = _safe_float(series_list[j].values[i])
            row.append(np.nan if np.isnan(v) else transforms[j](v))
        Ts.append(row)
        
    flat = [t for col in Ts for t in col if not np.isnan(t)]
    if not flat:
        return
    t_min = min(flat)
    t_max = max(flat)
    t_min = min(t_min, 0.0)
    t_max = max(t_max, 0.0)
    span = t_max - t_min if t_max > t_min else 1.0
    t_min -= span * 0.05
    t_max += span * 0.05

    def to_y(t: float) -> float:
        if np.isnan(t): return np.nan
        return PAD_LO + (t - t_min) / (t_max - t_min) * (PAD_HI - PAD_LO)

    ys = [[to_y(Ts[i][j]) for j in range(n_series)] for i in range(n_cols)]

    xs_arr = np.asarray(x, dtype=float)
    for j, s in enumerate(series_list):
        is_overall = s.name.lower() == 'overall'
        line_color = '#000000' if is_overall else s.color
        line_width = 4.0 if is_overall else 3.0
        zorder = 10 if is_overall else 5
        line_y = np.asarray([ys[i][j] for i in range(n_cols)], dtype=float)
        _plot_smooth_line(ax, xs_arr, line_y, line_width, line_color, zorder)

    for i in range(n_cols):
        col_points = []
        for j, s in enumerate(series_list):
            val = _safe_float(s.values[i])
            if np.isnan(val):
                continue
            y_pos = ys[i][j]
            col_points.append({
                'orig_y': y_pos,
                'adj_y': y_pos,
                'text': f'{val:,.0f}%',
                'color': '#000000' if s.name.lower() == 'overall' else s.color,
            })
        if col_points:
            _resolve_y_overlaps(col_points, 0.038)
            for p in col_points:
                ax.text(x[i], p['adj_y'], p['text'], fontsize=18,
                        color=p['color'], ha='center', va='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.95),
                        zorder=15)

    last_idx = n_cols - 1
    # Find global last valid column — every series-name label is x-anchored
    # to this column so all item titles line up vertically at the right edge
    # of the subplot (a hard requirement: "all data title need to be at the
    # end of the subplot"). For the y-position we still use each series's
    # OWN last valid y-value so a series that ends early doesn't fall back
    # to NaN (which would hide the label entirely — that's the "no item
    # title" bug from the previous round).
    global_last_valid = 0
    for j, s in enumerate(series_list):
        valid_indices = [i for i in range(n_cols) if not np.isnan(ys[i][j])]
        if valid_indices:
            global_last_valid = max(global_last_valid, valid_indices[-1])

    name_points = []
    for j, s in enumerate(series_list):
        valid_indices = [i for i in range(n_cols) if not np.isnan(ys[i][j])]
        if not valid_indices:
            continue
        # x: pinned to the global last valid column (right edge of subplot).
        # y: this series's own last valid y, so the label never lands on NaN
        # and stays adjacent to where the line actually ends.
        series_last_valid = valid_indices[-1]
        y_pos = ys[series_last_valid][j]
        # Use the series's own last value for the bbox-width estimate; this
        # only affects how far right the label is offset from the column.
        val = _safe_float(s.values[series_last_valid])
        last_text = f'{val:,.0f}%'
        name_points.append({
            'orig_y': y_pos, 'adj_y': y_pos, 'text': s.name,
            'color': '#000000' if s.name.lower() == 'overall' else s.color,
            'x_offset': 0.1 + len(last_text) * 0.06,
            'x_pos': x[global_last_valid],
        })
    if name_points:
        # Normalize all x_offsets to the maximum so every name label lands
        # on the SAME final x-coordinate — i.e. all item titles line up
        # vertically at the right edge of the subplot. Per-point offsets
        # were originally sized to clear each series's own rightmost value
        # bubble; pinning to the max preserves that clearance for the
        # widest bubble while keeping shorter ones aligned.
        max_x_offset = max(p['x_offset'] for p in name_points)
        for p in name_points:
            p['x_offset'] = max_x_offset
        _resolve_y_overlaps(name_points, 0.048)
        for p in name_points:
            ax.text(p['x_pos'] + p['x_offset'], p['adj_y'], p['text'], fontsize=20,
                    color=p['color'], va='center', fontweight='bold', zorder=20)

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, n_cols + 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight='bold', color='#444444')
    ax.yaxis.set_visible(False)
    ax.tick_params(axis='x', length=0, pad=8)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    _decorate_sections(ax, q_count, m_count, w_count)


def _plot_yoy_lines_shared(ax, x, labels, series_list, q_count, m_count, w_count):
    """Original single-y-axis YoY chart, used when only one section is
    present (so a broken axis would be silly)."""
    if not series_list:
        ax.set_xlim(-0.5, len(x) + 2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold", color="#444444")
        ax.yaxis.set_visible(False)
        ax.tick_params(axis="x", length=0, pad=14)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        _decorate_sections(ax, q_count, m_count, w_count)
        return
    all_y = np.concatenate([[_safe_float(v) for v in s.values] for s in series_list])
    all_y = all_y[~np.isnan(all_y)]
    y_range = all_y.max() - all_y.min() if len(all_y) > 0 else 1.0
    # Zero baseline intentionally NOT drawn (matches symlog variant above).

    for s in series_list:
        y = np.array([_safe_float(v) for v in s.values], dtype=float)
        is_overall = s.name.lower() == "overall"
        _plot_smooth_line(ax, x, y, 4.0 if is_overall else 3.0, "#000000" if is_overall else s.color, 10 if is_overall else 5)

    min_gap_yoy = y_range * 0.035
    for i in range(len(x)):
        col_points = []
        for s in series_list:
            val = _safe_float(s.values[i])
            if np.isnan(val):
                continue
            col_points.append({
                "orig_y": val, "adj_y": val, "text": f"{val:,.0f}%",
                "color": "#000000" if s.name.lower() == "overall" else s.color,
            })
        if col_points:
            _resolve_y_overlaps(col_points, min_gap_yoy)
            for p in col_points:
                ax.text(x[i], p["adj_y"], p["text"], fontsize=18, color=p["color"],
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.9),
                        zorder=15)

    # Find global last valid column. Every series-name label is x-anchored
    # to this column so all item titles line up at the right edge of the
    # subplot. y still uses each series's own last valid value so the label
    # never collapses to NaN (= invisible) when a series ends early.
    global_last_valid = 0
    for s in series_list:
        valid_indices = [i for i, v in enumerate(s.values) if not np.isnan(_safe_float(v))]
        if valid_indices:
            global_last_valid = max(global_last_valid, valid_indices[-1])

    name_points = []
    for s in series_list:
        valid_indices = [i for i, v in enumerate(s.values) if not np.isnan(_safe_float(v))]
        if not valid_indices:
            continue
        # x: pinned to global_last_valid (right edge). y: this series's own
        # last valid value so it never lands on NaN.
        series_last_valid = valid_indices[-1]
        last_val = _safe_float(s.values[series_last_valid])
        last_text = f'{last_val:,.0f}%'
        name_points.append({
            "orig_y": last_val, "adj_y": last_val, "text": s.name,
            "color": "#000000" if s.name.lower() == "overall" else s.color,
            "x_offset": 0.1 + len(last_text) * 0.06,
            "x_pos": x[global_last_valid],
        })
    if name_points:
        # Normalize x_offset to the max so all item titles align at the
        # same final x (right edge of subplot, same column line).
        max_x_offset = max(p["x_offset"] for p in name_points)
        for p in name_points:
            p["x_offset"] = max_x_offset
        _resolve_y_overlaps(name_points, y_range * 0.05)
        for p in name_points:
            ax.text(p["x_pos"] + p["x_offset"], p["adj_y"], p["text"], fontsize=20,
                    color=p["color"], va="center", fontweight="bold",
                    zorder=20)

    ax.set_xlim(-0.5, len(x) + 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold",
                       color="#444444")
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=14)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    _decorate_sections(ax, q_count, m_count, w_count)


def _decorate_sections(ax, q_count, m_count, w_count):
    boundaries = []
    if q_count > 0 and m_count > 0:
        boundaries.append(q_count - 0.5)
    if m_count > 0 and w_count > 0:
        boundaries.append(q_count + m_count - 0.5)
    for b in boundaries:
        ax.axvline(b, color="#BDBDBD", linestyle="--", linewidth=1.5, zorder=10)

    total = q_count + m_count + w_count
    if total <= 0:
        return
    sections = []
    if q_count > 0:
        # Center "By Q" at the midpoint between the first and last Q data points.
        sections.append(((q_count - 1) / 2, "By Q"))
    if m_count > 0:
        # Center "By M" between first and last M data point (offset past Q).
        sections.append((q_count + (m_count - 1) / 2, "By M"))
    if w_count > 0:
        # Center "By W" between first and last W data point.
        sections.append((q_count + m_count + (w_count - 1) / 2, "By W"))
    # Use a blended transform: x is in DATA coords (so the label sits at the
    # true horizontal middle of each section's data range), y is in
    # axes-fraction (so the label always floats just above the chart top).
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for c, n in sections:
        ax.text(c, 1.05, n, transform=trans,
                ha="center", va="bottom", fontsize=20, fontweight="bold",
                color="#000000")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _plot_stacked_area_raw(ax, x, labels, series_list, q_count, m_count, w_count):
    """Stacked-area subplot for non-revenue measures (e.g. ARPU).

    Same visual contract as ``_plot_stacked_area`` but value labels are
    rendered as raw integers (no K/M division, no per-cell ``%`` row).
    Use this for subplot types like ``stackplot`` whose name does NOT
    contain the revenue/GMV hint — the user wants the raw measured value
    shown, not a divided/scaled version.
    """
    if not series_list:
        ax.set_xlim(-0.5, len(x) + 2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold", color="#444444")
        ax.yaxis.set_visible(False)
        ax.tick_params(axis="x", length=0, pad=14)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        _decorate_sections(ax, q_count, m_count, w_count)
        return
        
    values_matrix = np.array([[_safe_float(v, null_as_zero=True) for v in s.values] for s in series_list], dtype=float)
    colors = [s.color for s in series_list]
    names = [s.name for s in series_list]

    ax.stackplot(x, values_matrix, colors=colors, alpha=1.0, linewidth=0,
                 edgecolor="none")
    cumulative = np.cumsum(values_matrix, axis=0)
    lower_bounds = np.vstack([np.zeros(len(x)), cumulative[:-1]])
    col_totals = cumulative[-1]

    # See `_plot_stacked_area` for the rationale. Last (rightmost W)
    # column gets a separate ranking by (prev - last) so the biggest
    # period-over-period drop wins z-order ties.
    last_col_idx = len(x) - 1
    interior_jobs = []
    last_col_jobs = []
    for idx in range(len(series_list)):
        values = values_matrix[idx]
        bottoms = lower_bounds[idx]
        for xi_idx, (xi, value, bottom) in enumerate(zip(x, values, bottoms)):
            if value <= 0:
                continue
            payload = (
                idx, xi, bottom + value / 2,
                f"{value:,.0f}",
                colors[idx],
            )
            if xi_idx == last_col_idx and last_col_idx >= 1:
                prev_value = float(values[last_col_idx - 1])
                drop = prev_value - float(value)
                last_col_jobs.append((drop, *payload))
            else:
                interior_jobs.append((float(value), *payload))
    interior_jobs.sort(key=lambda t: t[0])
    for rank, (_v, idx, xi, ypos, text, color) in enumerate(interior_jobs):
        ax.text(
            xi, ypos, text,
            ha="center", va="center", fontsize=18, color="white",
            fontweight="black",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                      edgecolor="none", alpha=1.0),
            zorder=5 + rank,
        )
    last_col_jobs.sort(key=lambda t: t[0])
    base = 5 + len(interior_jobs)
    for rank, (_d, idx, xi, ypos, text, color) in enumerate(last_col_jobs):
        ax.text(
            xi, ypos, text,
            ha="center", va="center", fontsize=18, color="white",
            fontweight="black",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                      edgecolor="none", alpha=1.0),
            zorder=base + rank,
        )

    # Find the rightmost column that has data in ANY series so all
    # right-edge labels align to the same x-coordinate (end of subplot).
    global_last_valid = 0
    for idx in range(len(series_list)):
        raw_valid = [i for i, v in enumerate(series_list[idx].values)
                     if not np.isnan(_safe_float(v, null_as_zero=False))]
        if raw_valid:
            global_last_valid = max(global_last_valid, raw_valid[-1])

    name_points = []
    for idx, s in enumerate(series_list):
        # Anchor each label to the GLOBAL last column for consistent alignment.
        raw_valid = [i for i, v in enumerate(s.values)
                     if not np.isnan(_safe_float(v, null_as_zero=False))]
        if not raw_valid:
            continue
        series_last_valid = raw_valid[-1]
        if global_last_valid < len(s.values):
            y_pos = lower_bounds[idx, global_last_valid] + values_matrix[idx, global_last_valid] / 2
        else:
            y_pos = lower_bounds[idx, series_last_valid] + values_matrix[idx, series_last_valid] / 2
        name_points.append({"orig_y": y_pos, "adj_y": y_pos,
                             "text": names[idx], "color": colors[idx],
                             "x_offset": 0.2, "x_pos": x[global_last_valid]})
    if name_points:
        _resolve_y_overlaps(name_points, max(col_totals.max() * 0.05, 1e-9))
        for p in name_points:
            ax.text(p["x_pos"] + p["x_offset"], p["adj_y"], p["text"], fontsize=20,
                    color=p["color"], va="center", fontweight="bold")

    for xi, total in zip(x, col_totals):
        ax.text(xi, total + max(col_totals.max() * 0.05, 1e-9),
                f"{total:,.0f}",
                ha="center", va="bottom", fontsize=18, color="#222222",
                fontweight="bold")

    ax.set_xlim(-0.5, len(x) + 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold",
                       color="#444444")
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=14)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    _decorate_sections(ax, q_count, m_count, w_count)


def _plot_stacked_area_pct(ax, x, labels, series_list, q_count, m_count, w_count):
    """Stacked-area subplot whose cell labels are ``XX%`` (no division).

    Visual contract is identical to ``_plot_stacked_area_raw`` — same
    stack-from-zero geometry, same series-name + column-total layout —
    but each per-cell label is rendered as ``f"{value:,.0f}%"`` and the
    column-total label is also rendered as ``%``.

    Use this when the user supplies ``Subplot k format = percentage - ...``
    on a stackplot subplot whose underlying values are already percent
    points (e.g. share-of-something pre-computed by the pivot).
    """
    values_matrix = np.array([[_safe_float(v, null_as_zero=True) for v in s.values] for s in series_list], dtype=float)
    colors = [s.color for s in series_list]
    names = [s.name for s in series_list]

    ax.stackplot(x, values_matrix, colors=colors, alpha=1.0, linewidth=0,
                 edgecolor="none")
    cumulative = np.cumsum(values_matrix, axis=0)
    lower_bounds = np.vstack([np.zeros(len(x)), cumulative[:-1]])
    col_totals = cumulative[-1]

    # See `_plot_stacked_area` for the rationale. Last (rightmost W)
    # column gets a separate ranking by (prev - last) so the biggest
    # period-over-period drop wins z-order ties.
    last_col_idx = len(x) - 1
    interior_jobs = []
    last_col_jobs = []
    for idx in range(len(series_list)):
        values = values_matrix[idx]
        bottoms = lower_bounds[idx]
        for xi_idx, (xi, value, bottom) in enumerate(zip(x, values, bottoms)):
            if value <= 0:
                continue
            payload = (
                idx, xi, bottom + value / 2,
                f"{value:,.0f}%",
                colors[idx],
            )
            if xi_idx == last_col_idx and last_col_idx >= 1:
                prev_value = float(values[last_col_idx - 1])
                drop = prev_value - float(value)
                last_col_jobs.append((drop, *payload))
            else:
                interior_jobs.append((float(value), *payload))
    interior_jobs.sort(key=lambda t: t[0])
    for rank, (_v, idx, xi, ypos, text, color) in enumerate(interior_jobs):
        ax.text(
            xi, ypos, text,
            ha="center", va="center", fontsize=18, color="white",
            fontweight="black",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                      edgecolor="none", alpha=1.0),
            zorder=5 + rank,
        )
    last_col_jobs.sort(key=lambda t: t[0])
    base = 5 + len(interior_jobs)
    for rank, (_d, idx, xi, ypos, text, color) in enumerate(last_col_jobs):
        ax.text(
            xi, ypos, text,
            ha="center", va="center", fontsize=18, color="white",
            fontweight="black",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                      edgecolor="none", alpha=1.0),
            zorder=base + rank,
        )

    # Find the rightmost column that has data in ANY series so all
    # right-edge labels align to the same x-coordinate (end of subplot).
    global_last_valid = 0
    for idx in range(len(series_list)):
        raw_valid = [i for i, v in enumerate(series_list[idx].values)
                     if not np.isnan(_safe_float(v, null_as_zero=False))]
        if raw_valid:
            global_last_valid = max(global_last_valid, raw_valid[-1])

    name_points = []
    for idx, s in enumerate(series_list):
        # Anchor each label to the GLOBAL last column for consistent alignment.
        raw_valid = [i for i, v in enumerate(s.values)
                     if not np.isnan(_safe_float(v, null_as_zero=False))]
        if not raw_valid:
            continue
        series_last_valid = raw_valid[-1]
        if global_last_valid < len(s.values):
            y_pos = lower_bounds[idx, global_last_valid] + values_matrix[idx, global_last_valid] / 2
        else:
            y_pos = lower_bounds[idx, series_last_valid] + values_matrix[idx, series_last_valid] / 2
        name_points.append({"orig_y": y_pos, "adj_y": y_pos,
                             "text": names[idx], "color": colors[idx],
                             "x_offset": 0.2, "x_pos": x[global_last_valid]})
    if name_points:
        _resolve_y_overlaps(name_points, max(col_totals.max() * 0.05, 1e-9))
        for p in name_points:
            ax.text(p["x_pos"] + p["x_offset"], p["adj_y"], p["text"], fontsize=20,
                    color=p["color"], va="center", fontweight="bold")

    for xi, total in zip(x, col_totals):
        ax.text(xi, total + max(col_totals.max() * 0.05, 1e-9),
                f"{total:,.0f}%",
                ha="center", va="bottom", fontsize=18, color="#222222",
                fontweight="bold")

    ax.set_xlim(-0.5, len(x) + 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold",
                       color="#444444")
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=14)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    _decorate_sections(ax, q_count, m_count, w_count)


def _plot_yoy_lines_numeric(ax, x, labels, series_list, q_count, m_count, w_count, value_format):
    """Line chart variant whose per-datapoint labels are numeric (K/M/raw).

    Same geometry / smoothing / overlap-resolve / right-edge name layout as
    ``_plot_yoy_lines``, but each label is rendered via
    ``format_value(val, value_format)`` instead of ``f"{val:,.0f}%"``.

    ``value_format`` is one of:
        - ``"K"`` -> divide by 1,000, integer
        - ``"M"`` -> divide by 1,000,000, integer
        - ``"raw"`` (or anything else) -> raw integer with thousands separator

    Used when the user supplies ``Subplot k format = number - <unit>`` on a
    line subplot whose underlying values are absolute counts/amounts (e.g.
    Active Group Count) rather than YoY percentages.
    """
    import math

    if not series_list or len(x) == 0:
        _plot_yoy_lines_shared(ax, x, labels, series_list, q_count, m_count, w_count)
        return

    n_cols = len(x)
    n_series = len(series_list)

    abs_vals = []
    all_vals = []
    for s in series_list:
        for v in s.values:
            fv = _safe_float(v, null_as_zero=True)
            if np.isnan(fv):
                continue
            all_vals.append(fv)
            abs_vals.append(abs(fv))
    if not all_vals:
        _plot_yoy_lines_shared(ax, x, labels, series_list, q_count, m_count, w_count)
        return

    abs_sorted = sorted(abs_vals)
    median_abs = abs_sorted[len(abs_sorted) // 2]
    max_abs = abs_sorted[-1]
    linthresh = max(50.0, 0.5 * median_abs)
    linthresh = min(linthresh, max(50.0, 0.5 * max_abs))

    def T(v: float) -> float:
        if v == 0.0:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        return sign * math.log10(1.0 + abs(v) / linthresh)

    PAD_LO, PAD_HI = 0.04, 0.96

    Ts = []
    for i in range(n_cols):
        row = []
        for j, s in enumerate(series_list):
            v = _safe_float(s.values[i], null_as_zero=True)
            row.append(np.nan if np.isnan(v) else T(v))
        Ts.append(row)
        
    flat = [t for col in Ts for t in col if not np.isnan(t)]
    if not flat:
        return
    t_min = min(flat)
    t_max = max(flat)

    t_min = min(t_min, 0.0)
    t_max = max(t_max, 0.0)
    span = t_max - t_min if t_max > t_min else 1.0
    t_min -= span * 0.05
    t_max += span * 0.05

    def to_y(t: float) -> float:
        if np.isnan(t): return np.nan
        return PAD_LO + (t - t_min) / (t_max - t_min) * (PAD_HI - PAD_LO)

    ys = [[to_y(Ts[i][j]) for j in range(n_series)] for i in range(n_cols)]

    # Mode picker for format_value (which expects "K", "M", or anything else=raw).
    fmt_unit = value_format if value_format in ("K", "M") else "None"

    xs_arr = np.asarray(x, dtype=float)
    for j, s in enumerate(series_list):
        is_overall = s.name.lower() == "overall"
        line_color = "#000000" if is_overall else s.color
        line_width = 4.0 if is_overall else 3.0
        zorder = 10 if is_overall else 5
        line_y = np.asarray([ys[i][j] for i in range(n_cols)], dtype=float)
        _plot_smooth_line(ax, xs_arr, line_y, line_width, line_color, zorder)

    for i in range(n_cols):
        col_points = []
        for j, s in enumerate(series_list):
            val = _safe_float(s.values[i], null_as_zero=True)
            if np.isnan(val):
                continue
            y_pos = ys[i][j]
            col_points.append({
                "orig_y": y_pos,
                "adj_y": y_pos,
                "text": format_value(val, fmt_unit),
                "color": "#000000" if s.name.lower() == "overall" else s.color,
            })
        if col_points:
            _resolve_y_overlaps(col_points, 0.038)
            for p in col_points:
                ax.text(x[i], p["adj_y"], p["text"], fontsize=18,
                        color=p["color"], ha="center", va="center",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none", alpha=0.95),
                        zorder=15)

    last_idx = n_cols - 1
    # Find global last valid column across all series for consistent alignment
    global_last_valid = 0
    for j, s in enumerate(series_list):
        valid_indices = [i for i, v in enumerate(s.values)
                         if not np.isnan(_safe_float(v, null_as_zero=False))]
        if valid_indices:
            global_last_valid = max(global_last_valid, valid_indices[-1])

    name_points = []
    for j, s in enumerate(series_list):
        valid_indices = [i for i, v in enumerate(s.values)
                         if not np.isnan(_safe_float(v, null_as_zero=False))]
        if not valid_indices:
            continue
        # x: pinned to global_last_valid (right edge of subplot) so all item
        # titles line up vertically. y: this series's own last valid y so
        # the label stays adjacent to where the line actually ends and
        # never falls on a NaN cell.
        series_last_valid = valid_indices[-1]
        y_pos = ys[series_last_valid][j]
        val = _safe_float(s.values[series_last_valid], null_as_zero=True)
        last_text = format_value(val, fmt_unit)
        name_points.append({
            "orig_y": y_pos, "adj_y": y_pos, "text": s.name,
            "color": "#000000" if s.name.lower() == "overall" else s.color,
            "x_offset": 0.1 + len(last_text) * 0.06,
            "x_pos": x[global_last_valid],
        })
    if name_points:
        # Normalize x_offset so all item titles share the same final x
        # (right edge of subplot, same column line as the rest of the data).
        max_x_offset = max(p["x_offset"] for p in name_points)
        for p in name_points:
            p["x_offset"] = max_x_offset
        _resolve_y_overlaps(name_points, 0.048)
        for p in name_points:
            ax.text(p["x_pos"] + p["x_offset"], p["adj_y"], p["text"], fontsize=20,
                    color=p["color"], va="center", fontweight="bold",
                    zorder=20)

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, n_cols + 2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=16, fontweight="bold",
                       color="#444444")
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=14)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    _decorate_sections(ax, q_count, m_count, w_count)


def render_multisubplot_chart(spec: dict, output_path: str | Path, image_format: str = None, dpi: int = 220) -> Path:
    """Render a chart whose layout is driven by a ``Subplot Format`` directive.
    import pathlib
    out_p = pathlib.Path(output_path)
    if image_format is None:
        suffix = out_p.suffix.lower().lstrip(".")
        image_format = "jpeg" if suffix == "jpg" else suffix or "png"
    image_format = image_format.lower()
    if image_format == "jpg":
        image_format = "jpeg"

    Spec shape (in addition to the shared keys ``title``, ``source``,
    ``q_count``, ``m_count``, ``w_count``, ``x_labels``, ``custom_color``)::

        {
            ...,
            "subplots": [
                {
                    "name":         str,                 # default subtitle (measure column)
                    "title_override": str | None,        # `Subplot k Title = ...` override
                    "type":         "stackplot" | "line",
                    "names":        [str],               # series names
                    "values":       [[float] of len Q+M+W] per series,
                    "value_format": "K" | "M" | "raw" | "pct",
                    "include_overall": bool,             # only honored on `line`
                    "overall_values": [float] | None,    # synthesized Overall
                },
                ...
            ],
        }

    Routing rules per subplot:
      * ``type=stackplot`` + ``value_format`` in (K, M) -> ``_plot_stacked_area``
        with the matching unit (revenue / GMV measures honor the chart Format
        directive K|M|None).
      * ``type=stackplot`` + ``value_format=raw`` -> ``_plot_stacked_area_raw``
        (no K/M division, no per-cell ``%`` line).
      * ``type=line`` -> ``_plot_yoy_lines`` (always renders ``XX%``).
        ``Overall`` is appended only when ``include_overall`` is True.
    """
    q_count = spec["q_count"]
    m_count = spec["m_count"]
    w_count = spec["w_count"]
    x_labels = spec["x_labels"]
    total_pts = q_count + m_count + w_count
    if len(x_labels) != total_pts:
        raise ValueError(
            f"x_labels has {len(x_labels)} entries but Q+M+W = {total_pts}"
        )

    custom_color = spec.get("custom_color") or {}
    subplots = spec["subplots"]
    n_panels = len(subplots)

    max_label_len = 0
    for sp in subplots:
        names = sp["names"] + (["Overall"] if sp.get("include_overall") else [])
        for n in names:
            max_label_len = max(max_label_len, len(n))
            
    # Approx 0.16 inches per char for 20pt bold text
    label_space_inches = max(0.8, max_label_len * 0.16)

    x = np.arange(total_pts)
    COLUMN_WIDTH_INCHES = 1.4
    # Width scales with both panel count and column count. The base
    # accounts for right-edge series-name labels + outer gutter; we
    # then add COLUMN_WIDTH_INCHES per data column per panel so each panel
    # has comparable column width regardless of N.
    per_panel_width = max(8.0, 2.5 + total_pts * COLUMN_WIDTH_INCHES)
    dynamic_width = max(18.0, per_panel_width * n_panels + label_space_inches)
    fig, axes = plt.subplots(1, n_panels, figsize=(dynamic_width, 16), dpi=220)
    # Increase horizontal gap between subplots by 3x (default wspace ~0.2 → 0.6)
    fig.subplots_adjust(wspace=0.6)
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # Build a shared color map across all subplots so a series name keeps
    # the same color in every subplot it appears in.
    seen_colors: dict[str, str] = dict(custom_color)
    palette_iter = iter(PALETTE)

    def color_for(name: str) -> str:
        if name in seen_colors:
            return seen_colors[name]
        # walk the palette but skip colors already in seen_colors values
        used = set(seen_colors.values())
        for c in PALETTE:
            if c not in used:
                seen_colors[name] = c
                return c
        # fallback — wrap
        c = PALETTE[len(seen_colors) % len(PALETTE)]
        seen_colors[name] = c
        return c

    for sp_idx, (ax, sp) in enumerate(zip(axes, subplots)):
        names = sp["names"]
        values = sp["values"]
        sp_type = sp["type"]
        value_format = sp.get("value_format", "raw")

        series_list = []
        for i, n in enumerate(names):
            s = Series(name=n, values=list(values[i]), color=color_for(n))
            series_list.append(s)

        if sp_type == "stackplot":
            if value_format in ("K", "M"):
                _plot_stacked_area(ax, x, x_labels, series_list,
                                    q_count, m_count, w_count, value_format)
            elif value_format == "pct":
                _plot_stacked_area_pct(ax, x, x_labels, series_list,
                                        q_count, m_count, w_count)
            else:
                _plot_stacked_area_raw(ax, x, x_labels, series_list,
                                        q_count, m_count, w_count)
        elif sp_type == "line":
            # Optionally append Overall (synthesized) drawn black.
            if sp.get("include_overall") and sp.get("overall_values"):
                ov = Series(name="Overall",
                            values=list(sp["overall_values"]),
                            color="#000000")
                series_list.append(ov)
            if value_format == "pct":
                _plot_yoy_lines(ax, x, x_labels, series_list,
                                q_count, m_count, w_count)
            else:
                # Line chart with absolute number labels (rare).
                _plot_yoy_lines_numeric(ax, x, x_labels, series_list,
                                         q_count, m_count, w_count, value_format)
        else:
            raise ValueError(f"Unknown subplot type {sp_type!r}")

    # ------------------------------------------------------------------
    # Tighten xlim on each axis: reduce the right-side padding so there is
    # minimal blank space between panels.  The individual _plot_* functions
    # set xlim to len(x)+2.0 for series-name labels; we shrink to +0.6.
    for ax in axes:
        ax.set_xlim(-0.3, total_pts - 1 + 0.6)

    # ------------------------------------------------------------------
    # Title (same logic as the 2-subplot variant — yellow highlight on
    # the pre-"|" segment, full title centered horizontally).
    # ------------------------------------------------------------------
    TITLE_FONTSIZE = max(26, int(round(26 + (dynamic_width - 20.0) * 1.0)))
    TITLE_FONTSIZE = min(TITLE_FONTSIZE, 48)
    title = spec.get("title", "")
    title_bottom_frac = 0.95
    if "|" in title:
        parts = title.split("|", 1)
        part1 = parts[0].strip()
        part2_full = f" | {parts[1].strip()}"
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_w = fig.bbox.width
        fig_h = fig.bbox.height
        full_text = fig.text(0.5, 0.96, part1 + part2_full,
                              fontsize=TITLE_FONTSIZE, fontweight="bold",
                              ha="center", va="top")
        full_bbox = full_text.get_window_extent(renderer=renderer)
        full_left_frac = full_bbox.x0 / fig_w
        title_bottom_frac = full_bbox.y0 / fig_h
        probe = fig.text(0.0, -1.0, part1, fontsize=TITLE_FONTSIZE,
                          fontweight="bold", ha="left", va="top")
        p1_bbox = probe.get_window_extent(renderer=renderer)
        p1_w_frac = p1_bbox.width / fig_w
        probe.remove()
        full_text.remove()
        fig.text(full_left_frac, 0.96, part1,
                 fontsize=TITLE_FONTSIZE, fontweight="bold",
                 ha="left", va="top",
                 bbox=dict(facecolor="#FFFF00", edgecolor="none", pad=0.3))
        fig.text(full_left_frac + p1_w_frac, 0.96, part2_full,
                 fontsize=TITLE_FONTSIZE, fontweight="bold",
                 ha="left", va="top")
    elif title:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_h = fig.bbox.height
        suptitle = fig.suptitle(title, fontsize=TITLE_FONTSIZE,
                                 fontweight="bold", x=0.5, y=0.97)
        title_bottom_frac = (
            suptitle.get_window_extent(renderer=renderer).y0 / fig_h
        )

    source = spec.get("source")
    source_lines = source.count("\n") + 1 if source else 0
    
    label_space_frac = label_space_inches / dynamic_width
    
    plot_left = 0.02
    plot_right = 1.0 - label_space_frac - 0.01
    plot_right = max(0.30, min(0.98, plot_right))
    
    # Base bottom margin for x-axis ticks is ~0.8 inches.
    # Each line of source text needs ~0.25 inches.
    bottom_margin_inches = 0.8 + (source_lines * 0.25)
    plot_bottom = bottom_margin_inches / 16.0
    
    plot_top = 0.75  # lowered to 0.75 to match legacy L5 spacing for titles and grain labels
    
    gap = max(0.015, label_space_frac)  # Margin between subplots matches label space
    
    total_width = plot_right - plot_left
    panel_width = (total_width - gap * (n_panels - 1)) / n_panels
    panel_height = plot_top - plot_bottom
    for i, ax in enumerate(axes):
        ax.set_position([plot_left + i * (panel_width + gap), plot_bottom,
                         panel_width, panel_height])

    fig.canvas.draw()
    fig_w = fig.bbox.width
    n_pts = total_pts
    for ax_panel, sp in zip(axes, subplots):
        bbox = ax_panel.get_position()
        x0_disp, _ = ax_panel.transData.transform((0, 0))
        x1_disp, _ = ax_panel.transData.transform((n_pts - 1, 0))
        x_center = ((x0_disp + x1_disp) / 2) / fig_w
        # Subtitle sits halfway between the bottom of the main title and
        # the top of the "By Q / By M / By W" grain labels.
        # The grain labels are drawn at axes-fraction y=1.05.
        grain_labels_y_fig = bbox.y0 + 1.05 * (bbox.y1 - bbox.y0)
        
        # We calculate the visual midpoint. If title_bottom_frac is missing, we
        # fallback to something reasonable.
        y_top = (title_bottom_frac + grain_labels_y_fig) / 2.0
        
        # `Subplot k Title = ...` directive in the doc overrides the
        # default subtitle (which is the measure column name). Falls
        # back to sp["name"] when no override was supplied.
        subtitle_text = sp.get("title_override") or sp["name"]
        # Render subtitle in deep red (#BC2C22) with underline
        from matplotlib.lines import Line2D

        txt = fig.text(x_center, y_top, subtitle_text, color="#BC2C22",
                 fontsize=20, fontweight="bold", fontstyle="italic",
                 ha="center", va="center")

        # Draw underline below the text
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = txt.get_window_extent(renderer)
        # Convert bbox from display to figure coordinates
        fig_width = fig.bbox.width
        fig_height = fig.bbox.height
        x0 = bbox.x0 / fig_width
        x1 = bbox.x1 / fig_width
        # Adjust line y-position to account for new subtitle gap
        line_y = (bbox.y0 / fig_height) - 0.004  # Slight offset below text

        # Create and add the underline line
        line = Line2D([x0, x1], [line_y, line_y], color="#BC2C22", linewidth=2,
                      transform=fig.transFigure, solid_capstyle="round")
        fig.add_artist(line)

    source = spec.get("source")
    if source:
        bbox_l = axes[0].get_position()
        source_x = bbox_l.x0
        
        # 0.8 inches is reserved for x-axis ticks. Place source right below it.
        tick_margin_frac = 0.8 / 16.0
        source_y = bbox_l.y0 - tick_margin_frac
        
        src_text = fig.text(source_x, source_y, source, fontsize=13,
                             color="#555555", ha="left", va="top",
                             linespacing=1.5)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", format=image_format, dpi=dpi)
    validate_output_image(out, image_format)
    plt.close(fig)
    return out


def render_chart_spec(spec: dict, output_path: str | Path, image_format: str = None, dpi: int = 220) -> Path:
    """Render one chart section from a spec dict. Returns the written path."""
    import pathlib
    out_p = pathlib.Path(output_path)
    if image_format is None:
        suffix = out_p.suffix.lower().lstrip(".")
        image_format = "jpeg" if suffix == "jpg" else suffix or "png"
    image_format = image_format.lower()
    if image_format == "jpg":
        image_format = "jpeg"
    # Multi-subplot path — driven by the `Subplot Format` plaintext block in
    # the doc. When `subplots` is in the spec, the legacy 2-panel layout is
    # bypassed entirely.
    if spec.get("subplots"):
        return render_multisubplot_chart(spec, output_path, image_format=image_format, dpi=dpi)
    q_count = spec["q_count"]
    m_count = spec["m_count"]
    w_count = spec["w_count"]
    x_labels = spec["x_labels"]
    total_pts = q_count + m_count + w_count
    if len(x_labels) != total_pts:
        raise ValueError(
            f"x_labels has {len(x_labels)} entries but Q+M+W = {total_pts}"
        )

    # Build left (stacked) series with stable colors. User-defined hex
    # overrides from the per-chart "Custom Color" plaintext block win
    # via the `fixed=` slot on assign_colors; any legend NOT in the dict
    # falls back to the rotating PALETTE.
    custom_color = spec.get("custom_color") or {}
    left_colored = assign_colors(spec["left_names"], fixed=custom_color)
    for i, s in enumerate(left_colored):
        s.values = spec["left_values"][i]

    # Reuse left colors for matching right names, then allocate fresh slots.
    # Custom colors apply on the right side too (Overall is forced to black
    # below, so it's safe to feed the user dict in here as well).
    left_color_map = {s.name: s.color for s in left_colored}
    right_fixed = {**custom_color, **left_color_map}
    right_colored = assign_colors(spec["right_names"], fixed=right_fixed)
    for i, s in enumerate(right_colored):
        if s.name.lower() == "overall":
            s.color = "#000000"
        s.values = spec["right_values"][i]

    x = np.arange(total_pts)
    
    max_left_len = max([len(n) for n in spec["left_names"]], default=0)
    max_right_len = max([len(n) for n in spec["right_names"] + ["Overall"]], default=0)
    
    left_space_inches = max(0.8, max_left_len * 0.16)
    right_space_inches = max(0.8, max_right_len * 0.16)
    
    COLUMN_WIDTH_INCHES = 1.4
    
    pad_l = max(2.0, left_space_inches / COLUMN_WIDTH_INCHES + 0.5)
    pad_r = max(2.0, right_space_inches / COLUMN_WIDTH_INCHES + 0.5)
    
    # Force equal spans and padding so the legacy 2-panel chart stays perfectly centered
    max_pad = max(pad_l, pad_r)
    span_l = total_pts + max_pad
    span_r = total_pts + max_pad
    
    dynamic_width = max(20.0, 4.0 + (span_l + span_r) * COLUMN_WIDTH_INCHES)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(dynamic_width, 12), dpi=220,
                                     gridspec_kw={'width_ratios': [span_l, span_r]})

    unit = spec.get("unit", "K")
    _plot_stacked_area(ax_l, x, x_labels, left_colored, q_count, m_count, w_count, unit)
    _plot_yoy_lines(ax_r, x, x_labels, right_colored, q_count, m_count, w_count)
    
    ax_l.set_xlim(-0.5, total_pts - 1 + max_pad)
    ax_r.set_xlim(-0.5, total_pts - 1 + max_pad)

    # Title font size scales with figure width so it stays visually
    # proportional whether the chart is 20" or 40" wide. Baseline: 26pt
    # at the 20" minimum width — every extra inch adds 1pt up to a soft
    # ceiling. Without this, on a 38"+ canvas a literal "26pt" title
    # looks tiny relative to the chart body (which scales with the
    # figure) — the user reported seeing it as ~12/14pt at 38".
    TITLE_FONTSIZE = max(26, int(round(26 + (dynamic_width - 20.0) * 1.0)))
    TITLE_FONTSIZE = min(TITLE_FONTSIZE, 48)  # don't let it grow forever
    # Global title — centered as a single visual unit at x=0.5.
    # When the title contains "|", the pre-"|" segment is highlighted yellow.
    # Strategy: render the FULL title once, centered at x=0.5, to discover
    # its actual on-canvas extent; then use that extent to place the
    # highlighted part1 at the correct left position so the assembled
    # title is perfectly centered (matching the reference design).
    title = spec.get("title", "")
    title_bottom_frac = 0.95  # fallback if no title
    if "|" in title:
        parts = title.split("|", 1)
        part1 = parts[0].strip()
        part2_full = f" | {parts[1].strip()}"

        # Force a draw so text extents become measurable.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_w = fig.bbox.width
        fig_h = fig.bbox.height

        # Measure the full title to find where its center sits.
        full_text = fig.text(0.5, 0.96, part1 + part2_full, fontsize=TITLE_FONTSIZE,
                              fontweight="bold", ha="center", va="top")
        full_bbox = full_text.get_window_extent(renderer=renderer)
        full_left_frac = full_bbox.x0 / fig_w
        title_bottom_frac = full_bbox.y0 / fig_h  # actual visual bottom

        # Measure part1 alone to learn its width in figure-fraction units.
        probe = fig.text(0.0, -1.0, part1, fontsize=TITLE_FONTSIZE, fontweight="bold",
                          ha="left", va="top")
        p1_bbox = probe.get_window_extent(renderer=renderer)
        p1_w_frac = p1_bbox.width / fig_w
        probe.remove()

        # Replace the full-title draw with the highlighted-part1 + plain-part2
        # combo, anchored so part1.left = full_left_frac (perfect alignment).
        full_text.remove()
        fig.text(full_left_frac, 0.96, part1, fontsize=TITLE_FONTSIZE, fontweight="bold",
                 ha="left", va="top",
                 bbox=dict(facecolor="#FFFF00", edgecolor="none", pad=0.3))
        fig.text(full_left_frac + p1_w_frac, 0.96, part2_full,
                 fontsize=TITLE_FONTSIZE, fontweight="bold", ha="left", va="top")
    elif title:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_h = fig.bbox.height
        suptitle = fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight="bold", x=0.5, y=0.97)
        title_bottom_frac = suptitle.get_window_extent(renderer=renderer).y0 / fig_h

    source = spec.get("source")
    # NOTE: The Source footer is rendered AFTER tight_layout / subtitle
    # placement so we can anchor it under the Daily Revenue subplot (left
    # axes) rather than the global figure corner. The text preserves
    # newlines from the spec verbatim — matplotlib renders embedded "\n"
    # as line breaks naturally, which is what the user wants for multi-
    # line Source blocks pulled from the Lark plaintext code block.

    source_lines = source.count("\n") + 1 if source else 0
    bottom_margin_inches = 0.8 + (source_lines * 0.25)
    plot_bottom = bottom_margin_inches / 12.0

    fig.tight_layout(rect=[0, plot_bottom, 1, 0.82])

    # Subtitles MUST be placed AFTER tight_layout so we measure final bbox /
    # transData positions. Plain "Daily Revenue" / "YoY" in black, centered
    # horizontally over the actual DATA area of each subplot (not the full
    # axes bbox — the axes extend past the last data point to make room for
    # right-edge series labels). We map the data range [0, n-1] from data
    # coords -> display coords -> figure-fraction so the subtitle lines up
    # with the visual midpoint between the leftmost and rightmost data ticks.
    fig.canvas.draw()  # finalise layout so transData / bbox are accurate
    fig_w = fig.bbox.width
    n_pts = total_pts
    for ax_panel, subtitle in ((ax_l, "Daily Revenue"), (ax_r, "YoY")):
        bbox = ax_panel.get_position()  # figure-fraction Bbox (post-layout)
        x0_disp, _ = ax_panel.transData.transform((0, 0))
        x1_disp, _ = ax_panel.transData.transform((n_pts - 1, 0))
        x_center = ((x0_disp + x1_disp) / 2) / fig_w
        # Place subtitle just above the chart with a small fixed gap (math-based:
        # the subtitle text is centered at this y, then va="center" splits the
        # text height around it, leaving clear separation from the chart top).
        grain_labels_y_fig = bbox.y0 + 1.05 * (bbox.y1 - bbox.y0)
        y_top = (title_bottom_frac + grain_labels_y_fig) / 2.0

        from matplotlib.lines import Line2D
        txt = fig.text(x_center, y_top, subtitle, color="#BC2C22",
                 fontsize=20, fontweight="bold", fontstyle="italic",
                 ha="center", va="center")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        txt_bbox = txt.get_window_extent(renderer)
        fig_width = fig.bbox.width
        fig_height = fig.bbox.height
        x0 = txt_bbox.x0 / fig_width
        x1 = txt_bbox.x1 / fig_width
        line_y = (txt_bbox.y0 / fig_height) - 0.004
        
        line = Line2D([x0, x1], [line_y, line_y], color="#BC2C22", linewidth=2,
                      transform=fig.transFigure, solid_capstyle="round")
        fig.add_artist(line)

    # Source footer — anchored to the bottom-left corner of the Daily
    # Revenue (left) subplot's data area, sitting BELOW the x-axis labels.
    # We use the left axes bbox in figure-fraction units so the footer
    # tracks the chart even when the figure is resized. A small fixed
    # gap (figure-fraction) below `bbox.y0` keeps the text visually
    # separated from the x-axis tick labels.
    #
    # Multi-line safety: matplotlib renders the text as a vertical block
    # with va="top", so a 3-line source extends DOWN from the anchor and
    # may clip below the canvas. After drawing, we measure the actual
    # rendered bbox in figure-fraction; if its bottom dips below 0, we
    # push the whole text upward by exactly the overflow + a small margin
    # so the last line stays inside the canvas. This is purely corrective —
    # the user-requested anchor (bbox.y0 - 0.08) is preserved when there's
    # already room.
    if source:
        bbox_l = ax_l.get_position()
        source_x = bbox_l.x0
        
        # 0.8 inches reserved for x-axis ticks
        tick_margin_frac = 0.8 / 12.0
        source_y = bbox_l.y0 - tick_margin_frac
        
        src_text = fig.text(source_x, source_y, source, fontsize=13,
                             color="#555555", ha="left", va="top",
                             linespacing=1.5)
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", format=image_format, dpi=dpi)
    validate_output_image(out, image_format)
    plt.close(fig)
    return out


def validate_output_image(output_path: Path, image_format: Optional[str] = None) -> None:
    output_path = Path(output_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RenderValidationError(f"Output image was not created: {output_path}")
    fmt = (image_format or output_path.suffix.lower().lstrip(".")).lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt in ("png", "jpeg"):
        with Image.open(output_path) as image:
            image.verify()
        with Image.open(output_path) as image:
            image.load()
    elif fmt == "svg":
        ET.parse(output_path)
    else:
        raise RenderValidationError(f"Unsupported validation format: {fmt}")


def resolve_period_tokens(
    document_format: Dict[str, Any],
    chart_period_format: Optional[Dict[str, Any]],
    grain_key: str,
) -> Optional[List[str]]:
    if chart_period_format is not None:
        return list(chart_period_format.get(grain_key.upper(), []))
    return list(document_format.get(grain_key.upper(), []))


def hydrate_chart_data(
    chart: Dict[str, Any], document_format: Dict[str, Any], fetcher: DataFetcher
) -> Dict[str, Any]:
    hydrated = dict(chart)
    for grain_key in GRAIN_KEYS:
        source = chart.get(grain_key) or {}
        if not source.get("fetch_mode"):
            hydrated[grain_key] = _empty_grain()
            continue
        raw_data = fetcher.fetch(source)
        grain = transform_mdp_data(raw_data)
        tokens = resolve_period_tokens(document_format, chart.get("period_format"), grain_key)
        hydrated[grain_key] = apply_format_filter(grain, tokens)
        LOGGER.info(
            "Prepared grain %s for %s with %s periods",
            grain_key,
            chart["title"],
            len(hydrated[grain_key]["x_labels"]),
        )
    return hydrated


def render_workflow(
    templates: Sequence[Path],
    output_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    chart_index: Optional[int] = None,
    image_format: str = "png",
    dpi: int = 220,
    mdp_cli: str = "mdp-cli",
    retries: int = 3,
    timeout: int = 120,
    unit_override: Optional[str] = None,
    source_override: Optional[str] = None,
) -> List[Path]:
    documents = [load_template_file(Path(path)) for path in templates]
    fetcher = DataFetcher(mdp_cli=mdp_cli, retries=retries, timeout=timeout)
    rendered_paths = []

    all_chart_entries = []
    for document in documents:
        for local_index, chart in enumerate(document["charts"]):
            all_chart_entries.append((document, local_index, chart))

    if chart_index is not None:
        if chart_index < 0 or chart_index >= len(all_chart_entries):
            raise IndexError(
                "chart-index %s is out of range for %s total charts"
                % (chart_index, len(all_chart_entries))
            )
        all_chart_entries = [all_chart_entries[chart_index]]

    single_output = output_path if len(all_chart_entries) == 1 else None
    if output_path and len(all_chart_entries) > 1:
        raise ValueError("--output can only be used when rendering exactly one chart")
    if output_dir is None and single_output is None:
        output_dir = Path("charts_generated")

    for global_index, (document, local_index, chart) in enumerate(all_chart_entries):
        LOGGER.info(
            "Rendering chart %s/%s: %s",
            global_index + 1,
            len(all_chart_entries),
            chart["title"],
        )
        hydrated_chart = hydrate_chart_data(chart, document["format"], fetcher)
        spec = build_chart_spec(
            hydrated_chart, unit_override=unit_override, source_override=source_override
        )
        if single_output is not None:
            target_path = Path(single_output)
        else:
            template_slug = slugify(Path(document["source_path"]).stem)
            file_name = "chart_%02d_%s_%s.%s" % (
                local_index,
                template_slug,
                slugify(chart["title"]),
                image_format,
            )
            target_path = Path(output_dir) / file_name
        rendered_paths.append(
            render_chart_spec(spec, target_path, image_format=image_format, dpi=dpi)
        )

    return rendered_paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone QMW chart generation workflow")
    parser.add_argument(
        "--template",
        action="append",
        required=True,
        help="Template JSON path. Repeat for multiple templates.",
    )
    parser.add_argument(
        "--output",
        help="Single output file path. Only valid when rendering one chart.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory used when rendering multiple charts.",
    )
    parser.add_argument(
        "--chart-index",
        type=int,
        help="Render exactly one chart from the combined template chart list.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpeg", "svg"],
        default="png",
        help="Output image format.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI.")
    parser.add_argument("--mdp-cli", default="mdp-cli", help="Path to mdp-cli binary.")
    parser.add_argument("--timeout", type=int, default=120, help="Fetch timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Fetch retry attempts.")
    parser.add_argument(
        "--unit",
        choices=["K", "M", "None", "Full"],
        default=None,
        help="Override chart unit.",
    )
    parser.add_argument("--source", default=None, help="Override source footer text.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    template_paths = [Path(item).expanduser().resolve() for item in args.template]
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    LOGGER.info("Starting QMW workflow for %s template(s)", len(template_paths))
    try:
        rendered = render_workflow(
            templates=template_paths,
            output_path=output_path,
            output_dir=output_dir,
            chart_index=args.chart_index,
            image_format=args.format,
            dpi=args.dpi,
            mdp_cli=args.mdp_cli,
            retries=args.retries,
            timeout=args.timeout,
            unit_override=args.unit,
            source_override=args.source,
        )
    except Exception as exc:
        LOGGER.exception("QMW workflow failed: %s", exc)
        return 1

    for path in rendered:
        print(path)
    LOGGER.info("Completed QMW workflow: rendered %s chart(s)", len(rendered))
    return 0


if __name__ == "__main__":
    sys.exit(main())
