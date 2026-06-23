#!/usr/bin/env python3
"""
QMW Chart Studio — backend API.

This service WRAPS the chart-generation logic in ``qmw_chart_workflow.py``;
that file is imported unmodified and never rewritten. The web layer only:
  * validates uploaded JSON (using the workflow's own validator),
  * drives the workflow chart-by-chart in a background job,
  * publishes step-by-step progress + logs,
  * serves rendered images and a bundled download.

Two data modes are supported transparently:
  * LIVE  — grain refs carry report_id / url; data is fetched via mdp-cli/HTTP
            (the stock DataFetcher). Requires Lark/Byte Cloud authorization.
  * INLINE — grain refs embed `rows` / `data` / `results`; rendered fully
            offline (no mdp-cli auth needed). The workflow file stays untouched;
            we override DataFetcher.fetch on a per-job instance only.

Endpoints
---------
GET  /                         -> SPA (index.html)
POST /upload                   -> validate + register a job, returns job_id + metadata
POST /generate-chart           -> start generation for a job_id  (body: {"job_id": ...})
GET  /progress/<job_id>        -> JSON progress snapshot (poll this)
GET  /events/<job_id>          -> Server-Sent Events stream (real-time, optional)
GET  /chart/<job_id>/<index>   -> a single rendered chart PNG (?download=1 to attach)
GET  /download/<job_id>        -> ZIP of all rendered charts (or single PNG if one)
GET  /sample                   -> a ready-to-use inline demo template
GET  /healthz                  -> service + mdp-cli availability
"""
from __future__ import annotations

import io
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_file,
    send_from_directory,
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import qmw_chart_workflow as qmw  # noqa: E402  (imported unmodified)

app = Flask(__name__, static_folder=None)

WORK_ROOT = BASE_DIR / "_jobs"
WORK_ROOT.mkdir(exist_ok=True)

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# The canonical, user-facing pipeline stages (kept in sync with the frontend).
STAGES = [
    "File uploaded successfully",
    "Validating JSON structure",
    "Reading chart configuration",
    "Processing Q/M/W data",
    "Generating chart layout",
    "Rendering chart image",
    "Finalizing output",
    "Chart generation completed",
]


# ---------------------------------------------------------------------------
# Inline-data support (offline mode, no mdp-cli auth required)
# ---------------------------------------------------------------------------
def _grain_has_inline(ref: Any) -> bool:
    return isinstance(ref, dict) and any(k in ref for k in ("data", "rows", "results"))


def _template_has_inline(payload: Dict[str, Any]) -> bool:
    for chart in payload.get("charts", []) or []:
        if isinstance(chart, dict):
            for g in ("q", "m", "w"):
                if _grain_has_inline(chart.get(g)):
                    return True
    return False


def _build_inline_lookup(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace each inline grain ref with a synthetic report_id (so the
    workflow's schema validation passes) and map that id -> embedded rows."""
    lookup: Dict[str, Any] = {}
    counter = 0
    for chart in payload.get("charts", []) or []:
        if not isinstance(chart, dict):
            continue
        for g in ("q", "m", "w"):
            ref = chart.get(g)
            if _grain_has_inline(ref):
                counter += 1
                sid = f"inline-{counter}"
                if "data" in ref:
                    lookup[sid] = {"data": ref["data"]}
                elif "rows" in ref:
                    lookup[sid] = {"data": {"rows": ref["rows"]}}
                else:
                    lookup[sid] = {"data": {"results": ref["results"]}}
                chart[g] = {"report_id": sid}
    return lookup


# ---------------------------------------------------------------------------
# Job progress model (polled via /progress, streamed via /events)
# ---------------------------------------------------------------------------
def _new_progress() -> Dict[str, Any]:
    return {
        "state": "ready",          # ready | running | done | error
        "percent": 0,
        "stage": "",
        "stage_index": -1,
        "logs": [],                # [{ts, level, message}]
        "charts": [],              # [{index, title, ok, url?, error?}]
        "summary": None,           # {generated, failed, total}
        "error": None,
        "data_mode": None,
        "subscribers": [],         # list[queue.Queue] for SSE fan-out
    }


def _emit(job: Dict[str, Any], event: str, data: Dict[str, Any]) -> None:
    """Push an SSE-formatted event to every live subscriber of this job."""
    payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    for q in list(job["progress"]["subscribers"]):
        try:
            q.put_nowait(payload)
        except Exception:
            pass


def _log(job: Dict[str, Any], message: str, level: str = "info") -> None:
    entry = {"ts": time.time(), "level": level, "message": message}
    job["progress"]["logs"].append(entry)
    _emit(job, "log", entry)


def _set_stage(job: Dict[str, Any], stage: str, percent: int) -> None:
    p = job["progress"]
    p["stage"] = stage
    p["percent"] = max(p["percent"], percent)
    if stage in STAGES:
        p["stage_index"] = STAGES.index(stage)
    _emit(job, "stage", {"stage": stage, "percent": p["percent"],
                         "stage_index": p["stage_index"]})


# ---------------------------------------------------------------------------
# POST /upload  — validate and register a job
# ---------------------------------------------------------------------------
@app.post("/upload")
def upload():
    if "file" not in request.files:
        return jsonify(ok=False, error="No file was attached to the request."), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify(ok=False, error="No file was selected."), 400
    if not f.filename.lower().endswith(".json"):
        return jsonify(
            ok=False,
            error=f"Unsupported file type. Expected a .json file, got “{f.filename}”.",
        ), 400

    raw = f.read()
    if not raw:
        return jsonify(ok=False, error="The uploaded file is empty."), 400

    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return jsonify(ok=False, error="The file is not valid UTF-8 text."), 400
    except json.JSONDecodeError as exc:
        return jsonify(
            ok=False,
            error=f"Malformed JSON — {exc.msg} at line {exc.lineno}, column {exc.colno}.",
        ), 400

    if not isinstance(payload, dict):
        return jsonify(
            ok=False,
            error="The top-level value must be an object containing “format” and “charts”.",
        ), 400

    has_inline = _template_has_inline(payload)
    inline_lookup = _build_inline_lookup(payload) if has_inline else {}

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    template_path = job_dir / "template.json"
    template_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    try:
        validated = qmw.validate_template_document(payload, template_path)
    except qmw.TemplateValidationError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        msg = (str(exc).replace(str(template_path), "the template")
               .replace(str(job_dir), "the template"))
        return jsonify(ok=False, error=f"Template structure error — {msg}"), 400
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify(ok=False, error=f"Validation failed — {exc}"), 400

    charts_meta = []
    for idx, chart in enumerate(validated["charts"]):
        grains = [
            {"grain": g.upper(),
             "mode": (chart.get(g) or {}).get("fetch_mode"),
             "report_id": (chart.get(g) or {}).get("report_id")}
            for g in ("q", "m", "w")
        ]
        charts_meta.append({
            "index": idx,
            "title": chart["title"],
            "unit": chart.get("unit"),
            "is_subplot": bool(chart.get("subplot_format")),
            "grains": grains,
        })

    data_mode = "inline" if has_inline else "live"
    with JOBS_LOCK:
        JOBS[job_id] = {
            "dir": str(job_dir),
            "template_path": str(template_path),
            "charts": charts_meta,
            "image_format": "png",
            "has_inline": has_inline,
            "inline_lookup": inline_lookup,
            "filename": f.filename,
            "progress": _new_progress(),
        }
        JOBS[job_id]["progress"]["data_mode"] = data_mode

    return jsonify(
        ok=True,
        job_id=job_id,
        filename=f.filename,
        size=len(raw),
        chart_count=len(charts_meta),
        charts=charts_meta,
        data_mode=data_mode,
        mdp_available=bool(shutil.which("mdp-cli")),
        format=validated["format"].get("raw", ""),
    )


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------
def _run_generation(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return
    p = job["progress"]
    p["state"] = "running"

    try:
        _set_stage(job, "File uploaded successfully", 5)
        _log(job, f"Received template “{job['filename']}”.")

        _set_stage(job, "Validating JSON structure", 12)
        document = qmw.load_template_file(Path(job["template_path"]))
        charts = document["charts"]
        n = len(charts)
        _log(job, "Schema validation passed.", "success")

        _set_stage(job, "Reading chart configuration", 20)
        _log(job, f"Loaded {n} chart definition(s).")
        if job["has_inline"]:
            _log(job, "Inline dataset detected — rendering offline (mdp-cli not required).")
        else:
            _log(job, "Live data mode — values will be fetched via mdp-cli.")

        fetcher = qmw.DataFetcher()
        if job["has_inline"]:
            lookup = job["inline_lookup"]
            original_fetch = fetcher.fetch

            def inline_fetch(source, _lookup=lookup, _orig=original_fetch):
                rid = (source or {}).get("report_id")
                return _lookup[rid] if rid in _lookup else _orig(source)

            fetcher.fetch = inline_fetch  # instance-only override

        out_dir = Path(job["dir"]) / "charts"
        out_dir.mkdir(exist_ok=True)
        image_format = job["image_format"]
        results: List[Dict[str, Any]] = []

        for idx, chart in enumerate(charts):
            title = chart["title"]
            span_lo = 20 + int((idx / max(n, 1)) * 70)
            span_hi = 20 + int(((idx + 1) / max(n, 1)) * 70)

            _set_stage(job, "Processing Q/M/W data", span_lo)
            _log(job, f"[{idx + 1}/{n}] Processing Q/M/W data — {title}")
            try:
                hydrated = qmw.hydrate_chart_data(chart, document["format"], fetcher)
                total_pts = sum(
                    len((hydrated.get(g) or {}).get("x_labels") or [])
                    for g in ("q", "m", "w")
                )
                if total_pts == 0:
                    raise ValueError(
                        "No data points were resolved for any grain (Q/M/W). "
                        "Verify report IDs / Format period filters or supply inline rows."
                    )
                _log(job, f"    Resolved {total_pts} data column(s) across Q/M/W.")

                _set_stage(job, "Generating chart layout",
                           span_lo + (span_hi - span_lo) // 3)
                spec = qmw.build_chart_spec(hydrated)

                _set_stage(job, "Rendering chart image",
                           span_lo + 2 * (span_hi - span_lo) // 3)
                out_path = out_dir / f"chart_{idx:02d}.{image_format}"
                qmw.render_chart_spec(spec, out_path, image_format=image_format)
                _log(job, f"    Rendered {out_path.name}.", "success")

                meta = {"index": idx, "title": title, "ok": True,
                        "url": f"/chart/{job_id}/{idx}"}
                results.append(meta)
                p["charts"].append(meta)
                _emit(job, "chart", meta)
            except Exception as exc:  # noqa: BLE001
                _log(job, f"    Failed at chart {idx + 1}: {exc}", "error")
                meta = {"index": idx, "title": title, "ok": False, "error": str(exc)}
                results.append(meta)
                p["charts"].append(meta)
                _emit(job, "chart", meta)

        _set_stage(job, "Finalizing output", 95)
        ok_count = sum(1 for r in results if r["ok"])
        fail_count = n - ok_count
        summary = {"generated": ok_count, "failed": fail_count, "total": n}
        p["summary"] = summary

        if ok_count == 0:
            p["state"] = "error"
            p["error"] = (
                "No charts could be generated. "
                + ("Authorize mdp-cli in Lark, or upload a template with inline data."
                   if not job["has_inline"] else "Check the dataset structure.")
            )
            _log(job, p["error"], "error")
            _emit(job, "done", {"ok": False, **summary, "error": p["error"]})
        else:
            _set_stage(job, "Chart generation completed", 100)
            p["state"] = "done"
            level = "success" if fail_count == 0 else "warn"
            _log(job, f"Completed — {ok_count} chart(s) generated"
                      + (f", {fail_count} failed." if fail_count else "."), level)
            _emit(job, "done", {"ok": True, **summary})

    except qmw.TemplateValidationError as exc:
        _fail(job, f"Template error — {exc}")
    except qmw.DataFetchError as exc:
        msg = str(exc)
        hint = ""
        if "authoriz" in msg.lower() or "pending" in msg.lower():
            hint = (" Approve “Byte Cloud Application Authorization Application” in Lark, "
                    "or upload a template with inline data to render offline.")
        _fail(job, "Data fetch failed." + hint, detail=msg)
    except Exception as exc:  # noqa: BLE001
        _fail(job, f"Backend processing error — {exc}",
              detail=traceback.format_exc(limit=3))
    finally:
        for q in list(job["progress"]["subscribers"]):
            try:
                q.put_nowait(None)
            except Exception:
                pass


def _fail(job: Dict[str, Any], message: str, detail: Optional[str] = None) -> None:
    p = job["progress"]
    p["state"] = "error"
    p["error"] = message
    if detail:
        _log(job, detail, "error")
    _log(job, message, "error")
    _emit(job, "done", {"ok": False, "error": message})


# ---------------------------------------------------------------------------
# POST /generate-chart  — kick off the background job
# ---------------------------------------------------------------------------
@app.post("/generate-chart")
def generate_chart():
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id") or request.form.get("job_id") or request.args.get("job_id")
    if not job_id:
        return jsonify(ok=False, error="Missing “job_id”."), 400
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(ok=False, error="Unknown or expired job id."), 404
    if job["progress"]["state"] == "running":
        return jsonify(ok=True, job_id=job_id, state="running"), 202

    job["progress"] = _new_progress()
    job["progress"]["data_mode"] = "inline" if job["has_inline"] else "live"
    threading.Thread(target=_run_generation, args=(job_id,), daemon=True).start()
    return jsonify(ok=True, job_id=job_id, state="running"), 202


# ---------------------------------------------------------------------------
# GET /progress/<job_id>  — poll a JSON snapshot
# ---------------------------------------------------------------------------
@app.get("/progress/<job_id>")
def progress(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(ok=False, error="Unknown or expired job id."), 404
    p = job["progress"]
    return jsonify(
        ok=True,
        job_id=job_id,
        state=p["state"],
        percent=p["percent"],
        stage=p["stage"],
        stage_index=p["stage_index"],
        stages=STAGES,
        logs=p["logs"][-200:],
        charts=p["charts"],
        summary=p["summary"],
        error=p["error"],
        data_mode=p["data_mode"],
    )


# ---------------------------------------------------------------------------
# GET /events/<job_id>  — Server-Sent Events stream (optional, real-time)
# ---------------------------------------------------------------------------
@app.get("/events/<job_id>")
def events(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(ok=False, error="Unknown or expired job id."), 404

    sub: "queue.Queue" = queue.Queue()
    job["progress"]["subscribers"].append(sub)

    def stream():
        # Replay current state so a late subscriber is in sync.
        p = job["progress"]
        yield (f"event: stage\ndata: "
               f"{json.dumps({'stage': p['stage'], 'percent': p['percent'], 'stage_index': p['stage_index']})}\n\n")
        for entry in p["logs"]:
            yield f"event: log\ndata: {json.dumps(entry, ensure_ascii=False)}\n\n"
        try:
            while True:
                item = sub.get()
                if item is None:
                    break
                yield item
        finally:
            try:
                job["progress"]["subscribers"].remove(sub)
            except ValueError:
                pass

    return Response(stream(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


# ---------------------------------------------------------------------------
# Image + bundle delivery
# ---------------------------------------------------------------------------
@app.get("/chart/<job_id>/<int:idx>")
def get_chart(job_id: str, idx: int):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(ok=False, error="Unknown job id."), 404
    img = Path(job["dir"]) / "charts" / f"chart_{idx:02d}.{job['image_format']}"
    if not img.exists():
        return jsonify(ok=False, error="Chart not found."), 404
    return send_file(
        img, mimetype="image/png",
        as_attachment=(request.args.get("download") == "1"),
        download_name=f"chart_{idx:02d}.png",
    )


@app.get("/download/<job_id>")
def download(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(ok=False, error="Unknown job id."), 404
    charts_dir = Path(job["dir"]) / "charts"
    images = sorted(charts_dir.glob(f"chart_*.{job['image_format']}")) if charts_dir.exists() else []
    if not images:
        return jsonify(ok=False, error="No rendered charts available to download yet."), 404
    if len(images) == 1:
        return send_file(images[0], mimetype="image/png", as_attachment=True,
                         download_name="qmw_chart.png")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            zf.write(img, arcname=img.name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="qmw_charts.zip")


# ---------------------------------------------------------------------------
# Static / utility
# ---------------------------------------------------------------------------
@app.get("/sample")
def sample():
    path = BASE_DIR / "sample_inline_demo.json"
    if not path.exists():
        return jsonify(ok=False, error="Sample not available."), 404
    return send_file(path, mimetype="application/json",
                     as_attachment=True, download_name="qmw_inline_demo.json")


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/healthz")
def healthz():
    return jsonify(ok=True, mdp_cli=bool(shutil.which("mdp-cli")), jobs=len(JOBS))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
