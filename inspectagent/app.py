"""Web API + UI for the inspection agent. Runs locally (uvicorn) or on Lambda (Mangum)."""
from __future__ import annotations

import base64
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import agent as A
from .store import encode_jpg, get_store, now_iso
from .synth import DEGRADATIONS, SimulatedStation, make_sample
from .vision import OPENCV_VERSION

ROOT = Path(__file__).parent
MAX_UPLOAD = 8 * 1024 * 1024
MAX_SIDE = 1024

app = FastAPI(title="InspectAgent", version="1.0.0")

ACTIONS = {"PASS": "release_to_next_station", "FAIL": "divert_to_rework_bin",
           "ESCALATE": "hold_and_queue_for_human_review"}


def _emit_metrics(rec: dict) -> None:
    """CloudWatch Embedded Metric Format: a structured log line becomes metrics."""
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "InspectAgent", "Dimensions": [["Disposition"]],
            "Metrics": [{"Name": "Inspections", "Unit": "Count"},
                        {"Name": "LatencyMs", "Unit": "Milliseconds"},
                        {"Name": "AgentSteps", "Unit": "Count"},
                        {"Name": "Recaptures", "Unit": "Count"}]}]},
        "Disposition": rec["disposition"], "Inspections": 1,
        "LatencyMs": rec["total_ms"], "AgentSteps": len(rec["trace"]),
        "Recaptures": rec.get("recaptures", 0), "inspection_id": rec["id"],
        "planner": rec["planner"]}))


def _b64(img: np.ndarray) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(encode_jpg(img)).decode()


def _run(img: np.ndarray, source: str, planner: str | None, camera=None) -> dict:
    p = A.make_planner(planner)
    res = A.inspect(img, planner=p, camera=camera)
    overlay = res.pop("_overlay")
    store = get_store()
    res.update({
        "created_at": now_iso(), "source": source,
        "n_defects": len(res["defects"]),
        "action": ACTIONS[res["disposition"]],
        "final_disposition": None if res["disposition"] == "ESCALATE" else res["disposition"],
        "review_status": "pending" if res["disposition"] == "ESCALATE" else "auto",
        "reviewer": None,
    })
    res["images"] = {"input": store.put_image(f"{res['id']}_input", img),
                     "overlay": store.put_image(f"{res['id']}_overlay", overlay)}
    store.put_record(res)
    _emit_metrics(res)
    out = dict(res)
    out["input_jpg"], out["overlay_jpg"] = _b64(img), _b64(overlay)
    return out


def _decode_upload(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image (use PNG or JPEG)")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------------ routes ---
@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "opencv": OPENCV_VERSION, "store": get_store().kind,
            "bedrock_configured": bool(os.environ.get("BEDROCK_MODEL_ID")),
            "default_planner": os.environ.get("PLANNER", "rules")}


@app.post("/api/station/inspect")
def station_inspect(seed: int | None = Query(None, ge=0, le=10**6),
                    degradation: str | None = Query(None),
                    defects: int | None = Query(None, ge=0, le=3),
                    planner: str | None = Query(None, pattern="^(rules|bedrock)$")):
    """Simulated camera station: shoot a synthetic part and let the agent inspect it.
    The agent can ask this station to re-shoot (lighting / focus)."""
    if degradation is not None and degradation not in DEGRADATIONS:
        raise HTTPException(400, f"degradation must be one of {DEGRADATIONS}")
    seed = random.randint(0, 999_999) if seed is None else seed
    img, sample = make_sample(seed, n_defects=defects, degradation=degradation)
    out = _run(img, f"station:seed={seed}", planner, camera=SimulatedStation(seed, n_defects=defects))
    out["ground_truth"] = sample.to_dict()
    return out


@app.post("/api/inspect")
async def inspect_upload(file: UploadFile = File(...),
                         planner: str | None = Query(None, pattern="^(rules|bedrock)$")):
    """Inspect an uploaded photo. No station attached, so the agent cannot re-shoot
    and will escalate frames it cannot fix in software."""
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image too large (8 MB max)")
    return _run(_decode_upload(data), f"upload:{(file.filename or 'image')[:60]}", planner)


@app.get("/api/inspections")
def list_inspections(limit: int = Query(30, ge=1, le=100), status: str | None = None):
    return get_store().list_records(limit=limit, status=status)


@app.get("/api/inspections/{rid}")
def get_inspection(rid: str):
    rec = get_store().get_record(rid)
    if not rec:
        raise HTTPException(404, "not found")
    return rec


@app.get("/api/images/{key}")
def get_image(key: str):
    if not key.replace("_", "").isalnum():
        raise HTTPException(400, "bad key")
    data = get_store().get_image(key)
    if data is None:
        raise HTTPException(404, "not found")
    return Response(data, media_type="image/jpeg")


class Review(BaseModel):
    decision: str = Field(pattern="^(PASS|FAIL)$")
    reviewer: str = Field(min_length=1, max_length=60)
    note: str = Field("", max_length=500)


@app.post("/api/inspections/{rid}/review")
def review(rid: str, body: Review):
    """Human-in-the-loop: a person resolves an escalated inspection; the agent's
    held part is then released or diverted accordingly."""
    store = get_store()
    rec = store.get_record(rid)
    if not rec:
        raise HTTPException(404, "not found")
    if rec.get("review_status") != "pending":
        raise HTTPException(409, "inspection is not awaiting review")
    rec.update({"review_status": "reviewed", "final_disposition": body.decision,
                "reviewer": body.reviewer, "review_note": body.note,
                "reviewed_at": now_iso(), "action": ACTIONS[body.decision]})
    store.put_record(rec)
    print(json.dumps({"event": "human_review", "inspection_id": rid,
                      "decision": body.decision, "agent_disposition": rec["disposition"]}))
    return {k: rec[k] for k in ("id", "final_disposition", "action", "reviewer", "review_status")}


@app.get("/api/eval")
def eval_metrics():
    base = ROOT.parent / "eval" / "results"
    out = {}
    for mode in ("station", "upload"):
        p = base / mode / "metrics.json"
        if p.exists():
            out[mode] = json.loads(p.read_text())
    if not out:
        return JSONResponse({"error": "run eval/run_all.sh first"}, 404)
    return out


try:  # Lambda entrypoint
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:  # pragma: no cover
    handler = None
