"""Inspection agent: perceive -> decide -> act loop over OpenCV 5 tools.

The planner chooses the next tool from the current observations. Two planners:

* ``RulePlanner``  - deterministic policy; always available, used for eval.
* ``BedrockPlanner`` - an LLM on Amazon Bedrock picks the next tool via the
  Converse tool-use API. Its choices are constrained to the same tool set,
  a step budget, and a safety guard on the final disposition. Any Bedrock
  error falls back to ``RulePlanner`` and is recorded in the trace.

Every step (tool, args, reason, observation, latency) is recorded so the
OpenCV output -> later decision link is auditable.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import vision as V

MAX_STEPS = 16
CONFIRM_CONF = 0.55          # min confidence for a defect to fail a part
MAX_VERIFY = 5               # zoom-verifications per inspection
MAX_RECAPTURE = 2            # re-shoots the agent may request from the station

TOOLS = {
    "fix_exposure": "Gamma + CLAHE to recover an underexposed frame.",
    "denoise": "Non-local-means denoising for a noisy frame.",
    "sharpen": "Unsharp mask for a blurry frame.",
    "suppress_glare": "Flatten specular glare; masks saturated pixels from detection.",
    "detect_defects": "Run defect detection. args: {sensitivity: 1.0 normal | 1.6 sweep for faint suspects}.",
    "zoom_verify": "Active perception: zoom 2x on a candidate and re-detect locally. args: {candidate_id}.",
    "request_recapture": "Ask the camera station to re-shoot the part. args: {hint: lighting|focus}. Only if recapture_available.",
    "finalize": "End inspection. args: {disposition: PASS|FAIL|ESCALATE, reason}.",
}


@dataclass
class Step:
    i: int
    tool: str
    args: dict
    reason: str
    observation: dict
    ms: float
    planner: str


@dataclass
class State:
    original: np.ndarray
    work: np.ndarray
    quality0: V.Quality
    quality: V.Quality
    remedies: list[str] = field(default_factory=list)
    ignore_mask: np.ndarray | None = None
    candidates: dict[int, dict] = field(default_factory=dict)  # id -> {cand, status}
    swept: bool = False
    detected: bool = False
    verifies: int = 0
    steps: list[Step] = field(default_factory=list)
    next_id: int = 0
    camera: object = None        # callable(hint, attempt) -> image, or None
    recaptures: int = 0

    def summary(self) -> dict:
        """Compact observation the planner sees."""
        return {
            "quality": self.quality.to_dict(),
            "initial_issues": self.quality0.issues(),
            "remedies_applied": self.remedies,
            "detected": self.detected,
            "swept_faint": self.swept,
            "verifications_used": self.verifies,
            "candidates": [
                {"id": cid, "status": c["status"], **c["cand"].to_dict()}
                for cid, c in self.candidates.items()
            ],
            "recapture_available": self.camera is not None and self.recaptures < MAX_RECAPTURE,
            "recaptures_used": self.recaptures,
            "steps_used": len(self.steps),
            "step_budget": MAX_STEPS,
        }


# ------------------------------------------------------------------ tools ---
REMEDY_FOR = {"underexposed": "fix_exposure", "noisy": "denoise",
              "blurry": "sharpen", "glare": "suppress_glare"}


def _add_candidates(st: State, cands: list[V.Candidate], status: str) -> list[int]:
    new = []
    for c in cands:
        if any(V.iou(c.bbox, o["cand"].bbox) > 0.1 or _contains(o["cand"].bbox, c.bbox)
               for o in st.candidates.values()):
            continue
        st.candidates[st.next_id] = {"cand": c, "status": status}
        new.append(st.next_id)
        st.next_id += 1
    return new


def _contains(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    cx, cy = bx + bw / 2, by + bh / 2
    return ax <= cx <= ax + aw and ay <= cy <= ay + ah


def run_tool(st: State, tool: str, args: dict) -> dict:
    if tool == "fix_exposure":
        st.work = V.fix_exposure(st.work)
    elif tool == "denoise":
        st.work = V.denoise(st.work)
    elif tool == "sharpen":
        st.work = V.sharpen(st.work)
    elif tool == "suppress_glare":
        st.work, st.ignore_mask = V.suppress_glare(st.work)
    elif tool == "detect_defects":
        sens = float(np.clip(args.get("sensitivity", 1.0), 0.5, 2.0))
        cands, _ = V.detect_defects(st.work, sens, st.ignore_mask)
        if sens <= 1.0:
            confirmed = [c for c in cands if c.confidence >= CONFIRM_CONF]
            weak = [c for c in cands if c.confidence < CONFIRM_CONF]
            ids = _add_candidates(st, confirmed, "detected") + _add_candidates(st, weak, "suspect")
            st.detected = True
        else:
            ids = _add_candidates(st, cands, "suspect")
            st.swept = True
        return {"sensitivity": sens, "new_candidates": ids, "total": len(st.candidates)}
    elif tool == "zoom_verify":
        cid = int(args["candidate_id"])
        if cid not in st.candidates:
            return {"error": f"no candidate {cid}"}
        entry = st.candidates[cid]
        c = entry["cand"]
        st.verifies += 1
        up, _ = V.zoom(st.work, c.bbox, scale=2.0, pad=32)
        local, _ = V.detect_defects(up, sensitivity=1.0, min_area=60)
        # Keep detections overlapping the candidate's (scaled) center region.
        hits = [l for l in local if l.strength >= 7.5 and l.area >= 80]
        if hits:
            best = max(hits, key=lambda l: l.confidence)
            entry["status"] = "confirmed"
            c.kind = best.kind if best.kind != "unknown" else c.kind
            c.confidence = float(max(c.confidence, best.confidence, CONFIRM_CONF))
            return {"candidate_id": cid, "verdict": "confirmed", "zoom_kind": best.kind,
                    "zoom_strength": round(best.strength, 2), "zoom_hits": len(hits)}
        entry["status"] = "rejected"
        return {"candidate_id": cid, "verdict": "rejected", "zoom_hits": 0}
    elif tool == "request_recapture":
        if st.camera is None or st.recaptures >= MAX_RECAPTURE:
            return {"error": "recapture unavailable"}
        st.recaptures += 1
        hint = str(args.get("hint", "lighting"))
        img = st.camera(hint, st.recaptures)
        st.original = img
        st.work = img.copy()
        st.quality0 = st.quality = V.assess_quality(img)
        st.remedies, st.ignore_mask = [], None
        st.candidates, st.detected, st.swept, st.next_id = {}, False, False, 0
        return {"hint": hint, "attempt": st.recaptures, "new_quality": st.quality.to_dict()}
    elif tool == "finalize":
        return {"disposition": args.get("disposition"), "reason": args.get("reason", "")}
    else:
        return {"error": f"unknown tool {tool}"}

    if tool in REMEDY_FOR.values():
        st.remedies.append(tool)
        st.quality = V.assess_quality(st.work)
        return {"quality_after": st.quality.to_dict()}
    return {}


# --------------------------------------------------------------- planners ---
class RulePlanner:
    name = "rules"

    def next(self, st: State) -> tuple[str, dict, str]:
        issues = st.quality0.issues()
        if st.camera is not None and st.recaptures < MAX_RECAPTURE and not st.remedies:
            if "glare" in issues:
                return "request_recapture", {"hint": "lighting"}, \
                    "glare saturates pixels (information lost) -> re-shoot with diffuse lighting"
            if "blurry" in issues:
                return "request_recapture", {"hint": "focus"}, \
                    "frame is out of focus -> re-shoot after refocus"
        for issue in issues:
            tool = REMEDY_FOR[issue]
            if tool not in st.remedies:
                return tool, {}, f"capture issue '{issue}' -> apply {tool}"
        if not st.detected:
            return "detect_defects", {"sensitivity": 1.0}, "image conditioned; run detection"
        if not st.swept:
            return "detect_defects", {"sensitivity": 1.6}, "sweep for faint defects at higher sensitivity"
        pending = [cid for cid, c in st.candidates.items()
                   if c["status"] in ("suspect",)]
        if pending and st.verifies < MAX_VERIFY:
            cid = pending[0]
            return "zoom_verify", {"candidate_id": cid}, f"candidate {cid} is uncertain -> zoom to verify"
        return "finalize", decide(st), "all evidence gathered"


def decide(st: State) -> dict:
    """Policy for the final disposition (also used as a guard on the LLM)."""
    confirmed = [c for c in st.candidates.values()
                 if c["status"] in ("detected", "confirmed")]
    unresolved = [c for c in st.candidates.values() if c["status"] == "suspect"]
    residual = st.quality.issues()
    if confirmed:
        kinds = sorted({c["cand"].kind for c in confirmed})
        return {"disposition": "FAIL", "reason": f"{len(confirmed)} defect(s): {', '.join(kinds)}"}
    if unresolved:
        return {"disposition": "ESCALATE", "reason": f"{len(unresolved)} unverified suspect region(s)"}
    if "glare" in st.quality0.issues() and st.ignore_mask is not None and \
            float((st.ignore_mask > 0).mean()) > 0.02:
        return {"disposition": "ESCALATE",
                "reason": "glare hid part of the surface; human should re-image or inspect"}
    if residual and "blurry" in residual:
        return {"disposition": "ESCALATE", "reason": "image still blurry after sharpening; request recapture"}
    return {"disposition": "PASS", "reason": "no defects found on a usable image"}


class BedrockPlanner:
    """LLM planner on Amazon Bedrock (Converse API with tool use)."""

    name = "bedrock"

    def __init__(self, model_id: str, region: str | None = None):
        import boto3
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.fallback = RulePlanner()

    def _tool_config(self):
        specs = []
        for name, desc in TOOLS.items():
            props, req = {}, []
            if name == "detect_defects":
                props = {"sensitivity": {"type": "number"}}
            elif name == "zoom_verify":
                props, req = {"candidate_id": {"type": "integer"}}, ["candidate_id"]
            elif name == "finalize":
                props = {"disposition": {"type": "string", "enum": ["PASS", "FAIL", "ESCALATE"]},
                         "reason": {"type": "string"}}
                req = ["disposition", "reason"]
            props["why"] = {"type": "string", "description": "one-line reason for this step"}
            specs.append({"toolSpec": {"name": name, "description": desc, "inputSchema": {
                "json": {"type": "object", "properties": props, "required": req}}}})
        return {"tools": specs, "toolChoice": {"any": {}}}

    SYSTEM = (
        "You are a manufacturing surface-inspection agent. You see JSON observations "
        "produced by OpenCV 5 tools and must choose exactly one next tool. Fix capture "
        "problems first, then detect, sweep for faint defects, zoom-verify uncertain "
        "candidates, and finalize. Never PASS a part with detected/confirmed defects or "
        "unverified suspects; ESCALATE to a human when evidence is insufficient.")

    def next(self, st: State) -> tuple[str, dict, str]:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": self.SYSTEM}],
            messages=[{"role": "user", "content": [{"text": json.dumps(st.summary())}]}],
            toolConfig=self._tool_config(),
            inferenceConfig={"maxTokens": 300, "temperature": 0},
        )
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                args = dict(block["toolUse"]["input"])
                why = str(args.pop("why", ""))
                return block["toolUse"]["name"], args, why or "llm choice"
        raise RuntimeError("no tool choice returned")


def make_planner(kind: str | None = None):
    kind = kind or os.environ.get("PLANNER", "rules")
    model = os.environ.get("BEDROCK_MODEL_ID")
    if kind == "bedrock" and model:
        return BedrockPlanner(model, os.environ.get("BEDROCK_REGION"))
    return RulePlanner()


# ------------------------------------------------------------------- loop ---
def _guard(st: State, tool: str, args: dict) -> tuple[str, dict, str | None]:
    """Hard constraints applied to whatever the planner chose."""
    if tool not in TOOLS:
        return "finalize", decide(st), f"planner chose unknown tool {tool!r}"
    if tool == "request_recapture" and (st.camera is None or st.recaptures >= MAX_RECAPTURE):
        t, a, _ = RulePlanner().next(st)
        return t, a, "guard: recapture unavailable"
    if tool == "zoom_verify" and st.verifies >= MAX_VERIFY:
        return "finalize", decide(st), "verification budget exhausted"
    if tool == "finalize":
        policy = decide(st)
        if args.get("disposition") == "PASS" and policy["disposition"] != "PASS":
            return "finalize", policy, "guard: PASS overridden by safety policy"
        if args.get("disposition") not in ("PASS", "FAIL", "ESCALATE"):
            return "finalize", policy, "guard: invalid disposition"
    if tool in REMEDY_FOR.values() and tool in st.remedies:
        t, a, _ = RulePlanner().next(st)
        return t, a, f"guard: {tool} already applied, redirected to {t}"
    return tool, args, None


def inspect(img: np.ndarray, planner=None, agentic: bool = True, camera=None) -> dict:
    """Run one inspection. ``agentic=False`` is the single-pass baseline."""
    t0 = time.perf_counter()
    q = V.assess_quality(img)
    st = State(original=img, work=img.copy(), quality0=q, quality=q, camera=camera)

    if not agentic:
        cands, _ = V.detect_defects(img)
        _add_candidates(st, [c for c in cands if c.confidence >= CONFIRM_CONF], "detected")
        disp = {"disposition": "FAIL" if st.candidates else "PASS",
                "reason": "single-pass baseline"}
        return _result(st, disp, t0, "baseline")

    planner = planner or make_planner()
    disp, used = None, planner.name
    for i in range(MAX_STEPS):
        ts = time.perf_counter()
        try:
            tool, args, reason = planner.next(st)
            pname = planner.name
        except Exception as e:  # LLM/network failure -> deterministic fallback
            tool, args, reason = RulePlanner().next(st)
            reason = f"fallback ({type(e).__name__}): {reason}"
            pname, used = "rules-fallback", f"{planner.name}+fallback"
        tool, args, note = _guard(st, tool, args)
        if note:
            reason = f"{reason} | {note}"
        obs = run_tool(st, tool, args)
        st.steps.append(Step(i, tool, args, reason, obs,
                             round((time.perf_counter() - ts) * 1000, 1), pname))
        if tool == "finalize":
            disp = {"disposition": args["disposition"], "reason": args.get("reason", "")}
            break
    if disp is None:
        disp = decide(st)
        disp["reason"] += " (step budget reached)"
    return _result(st, disp, t0, used)


def _result(st: State, disp: dict, t0: float, planner: str) -> dict:
    defects = [c["cand"] for c in st.candidates.values()
               if c["status"] in ("detected", "confirmed")]
    rejected = [c["cand"] for c in st.candidates.values() if c["status"] == "rejected"]
    overlay = V.draw_overlay(st.work, defects, rejected)
    return {
        "id": uuid.uuid4().hex[:12],
        "disposition": disp["disposition"],
        "reason": disp["reason"],
        "planner": planner,
        "opencv": V.OPENCV_VERSION,
        "quality_initial": st.quality0.to_dict(),
        "quality_final": st.quality.to_dict(),
        "defects": [d.to_dict() for d in defects],
        "candidates": [{"id": k, "status": v["status"], **v["cand"].to_dict()}
                       for k, v in st.candidates.items()],
        "trace": [s.__dict__ for s in st.steps],
        "recaptures": st.recaptures,
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "_overlay": overlay,
    }
