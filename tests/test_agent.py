import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from inspectagent import agent as A
from inspectagent import vision as V
from inspectagent.synth import SimulatedStation, make_sample


def test_opencv_major_version_is_5():
    assert V.OPENCV_VERSION.split(".")[0] == "5"


@pytest.mark.parametrize("deg,issue", [("blur", "blurry"), ("dark", "underexposed"),
                                       ("glare", "glare"), ("noise", "noisy")])
def test_quality_flags_degradations(deg, issue):
    img, _ = make_sample(11, degradation=deg, n_defects=0)
    assert issue in V.assess_quality(img).issues()


def test_clean_image_has_no_issues():
    img, _ = make_sample(11, degradation="none", n_defects=0)
    assert V.assess_quality(img).issues() == []


def test_detects_planted_defect_location():
    img, s = make_sample(3, degradation="none", n_defects=1)
    cands, _ = V.detect_defects(img)
    gx, gy, gw, gh = s.defects[0].bbox
    assert any(V.iou(c.bbox, (gx, gy, gw, gh)) > 0.2 for c in cands)


def test_agent_pass_and_fail():
    good, _ = make_sample(21, degradation="none", n_defects=0)
    bad, _ = make_sample(21, degradation="none", n_defects=2)
    assert A.inspect(good, A.RulePlanner())["disposition"] == "PASS"
    assert A.inspect(bad, A.RulePlanner())["disposition"] == "FAIL"


def test_agent_recaptures_on_glare_and_vision_changes_decision():
    img, _ = make_sample(42, degradation="glare", n_defects=1)
    r = A.inspect(img, A.RulePlanner(), camera=SimulatedStation(42, n_defects=1))
    tools = [s["tool"] for s in r["trace"]]
    assert tools[0] == "request_recapture"          # OpenCV quality -> action
    assert r["disposition"] in ("FAIL", "ESCALATE")


def test_upload_without_station_escalates_glare_when_unclear():
    img, _ = make_sample(42, degradation="glare", n_defects=0)
    r = A.inspect(img, A.RulePlanner(), camera=None)
    assert "request_recapture" not in [s["tool"] for s in r["trace"]]
    assert r["disposition"] in ("PASS", "ESCALATE")


class LiarPlanner:
    """Always claims PASS: the guard must not let it through on a defective part."""
    name = "liar"

    def next(self, st):
        if not st.detected:
            return "detect_defects", {"sensitivity": 1.0}, "detect"
        return "finalize", {"disposition": "PASS", "reason": "looks fine"}, "lie"


def test_guard_overrides_unsafe_pass():
    img, _ = make_sample(5, degradation="none", n_defects=2)
    r = A.inspect(img, LiarPlanner())
    assert r["disposition"] == "FAIL"
    assert "guard" in r["trace"][-1]["reason"]


class BrokenPlanner:
    name = "broken"

    def next(self, st):
        raise TimeoutError("bedrock unreachable")


def test_planner_failure_falls_back_to_rules():
    img, _ = make_sample(8, degradation="none", n_defects=1)
    r = A.inspect(img, BrokenPlanner())
    assert r["planner"] == "broken+fallback"
    assert r["disposition"] == "FAIL"


def test_api_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE_DIR", str(tmp_path))
    import inspectagent.store as S
    S._store = None
    from inspectagent.app import app
    c = TestClient(app)
    assert c.get("/api/health").json()["opencv"].startswith("5")
    r = c.post("/api/station/inspect?seed=7&degradation=none&defects=1").json()
    assert r["disposition"] == "FAIL" and r["action"] == "divert_to_rework_bin"
    img, _ = make_sample(42, degradation="glare", n_defects=0)
    ok, buf = cv2.imencode(".png", img)
    up = c.post("/api/inspect", files={"file": ("p.png", buf.tobytes(), "image/png")}).json()
    assert up["disposition"] in ("PASS", "ESCALATE")
    if up["disposition"] == "ESCALATE":
        rv = c.post(f"/api/inspections/{up['id']}/review",
                    json={"decision": "PASS", "reviewer": "qa1"}).json()
        assert rv["final_disposition"] == "PASS"
        assert c.post(f"/api/inspections/{up['id']}/review",
                      json={"decision": "PASS", "reviewer": "qa1"}).status_code == 409
    assert len(c.get("/api/inspections").json()) == 2
    assert c.post("/api/inspect", files={"file": ("x.png", b"nope", "image/png")}).status_code == 400


class FakeBedrock:
    """Stub of bedrock-runtime.converse that follows the rule policy, then lies."""

    def __init__(self):
        self.calls = []

    def converse(self, **kw):
        self.calls.append(kw)
        obs = json.loads(kw["messages"][0]["content"][0]["text"])
        assert {t["toolSpec"]["name"] for t in kw["toolConfig"]["tools"]} == set(A.TOOLS)
        if not obs["detected"]:
            use = {"name": "detect_defects", "input": {"sensitivity": 1.0, "why": "look"}}
        else:
            use = {"name": "finalize", "input": {"disposition": "PASS", "reason": "ok", "why": "done"}}
        return {"output": {"message": {"content": [{"toolUse": {"toolUseId": "1", **use}}]}}}


def test_bedrock_planner_contract_and_guard():
    p = A.BedrockPlanner.__new__(A.BedrockPlanner)
    p.model_id, p.client = "test-model", FakeBedrock()
    img, _ = make_sample(5, degradation="none", n_defects=1)
    r = A.inspect(img, p)
    assert r["planner"] == "bedrock"
    assert r["trace"][0]["planner"] == "bedrock" and r["trace"][0]["reason"] == "look"
    assert r["disposition"] == "FAIL"          # LLM's PASS overridden by guard
    assert len(p.client.calls) == 2
