"""Rights-clean synthetic inspection data.

Generates textured metal-like part surfaces with known defects (scratch, pit,
crack) and optional capture degradations (blur, low exposure, glare, noise).
Every sample carries ground truth so the agent can be evaluated honestly.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFECT_TYPES = ("scratch", "pit", "crack")
DEGRADATIONS = ("none", "blur", "dark", "glare", "noise")


@dataclass
class Defect:
    kind: str
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Sample:
    name: str
    defects: list[Defect] = field(default_factory=list)
    degradation: str = "none"

    @property
    def label(self) -> str:
        return "FAIL" if self.defects else "PASS"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        return d


def _base_surface(rng: np.random.Generator, size: int) -> np.ndarray:
    """Brushed-metal texture: directional noise + gentle illumination gradient."""
    noise = rng.normal(0, 1, (size, size)).astype(np.float32)
    k = int(rng.integers(15, 31))
    kernel = np.zeros((1, k), np.float32)
    kernel[0, :] = 1.0 / k
    brushed = cv2.filter2D(noise, -1, kernel)
    angle = float(rng.uniform(-20, 20))
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    brushed = cv2.warpAffine(brushed, m, (size, size), borderMode=cv2.BORDER_REFLECT)
    brushed = cv2.normalize(brushed, None, -1, 1, cv2.NORM_MINMAX)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    gx, gy = rng.uniform(-25, 25, 2)
    grad = gx * (xx - 0.5) + gy * (yy - 0.5)
    base = float(rng.uniform(135, 170))
    img = base + 16.0 * brushed + grad + rng.normal(0, 2.0, (size, size))
    return np.clip(img, 0, 255).astype(np.float32)


def _add_scratch(img, rng, size):
    length = int(rng.integers(size // 6, size // 3))
    x0, y0 = (int(v) for v in rng.integers(40, size - 40, 2))
    ang = rng.uniform(0, np.pi)
    x1 = int(np.clip(x0 + length * np.cos(ang), 10, size - 10))
    y1 = int(np.clip(y0 + length * np.sin(ang), 10, size - 10))
    th = int(rng.integers(2, 4))
    depth = float(rng.uniform(22, 80))
    mask = np.zeros_like(img)
    cv2.line(mask, (x0, y0), (x1, y1), 1.0, th, cv2.LINE_AA)
    img -= depth * cv2.GaussianBlur(mask, (3, 3), 0)
    return Defect("scratch", _bbox_of(mask))


def _add_pit(img, rng, size):
    cx, cy = (int(v) for v in rng.integers(40, size - 40, 2))
    r = int(rng.integers(6, 13))
    mask = np.zeros_like(img)
    cv2.ellipse(mask, (cx, cy), (r, int(r * rng.uniform(0.7, 1.0))),
                float(rng.uniform(0, 180)), 0, 360, 1.0, -1, cv2.LINE_AA)
    img -= float(rng.uniform(25, 85)) * cv2.GaussianBlur(mask, (5, 5), 0)
    return Defect("pit", _bbox_of(mask))


def _add_crack(img, rng, size):
    pts = [rng.integers(60, size - 60, 2).astype(float)]
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(int(rng.integers(8, 14))):
        ang += rng.uniform(-0.7, 0.7)
        step = rng.uniform(7, 13)
        nxt = pts[-1] + step * np.array([np.cos(ang), np.sin(ang)])
        pts.append(np.clip(nxt, 10, size - 10))
    mask = np.zeros_like(img)
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1.0, 2, cv2.LINE_AA)
    img -= float(rng.uniform(25, 80)) * mask
    return Defect("crack", _bbox_of(mask))


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect((mask > 0.05).astype(np.uint8))
    return (int(x), int(y), int(w), int(h))


def _degrade(img: np.ndarray, kind: str, rng) -> np.ndarray:
    if kind == "blur":
        k = int(rng.choice([7, 9, 11]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif kind == "dark":
        img = img * float(rng.uniform(0.18, 0.28))
    elif kind == "glare":
        size = img.shape[0]
        cx, cy = (int(v) for v in rng.integers(size // 4, 3 * size // 4, 2))
        yy, xx = np.mgrid[0:size, 0:size]
        r = float(rng.uniform(size * 0.12, size * 0.2))
        spot = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
        img = img + 140.0 * spot
    elif kind == "noise":
        img = img + rng.normal(0, 16, img.shape)
    return np.clip(img, 0, 255)


_ADDERS = {"scratch": _add_scratch, "pit": _add_pit, "crack": _add_crack}


def make_sample(seed: int, size: int = 512, n_defects: int | None = None,
                degradation: str | None = None) -> tuple[np.ndarray, Sample]:
    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    img = _base_surface(rng, size)
    if n_defects is None:
        n_defects = pyr.choice([0, 0, 1, 1, 2])
    defects = [_ADDERS[pyr.choice(DEFECT_TYPES)](img, rng, size) for _ in range(n_defects)]
    if degradation is None:
        degradation = pyr.choice(DEGRADATIONS)
    img = _degrade(img, degradation, rng)
    bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return bgr, Sample(name=f"part_{seed:04d}", defects=defects, degradation=degradation)


def write_dataset(out_dir: str | Path, n: int, start_seed: int = 0) -> list[Sample]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(start_seed, start_seed + n):
        img, s = make_sample(i)
        cv2.imwrite(str(out / f"{s.name}.png"), img)
        samples.append(s)
    (out / "labels.json").write_text(json.dumps([s.to_dict() for s in samples], indent=1))
    return samples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    print(f"wrote {len(write_dataset(a.out, a.n, a.seed))} samples to {a.out}")


class SimulatedStation:
    """A camera station that can re-shoot the same part.

    Re-rendering with the same seed reproduces the identical part and defects;
    only the capture condition changes. A re-shoot usually fixes the problem,
    but not always (20%), so the agent must re-check quality.
    """

    def __init__(self, seed: int, size: int = 512, n_defects: int | None = None):
        self.seed, self.size, self.n_defects = seed, size, n_defects

    def __call__(self, hint: str, attempt: int) -> np.ndarray:
        r = random.Random(self.seed * 31 + attempt)
        deg = "none" if r.random() < 0.8 else r.choice(DEGRADATIONS[1:])
        img, _ = make_sample(self.seed, self.size, n_defects=self.n_defects, degradation=deg)
        return img
