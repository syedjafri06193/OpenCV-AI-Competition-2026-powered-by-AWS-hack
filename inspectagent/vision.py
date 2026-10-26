"""OpenCV 5 perception tools used by the inspection agent.

Every tool is a pure function: image(s) + params in, image/observation out.
The agent decides which tools to call, in what order, with which params.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

OPENCV_VERSION = cv2.__version__


# ---------------------------------------------------------------- quality ---
@dataclass
class Quality:
    sharpness: float      # variance of Laplacian on a normalized image
    brightness: float     # mean intensity 0..255
    contrast: float       # std intensity
    glare_frac: float     # fraction of near-saturated pixels
    noise: float          # robust sigma of high-frequency residual

    def issues(self) -> list[str]:
        out = []
        if self.brightness < 70:
            out.append("underexposed")
        if self.glare_frac > 0.01:
            out.append("glare")
        if self.noise > 7.0:
            out.append("noisy")
        if self.sharpness < 20.0:
            out.append("blurry")
        return out

    def to_dict(self) -> dict:
        d = {k: round(float(v), 4) for k, v in asdict(self).items()}
        d["issues"] = self.issues()
        return d


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def assess_quality(img: np.ndarray) -> Quality:
    g = to_gray(img)
    gf = g.astype(np.float32)
    mean = float(gf.mean())
    # Normalize exposure before measuring sharpness so a dark image isn't "blurry".
    # High-frequency energy relative to overall variance: exposure-invariant.
    # Median over tiles so defects and glare (low-frequency) don't skew it.
    lap = cv2.Laplacian(cv2.GaussianBlur(gf, (0, 0), 0.8), cv2.CV_32F)
    hp = gf - cv2.GaussianBlur(gf, (0, 0), 6)
    H, W = g.shape
    ratios = []
    for ty in range(0, H - 63, 64):
        for tx in range(0, W - 63, 64):
            hv = float(hp[ty:ty + 64, tx:tx + 64].var())
            if hv > 1e-3:
                ratios.append(float(lap[ty:ty + 64, tx:tx + 64].var()) / hv)
    sharp = float(np.median(ratios)) * 100.0 if ratios else 0.0
    resid = gf - cv2.medianBlur(g, 3).astype(np.float32)
    noise = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    glare = float((g >= 250).mean())
    return Quality(sharp, mean, float(gf.std()), glare, noise)


# --------------------------------------------------------------- enhancers ---
def fix_exposure(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Gamma-lift dark frames, then CLAHE for local contrast."""
    g = to_gray(img)
    mean = max(float(g.mean()), 1.0)
    gamma = float(np.clip(np.log(150 / 255) / np.log(mean / 255), 0.3, 1.0))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    g = cv2.LUT(g, lut)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(g), cv2.COLOR_GRAY2BGR)


def denoise(img: np.ndarray, strength: float = 12.0) -> np.ndarray:
    g = to_gray(img)
    out = cv2.fastNlMeansDenoising(g, None, h=float(strength), templateWindowSize=7,
                                   searchWindowSize=21)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def sharpen(img: np.ndarray, amount: float = 1.5, sigma: float = 2.0) -> np.ndarray:
    g = to_gray(img).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    out = np.clip(g + amount * (g - blur), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def suppress_glare(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten specular hot-spots by dividing out a large-scale illumination field.

    Returns the corrected image and a mask of pixels that were saturated
    (detections there are unreliable and are down-weighted by the agent).
    """
    g = to_gray(img).astype(np.float32) + 1.0
    illum = cv2.GaussianBlur(g, (0, 0), 25)
    flat = g / illum * float(np.median(illum))
    out = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sat = (to_gray(img) >= 250).astype(np.uint8) * 255
    sat = cv2.dilate(sat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), sat


# --------------------------------------------------------------- detection ---
@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]
    area: float
    elongation: float
    solidity: float
    strength: float      # peak blackhat response / background sigma
    kind: str
    confidence: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [int(v) for v in self.bbox]
        for k in ("area", "elongation", "solidity", "strength", "confidence"):
            d[k] = round(float(d[k]), 3)
        return d


def _classify(elong: float, solidity: float, area: float) -> tuple[str, float]:
    if elong < 2.2 and solidity > 0.75:
        return "pit", min(1.0, solidity)
    if elong >= 4.0 and solidity >= 0.45:
        return "scratch", min(1.0, 0.5 + solidity / 2)
    if solidity < 0.45:
        return "crack", min(1.0, 1.0 - solidity)
    return "unknown", 0.4


def detect_defects(img: np.ndarray, sensitivity: float = 1.0,
                   ignore_mask: np.ndarray | None = None,
                   min_area: int = 25) -> tuple[list[Candidate], np.ndarray]:
    """Find dark surface anomalies (scratches, pits, cracks).

    Black-hat morphology isolates dark structures smaller than the kernel from
    the brushed background; a robust (median/MAD) threshold adapts to texture.
    ``sensitivity`` > 1 lowers the threshold (used for zoomed re-inspection).
    Returns candidates and the binary defect map.
    """
    g = to_gray(img)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    med = float(np.median(bh))
    mad = max(float(1.4826 * np.median(np.abs(bh - med))), 1.0)
    z = (bh - med) / mad
    thr = 6.0 / max(sensitivity, 0.1)
    binary = (z > thr).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if ignore_mask is not None:
        binary[ignore_mask > 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cands: list[Candidate] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        elong = max(rw, rh) / max(min(rw, rh), 1.0)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = float(area) / max(hull_area, 1.0)
        strength = float(z[labels == i].max())
        kind, kconf = _classify(elong, min(solidity, 1.0), area)
        # Confidence blends detection margin with shape certainty.
        margin = float(np.clip((strength - thr) / (2 * thr), 0, 1))
        conf = 0.6 * margin + 0.4 * kconf
        cands.append(Candidate((x, y, w, h), float(area), float(elong),
                               min(solidity, 1.0), strength, kind, conf))
    cands.sort(key=lambda c: -c.confidence)
    return cands, binary


def zoom(img: np.ndarray, bbox, scale: float = 2.0, pad: int = 24):
    """Active perception: crop a region with context and upsample it."""
    H, W = img.shape[:2]
    x, y, w, h = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1]
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return up, (x0, y0)


# ----------------------------------------------------------------- overlay ---
COLORS = {"scratch": (0, 140, 255), "pit": (255, 80, 200), "crack": (0, 0, 255),
          "unknown": (0, 220, 220)}


def draw_overlay(img: np.ndarray, cands: list[Candidate],
                 rejected: list[Candidate] | None = None) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in rejected or []:
        x, y, w, h = c.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (150, 150, 150), 1)
    for c in cands:
        x, y, w, h = c.bbox
        col = COLORS.get(c.kind, (0, 255, 0))
        cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), col, 2)
        cv2.putText(out, f"{c.kind} {c.confidence:.2f}", (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1)
