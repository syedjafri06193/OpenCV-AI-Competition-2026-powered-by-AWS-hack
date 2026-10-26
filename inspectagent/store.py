"""Persistence for inspection records, evidence images and the review queue.

Backends
  * local (default): ./data/store/ JSON + PNG files.
  * aws: S3 for evidence images and full traces, DynamoDB for records and the
    human-review queue. Selected with STORE=aws plus BUCKET / TABLE env vars
    (set automatically by the SAM template).
"""
from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, Decimal)):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def encode_jpg(img: np.ndarray, q: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


class LocalStore:
    kind = "local"

    def __init__(self, root: str | None = None):
        self.root = Path(root or os.environ.get("STORE_DIR", "data/store"))
        (self.root / "records").mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(parents=True, exist_ok=True)

    def put_image(self, key: str, img: np.ndarray) -> str:
        (self.root / "images" / f"{key}.jpg").write_bytes(encode_jpg(img))
        return f"images/{key}.jpg"

    def get_image(self, key: str) -> bytes | None:
        p = self.root / "images" / f"{key}.jpg"
        return p.read_bytes() if p.exists() else None

    def put_record(self, rec: dict) -> None:
        (self.root / "records" / f"{rec['id']}.json").write_text(json.dumps(_jsonable(rec)))

    def get_record(self, rid: str) -> dict | None:
        p = self.root / "records" / f"{rid}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def list_records(self, limit: int = 50, status: str | None = None) -> list[dict]:
        files = sorted((self.root / "records").glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for p in files:
            r = json.loads(p.read_text())
            if status and r.get("review_status") != status:
                continue
            out.append(_summary(r))
            if len(out) >= limit:
                break
        return out


class AwsStore:
    kind = "aws"

    def __init__(self):
        import boto3
        self.bucket = os.environ["BUCKET"]
        self.s3 = boto3.client("s3")
        self.table = boto3.resource("dynamodb").Table(os.environ["TABLE"])

    def put_image(self, key: str, img: np.ndarray) -> str:
        k = f"images/{key}.jpg"
        self.s3.put_object(Bucket=self.bucket, Key=k, Body=encode_jpg(img),
                           ContentType="image/jpeg", ServerSideEncryption="AES256")
        return k

    def get_image(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=f"images/{key}.jpg")["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put_record(self, rec: dict) -> None:
        rec = _jsonable(rec)
        # Full trace to S3 (can exceed DynamoDB's 400KB item limit); summary to DDB.
        self.s3.put_object(Bucket=self.bucket, Key=f"records/{rec['id']}.json",
                           Body=json.dumps(rec).encode(), ContentType="application/json",
                           ServerSideEncryption="AES256")
        item = json.loads(json.dumps(_summary(rec)), parse_float=Decimal)
        item["pk"] = "INSPECTION"
        item["sk"] = f"{rec['created_at']}#{rec['id']}"
        self.table.put_item(Item=item)

    def get_record(self, rid: str) -> dict | None:
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=f"records/{rid}.json")["Body"]
            return json.loads(body.read())
        except self.s3.exceptions.NoSuchKey:
            return None

    def list_records(self, limit: int = 50, status: str | None = None) -> list[dict]:
        from boto3.dynamodb.conditions import Attr, Key
        kw = {"KeyConditionExpression": Key("pk").eq("INSPECTION"),
              "ScanIndexForward": False, "Limit": limit if not status else 500}
        if status:
            kw["FilterExpression"] = Attr("review_status").eq(status)
        items = self.table.query(**kw).get("Items", [])
        return [_jsonable(i) for i in items][:limit]


def _summary(r: dict) -> dict:
    return {k: r.get(k) for k in (
        "id", "created_at", "source", "disposition", "reason", "final_disposition",
        "review_status", "reviewer", "planner", "n_defects", "action", "total_ms")}


_store = None


def get_store():
    global _store
    if _store is None:
        _store = AwsStore() if os.environ.get("STORE") == "aws" else LocalStore()
    return _store


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
