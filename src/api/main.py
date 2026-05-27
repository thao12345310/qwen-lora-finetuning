"""FastAPI server exposing /rewrite and /compare.

Run:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from src.api.schemas import CompareResponse, RewriteRequest, RewriteResponse
from src.inference.predict import load_config, rewrite

CONFIG_PATH = Path("configs/inference.yaml")
_cfg = load_config(CONFIG_PATH)

app = FastAPI(title="Vietnamese Dialogue Rewriter")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": _cfg["model_name_or_path"]}


@app.post("/rewrite", response_model=RewriteResponse)
def rewrite_endpoint(req: RewriteRequest) -> RewriteResponse:
    turns = [t.model_dump() for t in req.conversation]
    out = rewrite(turns, _cfg, use_adapter=True)
    return RewriteResponse(rewritten_query=out)


@app.post("/compare", response_model=CompareResponse)
def compare_endpoint(req: RewriteRequest) -> CompareResponse:
    turns = [t.model_dump() for t in req.conversation]
    base = rewrite(turns, _cfg, use_adapter=False)
    finetuned = rewrite(turns, _cfg, use_adapter=True)
    return CompareResponse(base_model_output=base, fine_tuned_output=finetuned)
