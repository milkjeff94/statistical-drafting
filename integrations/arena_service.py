#!/usr/bin/env python3
"""
Lightweight HTTP service to connect MTGA log clients to statisticaldrafting.

POST /recommend with JSON body:
{
  "set": "EOE",                 # optional if detect_set=true
  "draft_mode": "Premier",       # Premier | Trad (default Premier)
  "pack_ids": [12345, ...],       # Arena GRP IDs for current pack
  "picked_ids": [111, 222],       # Arena GRP IDs already picked (optional)
  "detect_set": true              # If true, infer set from ids when set is missing
}

Response:
{
  "set": "EOE",
  "recommendations": [
    {"grpId": 12345, "name": "Card Name", "rating": 92.1, "synergy": 1.3, "rank": 1},
    ...
  ]
}

Run:
  pip install flask pandas torch
  python integrations/arena_service.py

Then POST from your mtga-log-client when a pack changes.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Tuple

from flask import Flask, jsonify, request
import pandas as pd

from statisticaldrafting import DraftModel


app = Flask(__name__)


def _repo_data_dir() -> str:
    # Resolve data/ directory relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(here, "..", "data"))
    return data_dir


def _load_set_cards(set_code: str) -> pd.DataFrame:
    """Load per-set card list to establish model's card order."""
    path = os.path.join(_repo_data_dir(), "cards", f"{set_code}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing per-set cards file: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=64)
def _id_to_name_map(set_code: str) -> Dict[int, str]:
    """
    Build grpId -> name map scoped to the set using cards/cards.csv.
    Ensures names match those used by the model's per-set card list.
    """
    cards_all = os.path.join(_repo_data_dir(), "cards", "cards.csv")
    df_all = pd.read_csv(cards_all)
    # Filter to set & booster-eligible
    df_all = df_all[(df_all["expansion"] == set_code) & (df_all["is_booster"])]
    # Only include names present in the per-set list
    df_set = _load_set_cards(set_code)
    valid_names = set(df_set["name"].tolist())
    df_all = df_all[df_all["name"].isin(valid_names)]
    return {int(r.id): r.name for r in df_all.itertuples(index=False)}


def _detect_set_from_ids(grp_ids: List[int]) -> str | None:
    """Heuristic: pick the most common expansion among provided grpIds."""
    cards_all = os.path.join(_repo_data_dir(), "cards", "cards.csv")
    df_all = pd.read_csv(cards_all)
    df_sel = df_all[df_all["id"].isin(grp_ids)]
    if df_sel.empty:
        return None
    top = (
        df_sel[df_sel["is_booster"]]
        .groupby("expansion")["id"]
        .count()
        .sort_values(ascending=False)
    )
    return top.index[0] if len(top) else None


@lru_cache(maxsize=16)
def _load_model(set_code: str, draft_mode: str) -> DraftModel:
    return DraftModel(set=set_code, draft_mode=draft_mode, data_dir=_repo_data_dir())


def _recommend_core(set_code: str, draft_mode: str, pack_ids: List[int], picked_ids: List[int]):
    id2name = _id_to_name_map(set_code)
    # Translate ids -> names, drop unknowns
    pack_names = [id2name[i] for i in pack_ids if i in id2name]
    pick_names = [id2name[i] for i in picked_ids if i in id2name]

    model = _load_model(set_code, draft_mode)
    order = model.get_pick_order(pick_names)
    in_pack = order[order["name"].isin(pack_names)].copy()
    in_pack = in_pack.sort_values(by="rating", ascending=False).reset_index(drop=True)
    # Build reverse map name->id for response
    name2id = {v: k for k, v in id2name.items()}
    recs = []
    for rank, row in enumerate(in_pack.itertuples(index=False), start=1):
        recs.append({
            "grpId": int(name2id.get(row.name, -1)),
            "name": row.name,
            "rating": float(row.rating),
            "synergy": float(row.synergy),
            "rank": rank,
        })
    return recs


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        body = request.get_json(force=True)
        pack_ids = body.get("pack_ids") or body.get("packIds") or []
        picked_ids = body.get("picked_ids") or body.get("picks") or []
        draft_mode = (body.get("draft_mode") or body.get("mode") or "Premier").strip()
        set_code = body.get("set")

        if (not set_code) and body.get("detect_set", True):
            set_code = _detect_set_from_ids(pack_ids + picked_ids)
        if not set_code:
            return jsonify({"error": "Missing set code and auto-detection failed"}), 400

        recs = _recommend_core(set_code, draft_mode, [int(i) for i in pack_ids], [int(i) for i in picked_ids])
        return jsonify({"set": set_code, "recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Bind to localhost:5055 by default
    host = os.environ.get("SD_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("SD_SERVICE_PORT", "5055"))
    app.run(host=host, port=port)
