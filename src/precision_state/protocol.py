"""Deterministic outcome-blind prompt selection and fail-closed main authorization."""
import json
from pathlib import Path
from precision_state.artifacts import canonical_hash, sha256
from precision_state.harness import token_hash


def select_prompts(examples, *, chunk=64, pilot_n=4, main_n=32, seed=1729):
    if type(chunk) is not int or chunk <= 0 or type(pilot_n) is not int or not 1 <= pilot_n <= 4 or type(main_n) is not int or main_n < 16:
        raise ValueError("pilot 1..4; main >=16; positive chunk required")
    pool = []
    ids, hashes = set(), set()
    for row in examples:
        if row["task"] != "wikitext2":
            continue
        ident = f"wikitext2:{row['index']}"
        tokens = row["tokens"][:2*chunk+1]
        if len(tokens) != 2*chunk+1:
            raise ValueError(f"short candidate: {ident}")
        if any(type(i) is not int or i < 0 for i in tokens):
            raise ValueError("invalid token")
        h = token_hash(tokens)
        if ident in ids or h in hashes:
            raise ValueError("duplicate prompt ID or tokens")
        ids.add(ident)
        hashes.add(h)
        pool.append({"id": ident, "source_record_sha256": canonical_hash(row),
                     "token_ids": tokens, "token_sha256": h,
                     "selection_key": canonical_hash([seed, ident])})
    pool.sort(key=lambda p: (p["selection_key"], p["id"]))
    if len(pool) < pilot_n + main_n:
        raise ValueError("insufficient disjoint candidates")
    return {"pilot": pool[:pilot_n], "main": pool[pilot_n:pilot_n+main_n]}


def authorize_main(config, root):
    """A draft config, missing pilot, or edited freeze cannot authorize a main run."""
    root = Path(root)
    if config.get("status") != "frozen" or config.get("authorized_fresh_main_processes") != 2:
        raise ValueError("main is not authorized: protocol is not frozen")
    prompts = config["prompts"]
    if len(prompts["main"]) < 16 or not 1 <= len(prompts["pilot"]) <= 4:
        raise ValueError("minimum evidence or pilot cap violated")
    all_ids, all_hashes = set(), set()
    for row in prompts["pilot"] + prompts["main"]:
        if row["token_sha256"] != token_hash(row["token_ids"]) or len(row["token_ids"]) != 2*config["chunk_tokens"]+1:
            raise ValueError("invalid frozen prompt")
        if row["id"] in all_ids or row["token_sha256"] in all_hashes:
            raise ValueError("pilot/main overlap")
        all_ids.add(row["id"])
        all_hashes.add(row["token_sha256"])
    if not config.get("pilot_controls_passed") or not config.get("runtime_authorized"):
        raise ValueError("pilot/runtime gates not passed")
    for path, expected in config["frozen_files"].items():
        if sha256(root / path) != expected:
            raise ValueError(f"modified frozen input: {path}")
    lock = json.loads((root / "evidence/freeze.json").read_text())
    if lock["config_sha256"] != sha256(root / "configs/precision_state_v1.json"):
        raise ValueError("config freeze mismatch")
    if lock["protocol_sha256"] != sha256(root / "PRECISION_STATE_PROTOCOL.md"):
        raise ValueError("protocol freeze mismatch")
    return True
