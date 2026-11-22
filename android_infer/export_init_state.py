#!/usr/bin/env python3
"""Export the learned decoder initial state as raw fp32 arrays.

The split decoder carries learned tokens, raw anchors and raw radii. Radius
activation is performed by the attention and head graphs, not by this exporter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "LocusGS"))
sys.path.insert(0, str(REPO))

from android_infer.locusgs_export import build_model, load_weights  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-gs-tokens", type=int, default=4096)
    ap.add_argument("--img-size", default="256,448")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    opt, model = build_model(args)
    load_weights(model, args.checkpoint)
    model.eval()
    if args.dtype == "fp16":
        model.half()

    dt = np.float16 if args.dtype == "fp16" else np.float32
    with torch.no_grad():
        tokens = model.get_gs_tokens(1)[0]
        anchor_raw = model.gs_anchor_xyz_raw
        anchor_radius = model.gs_anchor_radius_raw[:, 0]

    files = {
        "gs_token_init.f32": tokens,
        "anchor_init.f32": anchor_raw,
        "anchor_radius_raw_init.f32": anchor_radius,
    }
    written: dict[str, dict] = {}
    for name, t in files.items():
        arr = t.detach().cpu().numpy().astype(dt)
        arr.tofile(out / name)
        written[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                         "bytes": int(arr.nbytes)}
        print(f"{name}: {written[name]}")

    (out / "init_state.json").write_text(json.dumps({
        "dtype": args.dtype,
        "num_gs_tokens": int(tokens.shape[0]),
        "token_dim": int(tokens.shape[1]),
        "files": written,
        "note": "anchor_radius_raw is a pre-softplus accumulator",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
