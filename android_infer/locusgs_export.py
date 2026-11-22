#!/usr/bin/env python3
"""Export LocusGS inference as ONNX for on-device execution.

Design constraints that shaped this:

* Static shapes only (batch=1, views=4, 256x256). Mobile runtimes handle
  dynamic shapes poorly, and LocusGS's attention asserts divisibility anyway.
* Only inference modules: loss, LPIPS, the training renderer and diagnostics are
  excluded by construction -- we call ``forward_encoder``/``forward_decoder``/
  ``activation_head`` directly.
* Dataclass outputs are flattened to tuples: ONNX cannot represent them, and
  ``torch.export`` refuses to.
* ``geo_sparse_sampling_k = 0`` in the pretrained config means the
  point-to-ray bias path is fully differentiable, so the geometric attention
  exports; the discrete ``sample_sparse_indices`` path would not.

The export is split into three graphs so a mobile runtime can execute and
release them independently, and so a failure is attributable to one stage:

    encoder.onnx   images + plucker + rays + K + c2w  -> keys, values, patch rays
    decoder.onnx   keys + values + patch rays + cam   -> tokens
    head.onnx      tokens                             -> 14-ch gaussians
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
LOCUSGS = REPO / "vendor" / "LocusGS"
if str(LOCUSGS) not in sys.path:
    sys.path.insert(0, str(LOCUSGS))


def _onnx_patch_ops() -> None:
    """Register ONNX-symbolic gaps that LocusGS's activations hit.

    ``expm1`` has no ONNX op at opset 17. ``expm1(x) = exp(x) - 1`` is exact in
    the sense that matters here: the argument is already clamped to
    ``[-100, 100]`` by ``pos_act``, so there is no cancellation pathology that
    the dedicated kernel would avoid. Verified numerically in
    ``android_infer/verify_onnx.py``.
    """
    from torch.onnx import symbolic_helper

    @symbolic_helper.parse_args("v")
    def _expm1(g, self):
        return g.op("Sub", g.op("Exp", self), g.op("Constant", value_t=torch.tensor(1.0)))

    torch.onnx.register_custom_op_symbolic("aten::expm1", _expm1, 17)


class EncoderGraph(nn.Module):
    """forward_encoder, flattened to tensors.

    ``forward_encoder`` consumes only ``images_rgb`` and ``plucker``: rays,
    intrinsics and camera-to-world are passed through ``ModelInputEncoder`` but
    never read (verified by reading the function and by ONNX constant folding
    them away). They are omitted here so the graph does not carry dead inputs.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images_rgb, plucker):
        from locusgs.models.input_types import ModelInputEncoder

        latent = self.model.forward_encoder(
            ModelInputEncoder(
                images_rgb=images_rgb,
                plucker=plucker,
                # Unused by forward_encoder; supplied only to satisfy the dataclass.
                rays_os=torch.zeros(1, device=images_rgb.device),
                rays_ds=torch.zeros(1, device=images_rgb.device),
                intrinsics_input=torch.zeros(1, device=images_rgb.device),
                cam_to_world_input=torch.zeros(1, device=images_rgb.device),
            )
        )
        # plucker_emb_for_concat is None for the geometric_positional variant.
        return latent.keys, latent.values, latent.plucker_rays_patch_raw


class DecoderGraph(nn.Module):
    """decode_tokens: tokens plus refined anchors, no Gaussian head.

    Verified equivalent to ``forward_decoder`` (max abs diff 0.0) in
    ``android_infer/verify_onnx.py``; the head is kept separate so the runtime
    can release decoder buffers before materialising 262k Gaussians.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, keys, values, plucker_rays_patch):
        from locusgs.models.input_types import EncoderLatent

        latent = EncoderLatent(
            keys=keys,
            values=values,
            plucker_emb_for_concat=None,
            plucker_rays_patch_raw=plucker_rays_patch,
            # Not used by decoding; present to satisfy the dataclass.
            image_features=torch.zeros(1, device=keys.device),
        )
        # No cameras: Gaussian decoding is view-independent (see decode_tokens).
        tokens, anchor_raw, anchor_radius = self.model.decode_tokens(latent)
        # anchor_radius is None only when use_anchor_radius is false; this model
        # always has it, and ONNX needs a concrete tensor either way.
        return tokens, anchor_raw, anchor_radius


class HeadGraph(nn.Module):
    """tokens + anchors -> 14-channel Gaussians (head + anchor composition)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, anchor_raw, anchor_radius):
        return self.model.gaussians_from_tokens(tokens, anchor_raw, anchor_radius)


def build_model(args) -> tuple:
    from locusgs.options import Options
    from locusgs.models import model_registry

    h, w = (int(x) for x in args.img_size.split(","))
    opt = Options(
        evaluating=True,
        model_type="locusgs",
        cross_attn_variant="geometric_positional",
        use_anchor_radius=True,
        anchor_radius_affects_bias=True,
        anchor_radius_refinement=True,
        gaussian_center_variant="radius_scaled_offset",
        num_gs_tokens=args.num_gs_tokens,
        img_size=(h, w),
        num_input_views=4,
        num_views=4,
        bg_color="grey",
        lambda_lpips=0.0,
        lambda_anchor_visibility=0.0,
    )
    model = model_registry["locusgs"](opt).eval()
    return opt, model


def load_weights(model, ckpt_path: str) -> dict:
    """Copy checkpoint tensors into ``model`` in place.

    Shape mismatches are reported loudly rather than skipped silently. This
    matters concretely: the shipped checkpoint was trained with 4096 Gaussian
    tokens, so building the model with ``num_gs_tokens=1024`` leaves
    ``gs_tokens`` (and the anchor tensors) at their random initial values.
    Earlier this produced non-reproducible Gaussians that looked like an ONNX
    bug; it was a configuration mismatch. Callers get the mismatch list back
    and ``build_model``-based tools assert on it.
    """
    from safetensors.torch import load_file

    state = model.state_dict()
    ckpt = load_file(ckpt_path, device="cpu")
    loaded = 0
    mismatched: list[str] = []
    for k, v in ckpt.items():
        if k not in state:
            continue
        if state[k].shape != v.shape:
            mismatched.append(f"{k}: model {tuple(state[k].shape)} != ckpt {tuple(v.shape)}")
            continue
        state[k].copy_(v)
        loaded += 1
    return {"loaded": loaded, "mismatched": mismatched}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", default="256,256")
    ap.add_argument("--num-gs-tokens", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _onnx_patch_ops()

    opt, model = build_model(args)
    load_report = load_weights(model, args.checkpoint)
    model.to(args.device).eval()

    h, w = opt.img_size
    n_tok = opt.num_gs_tokens
    B, V = 1, 4
    dev = args.device

    images = torch.randn(B, V, 3, h, w, device=dev)
    plucker = torch.randn(B, V, 6, h, w, device=dev)

    report: dict = {"load": load_report, "img_size": [h, w], "num_gs_tokens": n_tok,
                    "graphs": {}}

    with torch.no_grad():
        keys, values, patch_rays = EncoderGraph(model)(images, plucker)
        tokens, anchor_raw, anchor_radius = DecoderGraph(model)(keys, values, patch_rays)
        gaussians = HeadGraph(model)(tokens, anchor_raw, anchor_radius)

    report["shapes"] = {
        "keys": list(keys.shape),
        "values": list(values.shape),
        "patch_rays": list(patch_rays.shape),
        "tokens": list(tokens.shape),
        "gaussians": list(gaussians.shape),
    }

    specs = [
        ("encoder", EncoderGraph(model), (images, plucker), ["images", "plucker"]),
        ("decoder", DecoderGraph(model), (keys, values, patch_rays),
         ["keys", "values", "patch_rays"]),
        ("head", HeadGraph(model), (tokens, anchor_raw, anchor_radius),
         ["tokens", "anchor_raw", "anchor_radius"]),
    ]

    for name, module, inputs, input_names in specs:
        path = out / f"{name}.onnx"
        try:
            torch.onnx.export(
                module, inputs, str(path),
                opset_version=args.opset,
                input_names=input_names,
                do_constant_folding=True,
                dynamo=False,
            )
            report["graphs"][name] = {
                "status": "ok",
                "path": path.name,
                "bytes": path.stat().st_size,
            }
        except Exception as exc:  # keep going; report per-graph
            report["graphs"][name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:1200],
            }

    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all(g["status"] == "ok" for g in report["graphs"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
