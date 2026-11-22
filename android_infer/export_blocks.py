#!/usr/bin/env python3
"""Export LocusGS as per-block graphs instead of one monolithic decoder.

Why: exporting all 12 decoder blocks as a single graph produced outputs that
diverged from PyTorch by up to 4.8 (max abs) even though each block exported
correctly in isolation (1e-5). Rather than debug the legacy exporter's handling
of the long loop, we export each block separately and chain them in the runtime.
This is also the better mobile architecture:

  * one ~66 MB block resident at a time instead of a 788 MB graph,
  * buffers freed between blocks, which is what makes 8 GB phones viable,
  * a per-block failure is attributable, not a 12-layer mystery.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "LocusGS"))
sys.path.insert(0, str(REPO))

from android_infer.locusgs_export import (  # noqa: E402
    EncoderGraph,
    HeadGraph,
    _onnx_patch_ops,
    build_model,
    load_weights,
)


class BlockGraph(nn.Module):
    """One decoder block plus its anchor refinement.

    Mirrors ``decode_tokens`` exactly: block -> pos_act(xyz_embed(tokens)) added
    to the raw anchor, and radius refined when enabled.
    """

    def __init__(self, model, index: int):
        super().__init__()
        self.model = model
        self.index = index
        self.refines_radius = bool(model.opt.anchor_radius_refinement)

    @classmethod
    def build(cls, model, index: int) -> nn.Module:
        block = model.enc_dec_backbone.decoder_blocks[index]
        refines_radius = bool(model.opt.anchor_radius_refinement)
        uses_radius_bias = bool(model.opt.anchor_radius_affects_bias)
        xyz_embed = model.xyz_embed
        radius_embed = model.radius_embed if refines_radius else None
        pos_act = model.activation_head.pos_act
        activate_radius = model.activate_anchor_radius
        normalize_anchor = model.normalize_anchor

        class _B(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = block
                self.xyz_embed = xyz_embed
                if radius_embed is not None:
                    self.radius_embed = radius_embed

            def forward(self, tokens, anchor_raw, anchor_radius_raw, keys, values, patch_rays):
                # Cast inputs to the block's own precision: with mixed-precision
                # export the encoder emits fp32 while the blocks are fp16, and
                # torch refuses a dtype mismatch at the first Linear.
                # Cast to the linear weights' dtype. LayerNorm stays fp32, so
                # the block's own code promotes back where it needs to; doing the
                # cast here keeps the graph free of implicit promotions that the
                # ONNX exporter cannot represent.
                lin = next(
                    (m for m in self.block.modules() if isinstance(m, nn.Linear)),
                    None,
                )
                if lin is not None and tokens.dtype != lin.weight.dtype:
                    tokens = tokens.to(lin.weight.dtype)
                    anchor_raw = anchor_raw.to(lin.weight.dtype)
                    anchor_radius_raw = anchor_radius_raw.to(lin.weight.dtype)
                if lin is not None and keys.dtype != lin.weight.dtype:
                    keys = keys.to(lin.weight.dtype)
                    values = values.to(lin.weight.dtype)
                    patch_rays = patch_rays.to(lin.weight.dtype)
                out = self.block(
                    gs_tokens=tokens,
                    gs_anchor_normalized=normalize_anchor(anchor_raw),
                    gs_anchor_radius=activate_radius(anchor_radius_raw) if uses_radius_bias else None,
                    keys=keys,
                    values=values,
                    plucker_emb_for_concat=None,
                    plucker_rays_patch_raw=patch_rays,
                    gs_anchor_raw=anchor_raw,
                    handcraft=False,
                    return_attn=False,
                )
                new_raw = anchor_raw + pos_act(self.xyz_embed(out))
                if refines_radius:
                    anchor_radius_raw = anchor_radius_raw + self.radius_embed(out)
                return out, new_raw, anchor_radius_raw

        m = _B()
        m.index = index
        return m


class SplitHeadGraph(HeadGraph):
    """The split decoder carries raw radii; the unsplit decoder already activates them."""

    def forward(self, tokens, anchor_raw, anchor_radius_raw):
        return super().forward(tokens, anchor_raw, self.model.activate_anchor_radius(anchor_radius_raw))


def apply_decoder_precision(model, dtype) -> None:
    """Convert only the decoder path to ``dtype``; the encoder is untouched.

    Verified necessary: converting the whole model (``model.half()``) turned the
    encoder fp16 as well, costing 0.67 % relative error at its output which
    compounded to 9 % on the final Gaussians.
    """
    if dtype is not torch.float16:
        return
    # Whole decoder path to fp16. Keeping LayerNorm fp32 is tempting but ONNX's
    # LayerNormalization op requires one dtype per node, so a split-precision
    # block is not representable; the accuracy that matters comes from keeping
    # the *encoder* fp32 (see module docstring).
    for p_ in model.enc_dec_backbone.decoder_blocks.parameters():
        p_.data = p_.data.half()
    for mt in (model.activation_head, model.xyz_embed, model.radius_embed):
        for p_ in mt.parameters():
            p_.data = p_.data.half()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", default="256,256")
    ap.add_argument("--num-gs-tokens", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                    help="fp16 stores weights in half precision; use --encoder-dtype "
                         "to keep the sensitive encoder in fp32")
    ap.add_argument("--encoder-dtype", default=None, choices=["fp32", "fp16"],
                    help="override the encoder precision (defaults to --dtype)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _onnx_patch_ops()

    opt, model = build_model(args)
    load_report = load_weights(model, args.checkpoint)
    if load_report["mismatched"]:
        # Not fatal: the checkpoint may carry tensors for a different variant.
        # But a mismatch on gs_tokens/anchor_* means the "pretrained" model is
        # partly random, which silently destroys warm-start fidelity.
        print(f"[WARN] {len(load_report['mismatched'])} tensors did not load "
              f"({load_report['loaded']} did)")
        for m in load_report["mismatched"][:5]:
            print("   ", m)
    model.eval()

    h, w = opt.img_size
    B, V = 1, 4
    dt = torch.float16 if args.dtype == "fp16" else torch.float32
    enc_dtype = args.encoder_dtype or args.dtype
    enc_dt = torch.float16 if enc_dtype == "fp16" else torch.float32

    # The encoder is converted for its own export and converted back, so the
    # decoder keeps whatever precision the caller asked for. fp16 in every stage
    # measured 9% relative error on the final Gaussians; the encoder alone
    # contributes 0.67% of that, which is why it can be kept wider.
    enc_report: dict = {"dtype": enc_dtype}

    # Convert the encoder's *weights* when fp16 is requested. Previously only
    # the inputs were cast, which made the first Conv see a half input and a
    # float weight: "Input type (c10::Half) and bias type (float) should be the
    # same". The encoder is converted in place here and restored to fp32 after
    # its graph is exported, so the decoder export is unaffected.
    encoder_was_fp32 = enc_dt == torch.float32

    # Encoder inputs follow the encoder's precision, not the decoder's.
    images = torch.randn(B, V, 3, h, w, dtype=enc_dt)
    plucker = torch.randn(B, V, 6, h, w, dtype=enc_dt)

    report: dict = {"dtype": args.dtype, "graphs": {}, "n_blocks": None,
                    "load": {"loaded": load_report["loaded"],
                             "mismatched": load_report["mismatched"]}}

    with torch.no_grad():
        keys, values, patch = EncoderGraph(model)(images, plucker)
        keys, values, patch = keys.float(), values.float(), patch.float()

        tokens = model.get_gs_tokens(1).to(dt)
        anchor_raw = model.gs_anchor_xyz_raw.unsqueeze(0).to(dt)
        anchor_radius = model.gs_anchor_radius_raw.unsqueeze(0).to(dt)
        n_blocks = len(model.enc_dec_backbone.decoder_blocks)
        report["n_blocks"] = n_blocks
        for i in range(n_blocks):
            blk = BlockGraph.build(model, i)
            tokens, anchor_raw, anchor_radius = blk(
                tokens, anchor_raw, anchor_radius, keys, values, patch
            )
        gaussians = SplitHeadGraph(model)(tokens, anchor_raw, anchor_radius)

    report["shapes"] = {
        "keys": list(keys.shape),
        "patch_rays": list(patch.shape),
        "tokens": list(tokens.shape),
        "anchor_raw": list(anchor_raw.shape),
        "anchor_radius": list(anchor_radius.shape),
        "gaussians": list(gaussians.shape),
    }
    report["gaussian_count"] = int(gaussians.shape[1])

    # Encoder is exported on its own precision; decoder blocks and head follow
    # --dtype. Blocks receive fp32 encoder outputs and are cast internally.
    specs = [("encoder", EncoderGraph(model), (images, plucker),
              ["images", "plucker"])]
    for i in range(n_blocks):
        specs.append((f"block_{i:02d}", BlockGraph.build(model, i),
                      (tokens, anchor_raw, anchor_radius, keys, values, patch),
                      ["tokens", "anchor_raw", "anchor_radius_raw",
                       "keys", "values", "patch_rays"]))
    specs.append(("head", SplitHeadGraph(model), (tokens, anchor_raw, anchor_radius),
                  ["tokens", "anchor_raw", "anchor_radius_raw"]))

    if dt is torch.float16:
        apply_decoder_precision(model, torch.float16)
        # Blocks and head were traced in fp32 above; re-trace them in fp16.
        with torch.no_grad():
            tokens = tokens.to(dt)
            anchor_raw = anchor_raw.to(dt)
            anchor_radius = anchor_radius.to(dt)
            for i in range(n_blocks):
                tokens, anchor_raw, anchor_radius = BlockGraph.build(model, i)(
                    tokens, anchor_raw, anchor_radius, keys.to(dt), values.to(dt), patch.to(dt)
                )
            gaussians = SplitHeadGraph(model)(tokens, anchor_raw, anchor_radius)
        specs = [specs[0]] + [
            (f"block_{i:02d}", BlockGraph.build(model, i),
             (tokens, anchor_raw, anchor_radius, keys.to(dt), values.to(dt), patch.to(dt)),
             ["tokens", "anchor_raw", "anchor_radius_raw", "keys", "values", "patch_rays"])
            for i in range(n_blocks)
        ] + [("head", SplitHeadGraph(model), (tokens, anchor_raw, anchor_radius),
              ["tokens", "anchor_raw", "anchor_radius_raw"])]

    for name, module, inputs, names in specs:
        path = out / f"{name}.onnx"
        try:
            torch.onnx.export(module, inputs, str(path), opset_version=args.opset,
                              input_names=names, do_constant_folding=True, dynamo=False)
            report["graphs"][name] = {"status": "ok", "bytes": path.stat().st_size}
        except Exception as exc:
            report["graphs"][name] = {"status": "failed",
                                      "error": f"{type(exc).__name__}: {exc}"[:800]}


    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "graphs"}, indent=2))
    failed = [k for k, v in report["graphs"].items() if v["status"] != "ok"]
    print("failed:", failed if failed else "none")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
