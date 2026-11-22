#!/usr/bin/env python3
"""Chunk FP32 attention over queries, with optional additive geometric bias.

Why: LocusGS's encoder runs self-attention over all four views jointly, so the
score tensor is `[1, 16, 4096, 4096]` float32 = 1074 MB. PyTorch uses a fused
SDPA kernel and never writes it down; the ONNX export lowers SDPA to plain
MatMul/Softmax/MatMul, so every millisecond of that matrix is real memory.
Measured peak on device was 2.01 GB, of which ~1.7 GB is the encoder.

Chunking over the *query* axis is exactly equivalent, not an approximation:
every query still computes its softmax over all 4096 keys. It only stops the
whole `[16,4096,4096]` tensor from being live at once. With a chunk of 256 the
score tensor drops to `[16,256,4096]` = 67 MB.

Numerical check (`scripts/_chunk_check.py`) is 7.2e-7 against the unchunked
computation at N=512, i.e. float noise. A full end-to-end comparison against
PyTorch is run afterwards by `verify_real.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def _find_attention_triples(graph):
    """Match private MatMul → [Add bias →] Softmax → MatMul chains."""
    nodes = list(graph.node)
    producers = {out: i for i, node in enumerate(nodes) for out in node.output}
    users = {}
    for i, node in enumerate(nodes):
        for value in node.input:
            users.setdefault(value, []).append(i)
    outputs = {x.name for x in graph.output}

    def single_user(value, i):
        return value not in outputs and users.get(value) == [i]

    for sm_idx, sm in enumerate(nodes):
        if sm.op_type != "Softmax" or sm.domain or len(sm.input) != 1 or len(sm.output) != 1:
            continue
        axis = next((a.i for a in sm.attribute if a.name == "axis"), -1)
        if axis not in (-1, 3) or sm.output[0] in outputs:
            continue
        consumers = users.get(sm.output[0], [])
        if len(consumers) != 1:
            continue
        post_idx = consumers[0]
        post = nodes[post_idx]
        if post.op_type != "MatMul" or post.domain or len(post.input) != 2 or post.input[0] != sm.output[0]:
            continue
        logits_idx = producers.get(sm.input[0])
        if logits_idx is None or not single_user(sm.input[0], sm_idx):
            continue
        logits = nodes[logits_idx]
        if logits.domain:
            continue
        if logits.op_type == "MatMul":
            yield logits_idx, None, sm_idx, post_idx, None
        elif logits.op_type == "Add" and len(logits.input) == 2:
            pre_idx = producers.get(logits.input[0])
            if pre_idx is not None and nodes[pre_idx].op_type == "MatMul" and not nodes[pre_idx].domain:
                if single_user(logits.input[0], logits_idx):
                    yield pre_idx, logits_idx, sm_idx, post_idx, logits.input[1]


def _shape_of(graph, name: str) -> list[int] | None:
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            vals = [d.dim_value if d.dim_value else 0 for d in dims]
            if all(v > 0 for v in vals):
                return vals
    return None


def chunk_graph(graph, chunk: int) -> int:
    """Rewrite matched attention chains in place; return the number rewritten."""

    if chunk <= 0:
        raise ValueError("chunk must be positive")
    # Build replacements against immutable original indices; insert at the
    # original final MatMul so the bias and V were already produced.
    original = list(graph.node)
    replacements = {}
    removed = set()
    rewritten = 0
    for pre_idx, add_idx, sm_idx, post_idx, bias in _find_attention_triples(graph):
        pre, post = original[pre_idx], original[post_idx]

        if len(pre.input) != 2 or len(post.input) != 2:
            print(f"  skip {pre.output[0][:50]}: wrong arity")
            continue
        q_name, k_name = pre.input[0], pre.input[1]
        v_name = post.input[1]
        final_out = post.output[0]

        q_shape = _shape_of(graph, q_name)
        if q_shape is None:
            print(f"  skip {pre.output[0][:50]}: query shape unknown")
            continue
        if len(q_shape) != 4:
            print(f"  skip {pre.output[0][:50]}: rank {len(q_shape)}")
            continue
        seq = q_shape[2]
        if seq <= chunk:
            continue
        if bias is not None:
            bias_shape = _shape_of(graph, bias)
            k_shape = _shape_of(graph, k_name)
            if (bias_shape is None or k_shape is None or len(bias_shape) != 4 or
                    len(k_shape) != 4 or bias_shape[2:] != [seq, k_shape[-1]] or
                    any(b not in (1, q) for b, q in zip(bias_shape[:2], q_shape[:2]))):
                continue

        prefix = f"locusgs_query_chunk_{post_idx}"
        new_nodes = []
        slices = []
        offset = 0
        part = 0
        while offset < seq:
            length = min(chunk, seq - offset)
            sname = f"{prefix}_slice_{part}"
            new_nodes.append(helper.make_node(
                "Slice", [q_name, f"{prefix}_starts_{part}", f"{prefix}_ends_{part}",
                          f"{prefix}_axes_{part}"], [sname],
                name=f"{prefix}_slice_node_{part}"))
            graph.initializer.extend([
                numpy_helper.from_array(np.array([offset], np.int64), f"{prefix}_starts_{part}"),
                numpy_helper.from_array(np.array([offset + length], np.int64), f"{prefix}_ends_{part}"),
                numpy_helper.from_array(np.array([2], np.int64), f"{prefix}_axes_{part}"),
            ])
            mm = f"{prefix}_qk_{part}"
            smf = f"{prefix}_sm_{part}"
            pv = f"{prefix}_pv_{part}"
            new_nodes.append(helper.make_node(
                "MatMul", [sname, k_name], [mm], name=f"{prefix}_qk_node_{part}"))
            if bias is not None:
                b = f"{prefix}_bias_{part}"
                scores = f"{prefix}_scores_{part}"
                new_nodes.extend([
                    helper.make_node("Slice", [bias, f"{prefix}_starts_{part}",
                        f"{prefix}_ends_{part}", f"{prefix}_axes_{part}"], [b]),
                    helper.make_node("Add", [mm, b], [scores]),
                ])
                mm = scores
            new_nodes.append(helper.make_node(
                "Softmax", [mm], [smf], axis=-1, name=f"{prefix}_sm_node_{part}"))
            new_nodes.append(helper.make_node(
                "MatMul", [smf, v_name], [pv], name=f"{prefix}_pv_node_{part}"))
            slices.append(pv)
            offset += length
            part += 1

        new_nodes.append(helper.make_node(
            "Concat", slices, [final_out], axis=2, name=f"{prefix}_concat"))

        removed.update((pre_idx, sm_idx))
        if add_idx is not None:
            removed.add(add_idx)
        replacements[post_idx] = new_nodes
        rewritten += 1
        print(f"  chunked {final_out[:60]} seq={seq} -> {part} x {chunk}")
    if replacements:
        del graph.node[:]
        for i, node in enumerate(original):
            if i in replacements:
                graph.node.extend(replacements[i])
            elif i not in removed:
                graph.node.append(node)

    return rewritten


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if src.resolve() == out.resolve():
        ap.error("--out must differ from --src")
    total = 0
    for p in sorted(src.glob("*.onnx")):
        m = onnx.load(str(p), load_external_data=False)
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
        n = chunk_graph(m.graph, args.chunk)
        onnx.checker.check_model(m)
        target = out / p.name
        onnx.save(m, str(target))
        print(f"{p.name}: {n} attention blocks chunked")
        total += n
    print(f"total rewritten: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
