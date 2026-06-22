#!/usr/bin/env python
"""Structurally prune the base StandardUnet and export a small single-channel ONNX
for browser-side (onnxruntime-web / WebGPU) inference.

The full model is ~792M params / 3.17 GB fp32 — far too big for the browser, with
single 1.29 GB conv weights at the 3456-channel bottleneck that exceed WebGPU
buffer limits. This keeps the highest-L2 channels in every layer (DepGraph-based
structured pruning, so U-Net skip connections stay consistent), and slices the
output head to a single channel. No LoRA, no retraining — accuracy WILL drop;
this is for getting a browser-runnable demo model.

Requires: pip install torch-pruning   (plus the usual torch + fly_organelles env)

Usage:
    python scripts/prune_onnx.py -o web/models/mito_small.onnx --ratio 0.85

Pruning ratio p removes that fraction of channels per layer; params scale ~(1-p)^2
for conv->conv layers, so 0.85 takes ~792M -> ~20M params (bottleneck 3456 -> ~518).
Raise --ratio if it's still too big, lower it to keep more accuracy.
"""
import argparse

# Defaults from models/setup16_lsd_400k_all_16_finetuned_20260427_135909.py
BASE_CHECKPOINT = "/groups/cellmap/cellmap/zouinkhim/exp_mito/runs/setup_16/model_checkpoint_400000"
NUM_CLASSES = 13


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", required=True, help="output .onnx path")
    ap.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    ap.add_argument("--classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--keep-channel", type=int, default=0, help="output channel to keep")
    ap.add_argument("--ratio", type=float, default=0.85, help="fraction of channels to prune per layer")
    ap.add_argument("--input-size", type=int, nargs=3, default=[178, 178, 178])
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch_pruning as tp
    from fly_organelles.model import StandardUnet

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Rebuild the base model exactly like load_base_model() (no LoRA).
    backbone = StandardUnet(args.classes)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state_dict"])
    model = nn.Sequential(backbone, nn.Sigmoid()).to(device).eval()

    z, y, x = args.input_size
    example = torch.zeros(1, 1, z, y, x, device=device)

    # Locate the 1x1x1 output head (out_channels == classes) and keep its count
    # fixed during pruning; we slice it to one channel afterwards.
    final_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv3d) and m.out_channels == args.classes and m.kernel_size == (1, 1, 1):
            final_conv = m
    if final_conv is None:
        raise RuntimeError("could not locate the output conv (out_channels == classes, 1x1x1)")

    n0 = sum(p.numel() for p in model.parameters())
    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example,
        importance=imp,
        pruning_ratio=args.ratio,
        ignored_layers=[final_conv],
    )
    pruner.step()

    # Slice the output head to a single channel (the requested class).
    k = args.keep_channel
    with torch.no_grad():
        w = final_conv.weight.data[k : k + 1].clone()
        b = final_conv.bias.data[k : k + 1].clone() if final_conv.bias is not None else None
    final_conv.weight = nn.Parameter(w)
    if b is not None:
        final_conv.bias = nn.Parameter(b)
    final_conv.out_channels = 1

    n1 = sum(p.numel() for p in model.parameters())
    print(f"params: {n0:,} -> {n1:,}  ({n1 / n0:.3%}); ~{n1 * 4 / 1e6:.1f} MB fp32")

    with torch.no_grad():
        out = model(example)
    print(f"output shape: {tuple(out.shape)} (expected (1, 1, 56, 56, 56))")

    # Small now -> a single self-contained .onnx (no external data).
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model.cpu(),
            example.cpu(),
            args.output,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
    sz = Path(args.output).stat().st_size / 1e6
    print(f"wrote {args.output} ({sz:.1f} MB)")
    if sz > 100:
        print("⚠ still >100 MB — raise --ratio for the browser.")


if __name__ == "__main__":
    main()
