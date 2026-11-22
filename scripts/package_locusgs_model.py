#!/usr/bin/env python3
"""Package a 4-view, 4096-token FP32 export (256×256 or 256×448 H×W) for OpenSplat SAF import."""
import argparse
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks-dir', type=Path, required=True)
    parser.add_argument('--init-state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    names = ['encoder.onnx', *(f'block_{i:02d}.onnx' for i in range(12)), 'head.onnx']
    files = [args.blocks_dir / name for name in names]
    for name, size in {'gs_token_init.f32': 4096 * 1024 * 4,
                       'anchor_init.f32': 4096 * 3 * 4,
                       'anchor_radius_raw_init.f32': 4096 * 4}.items():
        path = args.init_state / name
        if path.stat().st_size != size:
            parser.error(f'{path}: expected {size} bytes')
        files.append(path)
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            parser.error(f'Missing or empty model file: {path}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.out, 'x', ZIP_STORED) as archive:
        for path in files:
            archive.write(path, path.name)
    print(f'{args.out}: {len(files)} files, {args.out.stat().st_size} bytes')


if __name__ == '__main__':
    main()
