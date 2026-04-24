#!/usr/bin/env python3
"""Standardize FITS cutouts to a fixed 120x120 image size.

Rule:
  * If both image dimensions are at least the target size, center-crop.
  * If either image dimension is smaller than the target size, bilinear-resize.

The script preserves relative paths under the output directory and writes a CSV
manifest with the original size and action used for each FITS file. FITS tables
such as catalog files are skipped. Requires numpy and astropy.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits


def center_crop(data, target_h, target_w):
    h, w = data.shape[-2:]
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    return data[..., y0 : y0 + target_h, x0 : x0 + target_w]


def bilinear_resize(data, target_h, target_w):
    dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else np.float32
    data = data.astype(dtype, copy=False)
    h, w = data.shape[-2:]
    y = (np.arange(target_h) + 0.5) * h / target_h - 0.5
    x = (np.arange(target_w) + 0.5) * w / target_w - 0.5
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (y - y0).astype(dtype)
    wx = (x - x0).astype(dtype)

    flat = data.reshape((-1, h, w))
    top = flat[:, y0[:, None], x0[None, :]] * (1 - wx) + flat[
        :, y0[:, None], x1[None, :]
    ] * wx
    bottom = flat[:, y1[:, None], x0[None, :]] * (1 - wx) + flat[
        :, y1[:, None], x1[None, :]
    ] * wx
    out = top * (1 - wy[:, None]) + bottom * wy[:, None]
    return out.reshape(data.shape[:-2] + (target_h, target_w))


def process_one(input_path, input_dir, output_dir, target_size, overwrite):
    input_path = Path(input_path)
    rel_path = input_path.relative_to(input_dir)
    output_path = output_dir / rel_path

    if output_path.exists() and not overwrite:
        return (str(rel_path), "", "", "skipped_exists", str(output_path))

    data, header = fits.getdata(input_path, header=True, memmap=False)
    ndim = getattr(data, "ndim", 0)
    if ndim < 2:
        return (str(rel_path), "", "", f"skipped_non_image_{ndim}d", "")

    target_h = target_w = target_size
    h, w = data.shape[-2:]

    if h == target_h and w == target_w:
        output = data
        action = "unchanged"
    elif h >= target_h and w >= target_w:
        output = center_crop(data, target_h, target_w)
        action = "center_crop"
    else:
        output = bilinear_resize(data, target_h, target_w)
        action = "bilinear_resize"

    header["ORIGH"] = (int(h), "Original image height before standardization")
    header["ORIGW"] = (int(w), "Original image width before standardization")
    header["STDIM"] = (f"{target_w}x{target_h}", "Standardized image size")
    header["STDACT"] = (action, "Standardization action")
    header.add_history(f"Standardized from {w}x{h} to {target_w}x{target_h} via {action}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(output_path, output, header=header, overwrite=True)
    return (str(rel_path), h, w, action, str(output_path))


def process_one_task(task):
    return process_one(*task)


def iter_fits(input_dir, recursive):
    pattern = "**/*.fits" if recursive else "*.fits"
    return sorted(input_dir.glob(pattern))


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing input FITS cutouts.")
    parser.add_argument("--output-dir", required=True, help="Directory for standardized FITS cutouts.")
    parser.add_argument("--target-size", type=int, default=120, help="Output height/width in pixels.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes.")
    parser.add_argument("--chunksize", type=int, default=64, help="Files per process-pool work chunk.")
    parser.add_argument("--recursive", action="store_true", help="Search input-dir recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output FITS files.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV manifest path. Default: <output-dir>/standardization_manifest.csv",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output_dir / "standardization_manifest.csv"
    )

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise SystemExit("Use a different output directory; this script does not modify images in place.")
    if args.recursive and is_relative_to(output_dir, input_dir):
        raise SystemExit("For recursive runs, output-dir must not be inside input-dir.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be at least 1")

    files = iter_fits(input_dir, args.recursive)
    if not files:
        raise SystemExit(f"No FITS files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "original_height", "original_width", "action", "output_path"])
        tasks = ((path, input_dir, output_dir, args.target_size, args.overwrite) for path in files)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, row in enumerate(
                executor.map(process_one_task, tasks, chunksize=args.chunksize),
                start=1,
            ):
                writer.writerow(row)
                if i % 1000 == 0 or i == len(files):
                    print(f"Processed {i}/{len(files)} FITS files", flush=True)

    print(f"Wrote standardized cutouts to {output_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
