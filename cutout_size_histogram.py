#!/usr/bin/env python3
"""
Parallel FITS cutout size analysis with progress logging.

This script scans a directory of FITS cutouts, reads the primary-header image
dimensions, writes a CSV summary of exact size counts, and saves a PNG figure
containing width, height, area, and exact-size distributions.
"""

import argparse
import csv
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze FITS cutout sizes and save a histogram figure."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing FITS cutouts.",
    )
    parser.add_argument(
        "--pattern",
        default="*.fits",
        help="Glob pattern used to find FITS files. Default: *.fits",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(1, os.cpu_count() or 4)),
        help="Number of parallel workers to use. Default: min(16, cpu_count).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many files. Default: 1000",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of FITS files to scan.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins. Default: 30",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Path to the output PNG figure. Default: <input_dir>/cutout_size_histogram.png",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to the output CSV summary. Default: <input_dir>/cutout_size_summary.csv",
    )
    return parser.parse_args()


def parse_fits_value(raw_value):
    value = raw_value.split("/")[0].strip()
    if value.startswith("'") and value.endswith("'"):
        return value.strip("'")
    return int(value)


def read_primary_fits_size(path):
    header_values = {}
    required_keys = {"NAXIS", "NAXIS1", "NAXIS2"}

    with path.open("rb") as handle:
        while True:
            block = handle.read(2880)
            if not block:
                raise ValueError("FITS header ended before END card")

            for offset in range(0, len(block), 80):
                card = block[offset : offset + 80]
                keyword = card[:8].decode("ascii", errors="ignore").strip()

                if keyword == "END":
                    naxis = int(header_values.get("NAXIS", 0))
                    if naxis < 2:
                        return None, f"NAXIS={naxis}"

                    missing_keys = required_keys - header_values.keys()
                    if missing_keys:
                        missing = ", ".join(sorted(missing_keys))
                        return None, f"Missing FITS header keys: {missing}"

                    width = int(header_values["NAXIS1"])
                    height = int(header_values["NAXIS2"])
                    return (
                        {
                            "filename": path.name,
                            "width": width,
                            "height": height,
                            "pixel_area": width * height,
                        },
                        None,
                    )

                if card[8:10] != b"= ":
                    continue

                if keyword in required_keys:
                    raw_value = card[10:80].decode("ascii", errors="ignore")
                    header_values[keyword] = parse_fits_value(raw_value)


def safe_read_primary_fits_size(path):
    try:
        measurement, skip_reason = read_primary_fits_size(path)
        if measurement is None:
            return None, (path.name, skip_reason)
        return measurement, None
    except Exception as exc:
        return None, (path.name, str(exc))


def discover_files(input_dir, pattern, max_files):
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    files = sorted(input_path.glob(pattern), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {input_path}")

    if max_files is not None:
        files = files[:max_files]

    return input_path, files


def scan_files(files, workers, progress_every):
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    print(
        f"Scanning {len(files)} FITS files with {workers} worker(s)...",
        flush=True,
    )

    measurements = []
    skipped_files = []
    start_time = perf_counter()

    if workers == 1:
        for index, path in enumerate(files, start=1):
            measurement, skipped = safe_read_primary_fits_size(path)
            if measurement is None:
                skipped_files.append(skipped)
            else:
                measurements.append(measurement)

            if index % progress_every == 0 or index == len(files):
                elapsed = perf_counter() - start_time
                rate = index / elapsed if elapsed else float("inf")
                print(
                    f"Processed {index}/{len(files)} files ({rate:.1f} files/s)",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(safe_read_primary_fits_size, path): path for path in files
            }

            for index, future in enumerate(as_completed(future_map), start=1):
                measurement, skipped = future.result()
                if measurement is None:
                    skipped_files.append(skipped)
                else:
                    measurements.append(measurement)

                if index % progress_every == 0 or index == len(files):
                    elapsed = perf_counter() - start_time
                    rate = index / elapsed if elapsed else float("inf")
                    print(
                        f"Processed {index}/{len(files)} files ({rate:.1f} files/s)",
                        flush=True,
                    )

    elapsed = perf_counter() - start_time
    print(f"Finished header scan in {elapsed:.2f} s", flush=True)
    return measurements, skipped_files, elapsed


def write_summary_csv(output_csv_path, size_counts):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["width", "height", "pixel_area", "count"])
        for (width, height), count in sorted(size_counts.items()):
            writer.writerow([width, height, width * height, count])


def save_summary_plot(output_plot_path, width_values, height_values, area_values, size_counts, bins):
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()

    plot_specs = [
        ("Width distribution", "Width (pixels)", width_values, "#4C72B0"),
        ("Height distribution", "Height (pixels)", height_values, "#55A868"),
        ("Area distribution", "Pixel area", area_values, "#C44E52"),
    ]

    for axis, (title, xlabel, values, color) in zip(axes[:3], plot_specs):
        axis.hist(values, bins=min(bins, len(values)), color=color, edgecolor="black")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")

    size_items = size_counts.most_common()
    if len(size_items) > 40:
        size_items = size_items[:40]
        size_title = "Exact cutout sizes (top 40)"
    else:
        size_title = "Exact cutout sizes"

    labels = [f"{width}x{height}" for (width, height), _count in size_items]
    counts = [count for (_size, count) in size_items]

    axes[3].bar(labels, counts, color="#8172B2", edgecolor="black")
    axes[3].set_title(size_title)
    axes[3].set_xlabel("Size (width x height)")
    axes[3].set_ylabel("Count")
    axes[3].tick_params(axis="x", rotation=45)

    fig.suptitle(f"Cutout size distribution ({len(width_values)} FITS files)")
    fig.savefig(output_plot_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    if args.bins < 1:
        raise ValueError("--bins must be at least 1")

    input_path, files = discover_files(args.input_dir, args.pattern, args.max_files)
    print(f"Found {len(files)} FITS files in {input_path}", flush=True)
    print("First few files:", flush=True)
    for path in files[:5]:
        print(f"  {path.name}", flush=True)

    measurements, skipped_files, elapsed = scan_files(
        files=files,
        workers=args.workers,
        progress_every=args.progress_every,
    )

    if not measurements:
        raise RuntimeError("No readable 2D FITS cutouts were found.")

    width_values = [row["width"] for row in measurements]
    height_values = [row["height"] for row in measurements]
    area_values = [row["pixel_area"] for row in measurements]
    size_counts = Counter((row["width"], row["height"]) for row in measurements)

    output_plot_path = (
        Path(args.output_plot).expanduser().resolve()
        if args.output_plot
        else input_path / "cutout_size_histogram.png"
    )
    output_csv_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else input_path / "cutout_size_summary.csv"
    )

    print("Writing CSV summary...", flush=True)
    write_summary_csv(output_csv_path, size_counts)

    print("Saving histogram figure...", flush=True)
    save_summary_plot(
        output_plot_path=output_plot_path,
        width_values=width_values,
        height_values=height_values,
        area_values=area_values,
        size_counts=size_counts,
        bins=args.bins,
    )

    print(f"Measured {len(measurements)} cutouts in {elapsed:.2f} s", flush=True)
    print(f"Skipped {len(skipped_files)} file(s)", flush=True)
    print(f"Width range: {min(width_values)} to {max(width_values)} pixels", flush=True)
    print(f"Height range: {min(height_values)} to {max(height_values)} pixels", flush=True)
    print(f"Unique size pairs: {len(size_counts)}", flush=True)
    print("Most common sizes:", flush=True)
    for (width, height), count in size_counts.most_common(10):
        print(f"  {width}x{height}: {count}", flush=True)

    print(f"CSV summary written to {output_csv_path}", flush=True)
    print(f"Histogram figure written to {output_plot_path}", flush=True)


if __name__ == "__main__":
    main()
