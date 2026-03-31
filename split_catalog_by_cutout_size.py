#!/usr/bin/env python3
"""
Split a Hyrax FITS catalog into two catalogs based on cutout size.

The split is performed at the object level, not the individual row level, so all
filter rows belonging to the same object_id stay together. This avoids producing
partial multi-filter objects that would be awkward for FitsImageDataSet.
"""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from astropy.table import Table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a Hyrax catalog into <= threshold and > threshold cutout-size catalogs."
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Input FITS catalog path.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing the FITS cutout files referenced by the catalog.",
    )
    parser.add_argument(
        "--width-threshold",
        type=int,
        default=120,
        help="Width threshold in pixels for the split. Default: 120",
    )
    parser.add_argument(
        "--height-threshold",
        type=int,
        default=120,
        help="Height threshold in pixels for the split. Default: 120",
    )
    parser.add_argument(
        "--object-id-column",
        default="object_id",
        help="Catalog column containing object IDs. Default: object_id",
    )
    parser.add_argument(
        "--filename-column",
        default="filename",
        help="Catalog column containing FITS filenames. Default: filename",
    )
    parser.add_argument(
        "--low-output",
        default=None,
        help="Output path for the <= threshold catalog. Default: <catalog_stem>_le_<WxH>.fits",
    )
    parser.add_argument(
        "--high-output",
        default=None,
        help="Output path for the > threshold catalog. Default: <catalog_stem>_gt_<WxH>.fits",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(1, os.cpu_count() or 4)),
        help="Number of worker threads to use for header reads. Default: min(16, cpu_count).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N objects. Default: 500",
    )
    return parser.parse_args()


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
                        raise ValueError(f"Expected 2D FITS image but found NAXIS={naxis}")

                    missing_keys = required_keys - header_values.keys()
                    if missing_keys:
                        missing = ", ".join(sorted(missing_keys))
                        raise ValueError(f"Missing FITS header keys: {missing}")

                    return int(header_values["NAXIS1"]), int(header_values["NAXIS2"])

                if card[8:10] != b"= ":
                    continue

                if keyword in required_keys:
                    raw_value = card[10:80].decode("ascii", errors="ignore")
                    value = raw_value.split("/")[0].strip()
                    header_values[keyword] = int(value)


def resolve_output_paths(catalog_path, width_threshold, height_threshold, low_output, high_output):
    suffix = f"{width_threshold}x{height_threshold}"
    if low_output is None:
        low_path = catalog_path.with_name(f"{catalog_path.stem}_le_{suffix}.fits")
    else:
        low_path = Path(low_output).expanduser().resolve()

    if high_output is None:
        high_path = catalog_path.with_name(f"{catalog_path.stem}_gt_{suffix}.fits")
    else:
        high_path = Path(high_output).expanduser().resolve()

    return low_path, high_path


def add_or_replace_column(table, name, values):
    if name in table.colnames:
        table.replace_column(name, values)
    else:
        table[name] = values


def inspect_object(object_id, filenames, image_dir):
    sizes = {}
    for filename in filenames:
        path = image_dir / str(filename)
        if not path.exists():
            raise FileNotFoundError(f"Catalog references missing file: {path}")
        sizes[str(filename)] = read_primary_fits_size(path)

    unique_sizes = sorted(set(sizes.values()))
    if len(unique_sizes) != 1:
        raise ValueError(
            f"Object {object_id} has inconsistent cutout sizes across filters: {unique_sizes}"
        )

    width, height = unique_sizes[0]
    return {
        "object_id": object_id,
        "width": width,
        "height": height,
        "pixel_area": width * height,
    }


def build_object_groups(table, object_id_column, filename_column):
    groups = defaultdict(list)
    for row in table:
        groups[row[object_id_column]].append(row[filename_column])
    return groups


def split_catalog(table, object_sizes, object_id_column, width_threshold, height_threshold):
    low_object_ids = {
        object_id
        for object_id, info in object_sizes.items()
        if info["width"] <= width_threshold and info["height"] <= height_threshold
    }
    high_object_ids = set(object_sizes) - low_object_ids

    low_mask = [row[object_id_column] in low_object_ids for row in table]
    high_mask = [row[object_id_column] in high_object_ids for row in table]

    low_table = table[low_mask]
    high_table = table[high_mask]

    add_or_replace_column(
        low_table,
        "cutout_width",
        [object_sizes[row[object_id_column]]["width"] for row in low_table],
    )
    add_or_replace_column(
        low_table,
        "cutout_height",
        [object_sizes[row[object_id_column]]["height"] for row in low_table],
    )
    add_or_replace_column(
        low_table,
        "cutout_pixel_area",
        [object_sizes[row[object_id_column]]["pixel_area"] for row in low_table],
    )

    add_or_replace_column(
        high_table,
        "cutout_width",
        [object_sizes[row[object_id_column]]["width"] for row in high_table],
    )
    add_or_replace_column(
        high_table,
        "cutout_height",
        [object_sizes[row[object_id_column]]["height"] for row in high_table],
    )
    add_or_replace_column(
        high_table,
        "cutout_pixel_area",
        [object_sizes[row[object_id_column]]["pixel_area"] for row in high_table],
    )

    return low_table, high_table, low_object_ids, high_object_ids


def main():
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    catalog_path = Path(args.catalog).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    low_output_path, high_output_path = resolve_output_paths(
        catalog_path=catalog_path,
        width_threshold=args.width_threshold,
        height_threshold=args.height_threshold,
        low_output=args.low_output,
        high_output=args.high_output,
    )

    print(f"Reading catalog: {catalog_path}", flush=True)
    table = Table.read(catalog_path)

    if args.object_id_column not in table.colnames:
        raise ValueError(f"Catalog is missing object ID column: {args.object_id_column}")
    if args.filename_column not in table.colnames:
        raise ValueError(f"Catalog is missing filename column: {args.filename_column}")
    if not image_dir.exists() or not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")

    object_groups = build_object_groups(table, args.object_id_column, args.filename_column)
    print(
        f"Catalog has {len(table)} rows and {len(object_groups)} unique objects",
        flush=True,
    )
    print(
        f"Splitting on <= {args.width_threshold}x{args.height_threshold} vs > {args.width_threshold}x{args.height_threshold}",
        flush=True,
    )

    object_sizes = {}
    start_time = perf_counter()

    print(
        f"Inspecting cutout sizes with {args.workers} worker(s)...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(inspect_object, object_id, filenames, image_dir): object_id
            for object_id, filenames in object_groups.items()
        }

        for index, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            object_sizes[result["object_id"]] = result

            if index % args.progress_every == 0 or index == len(future_map):
                elapsed = perf_counter() - start_time
                rate = index / elapsed if elapsed else float("inf")
                print(
                    f"Processed {index}/{len(future_map)} objects ({rate:.1f} objects/s)",
                    flush=True,
                )

    elapsed = perf_counter() - start_time
    print(f"Finished size inspection in {elapsed:.2f} s", flush=True)

    low_table, high_table, low_object_ids, high_object_ids = split_catalog(
        table=table,
        object_sizes=object_sizes,
        object_id_column=args.object_id_column,
        width_threshold=args.width_threshold,
        height_threshold=args.height_threshold,
    )

    low_output_path.parent.mkdir(parents=True, exist_ok=True)
    high_output_path.parent.mkdir(parents=True, exist_ok=True)
    low_table.write(low_output_path, overwrite=True)
    high_table.write(high_output_path, overwrite=True)

    print(
        f"Low-size catalog: {len(low_table)} rows, {len(low_object_ids)} objects -> {low_output_path}",
        flush=True,
    )
    print(
        f"High-size catalog: {len(high_table)} rows, {len(high_object_ids)} objects -> {high_output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
