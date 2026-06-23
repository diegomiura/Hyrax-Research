"""
Command-line utilities for working with IllustrisTNG HSC FITS imagery.

This module exposes two primary operations:
    * fetch: collect and store the list of available parent FITS file URLs.
    * split: download parent FITS files, optionally split them into filters,
      and write an optional Hyrax-compatible catalog.
    * analyze-sizes: inspect a directory of FITS cutouts and summarize their
      size distribution.
Use ``python download_hsc_pipeline.py --help`` for invocation details.
"""

#-------------Imports-------------#

import csv
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests
from astropy.io import fits
from astropy.table import Table
import argparse







#-------------Fetch-------------#

def make_list_of_urls(API_KEY=None,
                      BASE_API_URL='https://www.tng-project.org/',
                      ENDING='/api/TNG50-1/files/skirt_images_hsc/',
                      SNAPSHOT_FILTER='_realistic_v2_91',
                      output_path='all_file_urls.txt'):
    '''
    Fetches all FITS image URLs from the TNG50-1 API, filters by snapshot tag,
    and writes them to a text file.

    Args:
        API_KEY (str): API key for authenticating with the TNG50-1 API.
        BASE_API_URL (str, optional): Base URL of the TNG API.
        ENDING (str, optional): API endpoint path for HSC FITS files.
        SNAPSHOT_FILTER (str, optional): Substring to filter snapshot URLs.
        output_path (str, optional): Destination file for the URL list. Defaults to 'all_file_urls.txt'.

    Writes:
        The URL list to `output_path`, containing one URL per line.
    '''
    # Retrive data from a given API endpoint
    def get_endpoint(url):
        r = requests.get(url, headers={'API-Key': API_KEY})
        r.raise_for_status()
        return r.json()

    snapshot_urls = get_endpoint(BASE_API_URL + ENDING)
    filtered = [u for u in snapshot_urls if SNAPSHOT_FILTER in u]
    all_urls = []
    for url in filtered:
        all_urls += get_endpoint(url)['files']

    with open(output_path, 'w') as f:
        for u in all_urls:
            f.write(u + '\n')









#-------------Split-------------#

def download_and_split_hsc_images(
    split_output_dir='split_images',
    URL_LIST=None,
    BATCH_START=None,
    BATCH_SIZE=None,
    API_KEY=None,
    remove_parent: bool = False,
    catalog_path=None,
    parent_file_only: bool = False,
    parent_output_dir: str = None
):
    '''
    Downloads and splits HSC survey FITS images from the TNG50-1 API into individual filters,
    optionally removes the original parent FITS files, and can generate a catalog compatible with Hyrax.

    Args:
        split_output_dir (str, optional): Directory to save split FITS images. Defaults to 'split_images'.
        URL_LIST (str): Path to a text file containing one URL per line.
        BATCH_START (int, optional): Starting index for the batch of URLs to download. Defaults to 0.
        BATCH_SIZE (int, optional): Number of URLs to process in this batch. If None, processes all remaining URLs.
        API_KEY (str): API key required to access the TNG50-1 API.
        remove_parent (bool, optional): If True, delete the original downloaded FITS file after splitting. Defaults to False.
        catalog_path (str, optional): If provided, saves a Hyrax-compatible FITS catalog at this location.
            The catalog will include columns: 'object_id' (an integer composed of snapshot and subhalo), 'filename', and 'filter'.
        parent_file_only (bool, optional): If True, only download the parent FITS files and skip splitting and catalog creation. Defaults to False.
        parent_output_dir (str, optional): Directory to save downloaded parent FITS files.
            If None, uses split_output_dir. Defaults to None.

    Notes:
        - Split FITS images will be named as: SNAPSHOT_SUBHALO_FILTER_VERSION_hsc_realistic.fits
          (e.g., 72_0_G_v2_hsc_realistic.fits)
        - Catalog format is compatible with Hyrax's FitsImageDataSet expectations.
        - The 'object_id' in the catalog is an integer composed of the snapshot (2 digits) followed by the subhalo (6-digit zero-padded).

    Example:
        # Save split images and keep the original FITS files
        download_and_split_hsc_images(
            split_output_dir='split_images',
            URL_LIST='urls.txt',
            BATCH_START=0,
            BATCH_SIZE=50,
            API_KEY='YOUR_API_KEY'
        )

        # Save split images, remove the parent file, and write a catalog
        download_and_split_hsc_images(
            split_output_dir='split_images',
            URL_LIST='urls.txt',
            BATCH_START=0,
            BATCH_SIZE=50,
            API_KEY='YOUR_API_KEY',
            remove_parent=True,
            catalog_path='split_images/catalog.fits'
        )

        # Download only the parent files, no splitting or catalog
        download_and_split_hsc_images(
            URL_LIST='urls.txt',
            BATCH_START=0,
            BATCH_SIZE=10,
            API_KEY='YOUR_API_KEY',
            parent_file_only=True
        )
    '''
    # determine directory for parent files
    parent_dir = parent_output_dir or split_output_dir
    os.makedirs(parent_dir, exist_ok=True)

    # ensure output dir exists
    if not parent_file_only:
        os.makedirs(split_output_dir, exist_ok=True)

    # load URLs and pick batch
    with open(URL_LIST) as f:
        urls = [u.strip() for u in f if u.strip()]
    if BATCH_START is None:
        BATCH_START = 0
    if BATCH_SIZE is None:
        batch = urls[BATCH_START:]
    else:
        batch = urls[BATCH_START : BATCH_START + BATCH_SIZE]

    if not batch:
        print(' ⚠️  no URLs to process with the specified batch parameters')
        return

    catalog_entries = [] if catalog_path else None

    # helper to pull snapshot, subhalo, version from URL
    def parse_url(u):
        parts = u.split('/')
        snapshot = parts[6]        # e.g. '72'
        subhalo = parts[8]        # e.g. '0'
        fn = parts[-1]       # e.g. 'skirt_images_hsc_realistic_v2.fits'
        v_match = re.search(r'(v\d+)', fn)
        version = v_match.group(1) if v_match else 'v?'
        return snapshot, subhalo, version

    # main loop
    for url in batch:
        snapshot, subhalo, version = parse_url(url)

        # download parent file
        fname_parent = f'{snapshot}_{subhalo}_{version}_parent.fits'
        parent_path = os.path.join(parent_dir, fname_parent)
        print(f'\nDownloading {fname_parent} into {parent_dir} …')
        r = requests.get(url, headers={'API-Key': API_KEY}, stream=True)
        r.raise_for_status()
        with open(parent_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        if not parent_file_only:
            # open and split
            with fits.open(parent_path, memmap=True) as hdul:
                for filt in ['G', 'R', 'I', 'Z', 'Y']:
                    target_ext = f'SUBARU_HSC.{filt}'
                    sci_hdu = next(
                        (h for h in hdul if h.header.get('EXTNAME','') == target_ext),
                        None
                    )
                    if sci_hdu is None:
                        print(f' ⚠️  no extension {target_ext} in {fname_parent}')
                        continue

                    new_hdu = fits.PrimaryHDU(data=sci_hdu.data, header=sci_hdu.header)
                    out_name = f'{snapshot}_{subhalo}_{filt}_{version}_hsc_realistic.fits'
                    out_path = os.path.join(split_output_dir, out_name)
                    new_hdu.writeto(out_path, overwrite=True)
                    print(f' ✅ wrote {out_name}')
                    if catalog_entries is not None:
                        # construct 8-digit object_id: snapshot (2 digits) + subhalo (6-digit zero-padded)
                        obj_id = int(snapshot) * 1000000 + int(subhalo)
                        catalog_entries.append({
                            'object_id': obj_id,
                            'filename': out_name,
                            'filter': filt
                        })

            # optionally remove parent file
            if remove_parent:
                try:
                    os.remove(parent_path)
                    print(f' 🗑 removed parent file {fname_parent}')
                except OSError as e:
                    print(f' ⚠️  could not remove {fname_parent}: {e}')

            # be gentle on the API server
            time.sleep(0.2)

    if catalog_entries is not None and not parent_file_only:
        table = Table(rows=catalog_entries, names=['object_id', 'filename', 'filter'])
        table.write(catalog_path, overwrite=True)
        print(f' 📄 wrote catalog with {len(catalog_entries)} entries to {catalog_path}')


def analyze_cutout_sizes(
    input_dir='split_images',
    output_plot=None,
    output_csv=None,
    bins=30,
    pattern='*.fits'
):
    '''
    Inspect FITS cutouts in a directory and summarize their size distribution.

    Args:
        input_dir (str, optional): Directory containing FITS cutouts. Defaults to 'split_images'.
        output_plot (str, optional): Path for the histogram image. Defaults to
            '<input_dir>/cutout_size_histogram.png'.
        output_csv (str, optional): Path for the CSV summary of exact size counts.
            Defaults to '<input_dir>/cutout_size_summary.csv'.
        bins (int, optional): Number of histogram bins to use. Defaults to 30.
        pattern (str, optional): Glob pattern used to select files. Defaults to '*.fits'.
    '''
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f'Input directory does not exist: {input_path}')
    if not input_path.is_dir():
        raise SystemExit(f'Input path is not a directory: {input_path}')

    files = sorted(path for path in input_path.glob(pattern) if path.is_file())
    if not files:
        raise SystemExit(f'No files matched {pattern!r} in {input_path}')

    measurements = []
    skipped_files = []

    for path in files:
        try:
            with fits.open(path, memmap=True) as hdul:
                header = hdul[0].header
                naxis = int(header.get('NAXIS', 0))
                if naxis < 2:
                    skipped_files.append((path.name, f'NAXIS={naxis}'))
                    continue

                width = int(header['NAXIS1'])
                height = int(header['NAXIS2'])
                measurements.append({
                    'filename': path.name,
                    'width': width,
                    'height': height,
                    'pixel_area': width * height
                })
        except Exception as exc:
            skipped_files.append((path.name, str(exc)))

    if not measurements:
        raise SystemExit(f'No 2D FITS cutouts were found in {input_path}')

    if bins <= 0:
        raise SystemExit('--bins must be a positive integer')

    if output_plot is None:
        output_plot_path = input_path / 'cutout_size_histogram.png'
    else:
        output_plot_path = Path(output_plot).expanduser()

    if output_csv is None:
        output_csv_path = input_path / 'cutout_size_summary.csv'
    else:
        output_csv_path = Path(output_csv).expanduser()

    if not output_plot_path.is_absolute():
        output_plot_path = Path.cwd() / output_plot_path
    if not output_csv_path.is_absolute():
        output_csv_path = Path.cwd() / output_csv_path

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    width_values = [row['width'] for row in measurements]
    height_values = [row['height'] for row in measurements]
    area_values = [row['pixel_area'] for row in measurements]
    size_counts = Counter((row['width'], row['height']) for row in measurements)

    with output_csv_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['width', 'height', 'pixel_area', 'count'])
        for (width, height), count in sorted(size_counts.items()):
            writer.writerow([width, height, width * height, count])

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            f'Generating the histogram requires matplotlib to be installed: {exc}'
        ) from exc

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    plot_specs = [
        ('Width distribution', 'Width (pixels)', width_values, '#4C72B0'),
        ('Height distribution', 'Height (pixels)', height_values, '#55A868'),
        ('Area distribution', 'Pixel area', area_values, '#C44E52'),
    ]

    for ax, (title, xlabel, values, color) in zip(axes, plot_specs):
        ax.hist(values, bins=min(bins, len(values)), color=color, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')

    fig.suptitle(f'Cutout size distribution ({len(measurements)} FITS files)')
    fig.savefig(output_plot_path, dpi=200)
    plt.close(fig)

    print(f'Analyzed {len(measurements)} FITS files from {input_path}')
    print(f'Histogram written to {output_plot_path}')
    print(f'Summary CSV written to {output_csv_path}')
    print(f'Width range: {min(width_values)} to {max(width_values)} pixels')
    print(f'Height range: {min(height_values)} to {max(height_values)} pixels')
    print(f'Unique size pairs: {len(size_counts)}')

    if skipped_files:
        print(f'Skipped {len(skipped_files)} file(s) that were not readable 2D FITS cutouts')

    print('Most common cutout sizes:')
    for (width, height), count in size_counts.most_common(10):
        print(f'  {width}x{height}: {count}')


def _resolve_api_key(explicit, command_name):
    """Return the API key provided or fallback to the ``TNG_API_KEY`` environment variable."""
    api_key = explicit or os.environ.get('TNG_API_KEY')
    if not api_key:
        raise SystemExit(
            f'{command_name} requires an API key. Provide --api-key or set the TNG_API_KEY environment variable.'
        )
    return api_key


def build_parser():
    """Construct and return the argument parser for the CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Utilities for fetching and processing IllustrisTNG HSC FITS images.'
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    fetch_parser = subparsers.add_parser(
        'fetch',
        help='Fetch the list of HSC FITS file URLs and write them to disk.'
    )
    fetch_parser.add_argument(
        '--api-key',
        default=None,
        help='IllustrisTNG API key. Defaults to the TNG_API_KEY environment variable.'
    )
    fetch_parser.add_argument(
        '--base-api-url',
        default='https://www.tng-project.org/',
        help='Base URL for the TNG API.'
    )
    fetch_parser.add_argument(
        '--ending',
        default='/api/TNG50-1/files/skirt_images_hsc/',
        help='API endpoint path providing the URL list.'
    )
    fetch_parser.add_argument(
        '--snapshot-filter',
        default='_realistic_v2_91',
        help='Substring used to filter snapshot URLs.'
    )
    fetch_parser.add_argument(
        '--output-path',
        default='all_file_urls.txt',
        help='File path to store the fetched URLs. Default: all_file_urls.txt'
    )

    split_parser = subparsers.add_parser(
        'split',
        help='Download HSC FITS files and optionally split them into individual filters.'
    )
    split_parser.add_argument(
        '--url-list',
        required=True,
        help='Path to the text file containing parent FITS URLs (one per line).'
    )
    split_parser.add_argument(
        '--batch-start',
        type=int,
        default=0,
        help='Index of the first URL to process. Default: 0'
    )
    split_parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Number of URLs to process. Processes all remaining if omitted.'
    )
    split_parser.add_argument(
        '--api-key',
        default=None,
        help='IllustrisTNG API key. Defaults to the TNG_API_KEY environment variable.'
    )
    split_parser.add_argument(
        '--split-output-dir',
        default='split_images',
        help='Directory to store split FITS files. Default: split_images'
    )
    split_parser.add_argument(
        '--remove-parent',
        action='store_true',
        help='Delete the downloaded parent FITS file after splitting.'
    )
    split_parser.add_argument(
        '--catalog-path',
        default=None,
        help='Path to write a Hyrax-compatible FITS catalog.'
    )
    split_parser.add_argument(
        '--parent-file-only',
        action='store_true',
        help='Only download the parent FITS files without splitting.'
    )
    split_parser.add_argument(
        '--parent-output-dir',
        default=None,
        help='Directory to store downloaded parent FITS files. Defaults to the split output directory.'
    )

    analyze_parser = subparsers.add_parser(
        'analyze-sizes',
        help='Analyze the FITS cutout size distribution in a directory.'
    )
    analyze_parser.add_argument(
        '--input-dir',
        default='split_images',
        help='Directory containing FITS cutouts. Default: split_images'
    )
    analyze_parser.add_argument(
        '--output-plot',
        default=None,
        help='Path for the histogram image. Defaults to <input_dir>/cutout_size_histogram.png'
    )
    analyze_parser.add_argument(
        '--output-csv',
        default=None,
        help='Path for the CSV summary. Defaults to <input_dir>/cutout_size_summary.csv'
    )
    analyze_parser.add_argument(
        '--bins',
        type=int,
        default=30,
        help='Number of histogram bins. Default: 30'
    )
    analyze_parser.add_argument(
        '--pattern',
        default='*.fits',
        help='Glob pattern used to select FITS files. Default: *.fits'
    )

    return parser


def main(argv=None):
    """CLI entry point that dispatches to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'fetch':
        api_key = _resolve_api_key(args.api_key, 'fetch')
        make_list_of_urls(
            API_KEY=api_key,
            BASE_API_URL=args.base_api_url,
            ENDING=args.ending,
            SNAPSHOT_FILTER=args.snapshot_filter,
            output_path=args.output_path
        )
    elif args.command == 'split':
        api_key = _resolve_api_key(args.api_key, 'split')
        download_and_split_hsc_images(
            split_output_dir=args.split_output_dir,
            URL_LIST=args.url_list,
            BATCH_START=args.batch_start,
            BATCH_SIZE=args.batch_size,
            API_KEY=api_key,
            remove_parent=args.remove_parent,
            catalog_path=args.catalog_path,
            parent_file_only=args.parent_file_only,
            parent_output_dir=args.parent_output_dir
        )
    elif args.command == 'analyze-sizes':
        analyze_cutout_sizes(
            input_dir=args.input_dir,
            output_plot=args.output_plot,
            output_csv=args.output_csv,
            bins=args.bins,
            pattern=args.pattern
        )
    else:
        parser.error('Unknown command provided.')


if __name__ == '__main__':
    main()
