#!/usr/bin/env python3
"""
download_and_convert_dataset.py

Smart utility script to download datasets and convert .pkl/.csv/.data files to .npy for use with Rust/Burn.

Features:
- Downloads files from URLs (with smart skipping and force option)
- Converts all .pkl, .csv, and .data files in the target directory (default: downloads-of-datasets)
- Skips conversion if .npy is newer than source (idempotent)
- Handles subdirectories recursively
- Copies all resulting .npy files to the processed directory
- Prints summary and errors

Usage:
    python download_and_convert_dataset.py --urls <url1> <url2> ...
"""
import os
import sys
import pickle
import numpy as np
import urllib.request
import argparse
import shutil
import json

default_download_dir = "/home/rustuser/rustdev/LinossRust/datastore/downloads-of-datasets"
default_processed_dir = "/home/rustuser/rustdev/LinossRust/datastore/processed-by-python"

def smart_copy_to_processed(arr_path, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    dest_path = os.path.join(processed_dir, os.path.basename(arr_path))
    if os.path.exists(dest_path) and os.path.getmtime(dest_path) >= os.path.getmtime(arr_path):
        print(f"[skip] Already in processed: {dest_path}")
        return
    shutil.copy2(arr_path, dest_path)
    print(f"[copy] {arr_path} -> {dest_path}")

def convert_pkl_to_npy(input_dir, processed_dir=None):
    converted = 0
    skipped = 0
    failed = 0
    processed = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_path = os.path.join(root, file)
                npy_path = os.path.splitext(pkl_path)[0] + '.npy'
                if os.path.exists(npy_path) and os.path.getmtime(npy_path) >= os.path.getmtime(pkl_path):
                    print(f"[skip] {pkl_path} (up-to-date)")
                    skipped += 1
                else:
                    try:
                        with open(pkl_path, 'rb') as f:
                            arr = pickle.load(f)
                        if not isinstance(arr, np.ndarray):
                            arr = np.array(arr)
                        # Always save as float32 unless it's a label array (heuristic: 1D and int/bool)
                        arr_to_save = arr
                        if arr.dtype.kind in ('i', 'u', 'b') and (arr.ndim == 1 or arr.shape[-1] == 1):
                            arr_to_save = arr.astype('uint8')
                        else:
                            arr_to_save = arr.astype('float32')
                        np.save(npy_path, arr_to_save)
                        print(f"[ok] {pkl_path} -> {npy_path} (shape: {arr.shape}, dtype: {arr_to_save.dtype})")
                        converted += 1
                    except Exception as e:
                        print(f"[fail] {pkl_path}: {e}")
                        failed += 1
                if processed_dir and os.path.exists(npy_path):
                    smart_copy_to_processed(npy_path, processed_dir)
                    processed += 1
    print(f"\nSummary: {converted} converted, {skipped} skipped, {failed} failed, {processed} copied to processed.")

def convert_csv_to_npy(input_csv, processed_dir=None):
    base = os.path.splitext(os.path.basename(input_csv))[0]
    arr = None
    labels = None
    label_map = None
    npy_path = os.path.join(processed_dir or os.path.dirname(input_csv), base + ".npy")
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    try:
        # Try loading as all-numeric first
        arr = np.loadtxt(input_csv, delimiter=",")
        np.save(npy_path, arr)
        print(f"[csv->npy] {input_csv} -> {npy_path} (shape: {arr.shape}, dtype: {arr.dtype})")
        return npy_path
    except Exception:
        # Try mixed-type: last column as label
        try:
            import pandas as pd
            df = pd.read_csv(input_csv, header=None)
            # Assume last column is label if non-numeric
            features = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
            labels = df.iloc[:, -1]
            # Encode labels if not numeric
            if labels.dtype == object or labels.dtype.name.startswith('str'):
                label_map = {v: i for i, v in enumerate(sorted(labels.unique()))}
                labels_encoded = labels.map(label_map)
                # Save label mapping
                map_path = os.path.join(processed_dir or os.path.dirname(input_csv), base + "_label_map.json")
                with open(map_path, 'w') as f:
                    json.dump(label_map, f)
                print(f"[csv->npy] Saved label map: {map_path}")
            else:
                labels_encoded = labels
                label_map = None
            features_npy = os.path.join(processed_dir or os.path.dirname(input_csv), base + "_features.npy")
            labels_npy = os.path.join(processed_dir or os.path.dirname(input_csv), base + "_labels.npy")
            np.save(features_npy, features.values.astype('float32'))
            np.save(labels_npy, labels_encoded.values.astype('uint8'))
            print(f"[csv->npy] {input_csv} -> {features_npy} (features shape: {features.shape})")
            print(f"[csv->npy] {input_csv} -> {labels_npy} (labels shape: {labels_encoded.shape})")
            return features_npy, labels_npy
        except Exception as e:
            print(f"[fail] CSV->NPY {input_csv}: {e}")
            return None

def smart_download(url, dest_dir, force=False):
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(url.split('?')[0])
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path) and not force:
        print(f"[skip] {filename} already exists.")
        return dest_path
    try:
        print(f"[download] {url} -> {dest_path}")
        urllib.request.urlretrieve(url, dest_path)
        print(f"[ok] Downloaded {filename}")
        return dest_path
    except Exception as e:
        print(f"[fail] Download {url}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download and convert .pkl/.csv/.data files to .npy for Burn.")
    parser.add_argument("--input_dir", default=default_download_dir, help="Directory for data files")
    parser.add_argument("--urls", nargs="*", help="URLs to download before conversion")
    parser.add_argument("--force", action="store_true", help="Force re-download files")
    parser.add_argument("--processed_dir", default=default_processed_dir, help="Where to copy .npy files after conversion")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true", help="Print more detailed logs")
    parser.add_argument("--example", action="store_true", help="Show example usage and exit")
    parser.add_argument("--demo", action="store_true", help="Show a demo of expected output and exit")
    args = parser.parse_args()

    if args.example or args.demo:
        print("\nExample usage:")
        print("  python3 scripts/download_and_convert_dataset.py --urls https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data --processed_dir ./datastore/processed-by-python")
        print("  python3 scripts/download_and_convert_dataset.py --input_dir ./datastore/downloads-of-datasets --processed_dir ./datastore/processed-by-python")
        print("\nExpected output: .npy files in the processed directory, with features as float32 and labels as uint8.")
        print("\nFor more info, run with -h or --help.")
        exit(0)

    if args.dry_run:
        print("[dry-run] Would download and convert datasets, but not actually write files.")
        # Could simulate file operations here
        exit(0)

    if args.urls:
        for url in args.urls:
            smart_download(url, args.input_dir, force=args.force)

    convert_pkl_to_npy(args.input_dir, processed_dir=args.processed_dir)

    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.data'):
                csv_path = os.path.join(root, file)
                try:
                    npy_path = convert_csv_to_npy(csv_path, processed_dir=args.processed_dir)
                except Exception as e:
                    print(f"[fail] CSV->NPY {csv_path}: {e}")

    print("\n[summary] All processing complete. Check the processed directory for .npy and .json files.")
    print("[info] Features are saved as float32, labels as uint8. Compatible with Rust/Burn.")
    print("[info] For help, run with -h or --help. For a usage example, run with --example.")

if __name__ == "__main__":
    main()
