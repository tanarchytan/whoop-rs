#!/usr/bin/env python3
"""Convert AAUWSS aligned-sleep ECG pickles to plain-text fixtures the Rust tests read.

The pickles are pandas DataFrames (one 30 s epoch per row, 6000 ECG columns at 200 Hz plus
labels/epoch/time), so pandas + numpy are required to unpickle them; the OUTPUT is stdlib-parseable
text and is what gets committed. Source pickles are 70-90 MB each and stay outside the repo.

Selection rule, fixed so no epoch is cherry-picked for quality: take the middle row of each subject's
recording (index len//2); if it contains a non-finite sample, walk forward to the first row that does
not. The sample rate is measured from the row's own timestamp column, never assumed.

Usage:
  python tools/aauwss_ecg_to_fixture.py [SRC_DIR] [OUT_DIR]
"""
from pathlib import Path as _Path
_WHOOP = str(_Path(__file__).resolve().parents[2])  # whoop/ project root

import os
import pickle
import sys

import numpy as np

DEFAULT_SRC = _WHOOP + r"/whoop-data/datasets/AAUWSS/extracted/aligned_sleep_data_set/ecg"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "physio-algo", "tests", "fixtures", "aauwss_ecg")

# Stored integer = round(raw / SCALE). The published values are arbitrary EDF ECG-channel units with a
# per-subject SD of 200-450, so 0.1 units is ~0.03% of the SD: quantisation is far below any real detail.
SCALE = 0.1
N_ECG_COLS = 6000

# CC-BY-4.0 requires the attribution to travel with every redistributed copy, and these fixtures are
# committed, so the converter writes it into each file rather than leaving it to a README.
ATTRIBUTION = (
    "# source AAUWSS (Aalborg University Wearable Sleep Study) v1.1, Zenodo 10.5281/zenodo.16919071\n"
    "# licence CC-BY-4.0 - Djanian S, Nielsen T D, Nielsen S H, Bruun A (2025)\n"
)


def measured_fs_hz(times):
    """Sample rate from the row's own timestamps (median spacing), rounded to 1e-6 Hz."""
    t = np.asarray(times, dtype="datetime64[ns]").astype("int64")
    if t.size < 2:
        raise ValueError("need >= 2 timestamps to measure fs")
    step_ns = float(np.median(np.diff(t)))
    if step_ns <= 0:
        raise ValueError("non-positive timestamp step")
    return round(1e9 / step_ns, 6)


def pick_row(df):
    """Middle row, or the first later row with no non-finite sample."""
    n = df.shape[0]
    for i in range(n // 2, n):
        sig = df.iloc[i].values[:N_ECG_COLS].astype(float)
        if np.all(np.isfinite(sig)):
            return i, sig
    raise ValueError("no finite epoch at or after the midpoint")


def convert(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(src_dir) if f.endswith("_ecg.pkl"))
    if not names:
        raise SystemExit("no *_ecg.pkl under " + src_dir)
    for name in names:
        subject = name.split("_")[1]
        with open(os.path.join(src_dir, name), "rb") as fh:
            df = pickle.load(fh)
        row_idx, sig = pick_row(df)
        row = df.iloc[row_idx]
        fs = measured_fs_hz(row.values[N_ECG_COLS + 2])
        label = str(row.values[N_ECG_COLS])
        epoch = row.values[N_ECG_COLS + 1]
        counts = np.rint(sig / SCALE).astype(np.int64)
        out = os.path.join(out_dir, "subject_%s.txt" % subject)
        with open(out, "w", newline="\n") as fh:
            fh.write("# aauwss ecg epoch, one sample per line\n")
            fh.write(ATTRIBUTION)
            fh.write("# subject %s\n" % subject)
            fh.write("# fs_hz %s\n" % fs)
            fh.write("# samples %d\n" % counts.size)
            fh.write("# scale %s  (value = count * scale, arbitrary published units, NOT mV)\n" % SCALE)
            fh.write("# source_row %d of %d\n" % (row_idx, df.shape[0]))
            fh.write("# stage_label %s\n" % label)
            fh.write("# epoch %s\n" % epoch)
            for c in counts:
                fh.write("%d\n" % c)
        print("%s -> %s  fs=%s n=%d label=%s" % (name, os.path.basename(out), fs, counts.size, label))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    convert(src, os.path.normpath(out))
