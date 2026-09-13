#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNA-Free Adaptive Frequency Scouting (pilot) — Analysis script
Reads:  <data_dir>/trace_index.csv
Expects: complex64 .cfile traces located relative to <data_dir> (e.g., idle\*.cfile, load\*.cfile)
Writes:  <data_dir>/analysis/band_ranking.csv
         <data_dir>/analysis/plots/*.png

Run (Windows Radioconda Prompt):
  cd /d F:\EM_DATA
  python analyze_lna_free_scouting.py --data_dir "F:\EM_DATA"

Notes:
- Assumes each .cfile is complex64 interleaved IQ (float32 I, float32 Q) => np.complex64.
- Uses Welch PSD. If SciPy is missing, install: conda install -c conda-forge scipy numpy matplotlib pandas
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import welch
except Exception as e:
    welch = None


def read_cfile(path: Path) -> np.ndarray:
    """Read complex64 IQ from .cfile."""
    data = np.fromfile(str(path), dtype=np.complex64)
    if data.size == 0:
        raise ValueError(f"Empty file: {path}")
    return data


def compute_psd(x: np.ndarray, fs: float, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (f, Pxx) where f is frequency in Hz (baseband, 0..fs/2),
    and Pxx is power spectral density.
    """
    if welch is None:
        # Fallback: simple averaged periodogram using FFT chunks
        n = nperseg
        m = len(x) // n
        if m < 2:
            raise ValueError("Trace too short for FFT fallback; install SciPy for Welch.")
        x = x[: m * n]
        X = np.fft.rfft(x.reshape(m, n), axis=1)
        P = (np.abs(X) ** 2).mean(axis=0) / (fs * n)
        f = np.fft.rfftfreq(n, 1 / fs)
        return f, P

    f, Pxx = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )
    return f, Pxx


def mask_dc(f: np.ndarray, P: np.ndarray, dc_hz: float = 5000.0) -> np.ndarray:
    """Zero/ignore bins close to DC (center spur region)."""
    P2 = P.copy()
    P2[f <= dc_hz] = np.nan
    return P2


def summarize_band(df: pd.DataFrame, data_dir: Path, nperseg: int = 4096) -> pd.DataFrame:
    """
    For each center frequency:
      - compute PSD for each repeat
      - compute mean PSD per condition
      - compute delta PSD (load - idle) in dB
      - score = mean(|delta_dB|) over band (excluding DC) - stability_penalty
    """
    results = []
    plots_dir = data_dir / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Group by center frequency
    for cf_hz, g_cf in df.groupby("center_freq_hz"):
        fs = float(g_cf["samp_rate"].iloc[0])

        # Collect PSDs per condition
        psd = {"idle": [], "load": []}
        f_ref = None

        for _, row in g_cf.iterrows():
            rel_path = Path(row["filename"])
            fpath = data_dir / rel_path
            x = read_cfile(fpath)
            f, P = compute_psd(x, fs=fs, nperseg=nperseg)
            P = mask_dc(f, P, dc_hz=5000.0)  # mask first 5 kHz near DC

            if f_ref is None:
                f_ref = f
            else:
                # sanity check
                if len(f_ref) != len(f):
                    raise RuntimeError("PSD frequency grid mismatch across files.")

            psd[row["condition"]].append(P)

        # Ensure we have 3 repeats each
        if len(psd["idle"]) == 0 or len(psd["load"]) == 0:
            continue

        idle_stack = np.vstack(psd["idle"])
        load_stack = np.vstack(psd["load"])

        idle_mean = np.nanmean(idle_stack, axis=0)
        load_mean = np.nanmean(load_stack, axis=0)

        # Convert to dB (add eps to avoid log(0))
        eps = 1e-18
        idle_db = 10 * np.log10(idle_mean + eps)
        load_db = 10 * np.log10(load_mean + eps)
        delta_db = load_db - idle_db  # load - idle

        # Separability score: average absolute delta over the band
        sep = np.nanmean(np.abs(delta_db))

        # Stability penalty: how much delta varies across repeats
        # compute per-repeat delta, then std
        per_rep_delta = []
        for i in range(min(idle_stack.shape[0], load_stack.shape[0])):
            di = 10 * np.log10(load_stack[i] + eps) - 10 * np.log10(idle_stack[i] + eps)
            per_rep_delta.append(di)
        per_rep_delta = np.vstack(per_rep_delta)
        stab = np.nanmean(np.nanstd(per_rep_delta, axis=0))  # lower is better

        score = sep - 0.5 * stab  # weight penalty

        results.append({
            "center_freq_hz": int(cf_hz),
            "center_freq_mhz": float(cf_hz) / 1e6,
            "fs_hz": fs,
            "n_idle": int(idle_stack.shape[0]),
            "n_load": int(load_stack.shape[0]),
            "sep_mean_abs_delta_db": float(sep),
            "stability_mean_std_delta_db": float(stab),
            "score": float(score),
        })

        # Plot delta PSD for this center frequency
        plt.figure()
        plt.plot((f_ref / 1e3), delta_db)  # kHz offset
        plt.xlabel("Baseband Frequency (kHz)")
        plt.ylabel("ΔPSD (Load - Idle) [dB]")
        plt.title(f"ΔPSD at Center {float(cf_hz)/1e6:.0f} MHz (masked DC ±5 kHz)")
        plt.grid(True)
        out_png = plots_dir / f"delta_psd_cf_{int(cf_hz/1e6)}MHz.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    out_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to EM_DATA folder (contains trace_index.csv, idle/, load/, analysis/)")
    ap.add_argument("--nperseg", type=int, default=4096, help="Welch nperseg / FFT size")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    idx_path = data_dir / "trace_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(f"trace_index.csv not found at: {idx_path}")

    out_dir = data_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(idx_path)
    required_cols = {"filename", "center_freq_hz", "samp_rate", "gain_db", "condition", "repeat"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"trace_index.csv missing columns: {missing}")

    # Normalize condition values
    df["condition"] = df["condition"].astype(str).str.lower().str.strip()

    ranking = summarize_band(df, data_dir=data_dir, nperseg=args.nperseg)

    out_csv = out_dir / "band_ranking.csv"
    ranking.to_csv(out_csv, index=False)

    # Print top results
    print("\n=== Band ranking (top) ===")
    print(ranking.head(10).to_string(index=False))
    print(f"\nSaved ranking to: {out_csv}")
    print(f"Saved plots to: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
