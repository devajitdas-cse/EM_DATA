#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate any index CSV (validation/baseline/custom) for LNA-free EM bands.

Reads:  <data_dir>/<index_csv>
Writes: <data_dir>/analysis/<prefix>_summary.csv
        <data_dir>/analysis/<prefix>_plots/*.png

Run examples:
  python validate_any_index.py --data_dir "F:\EM_DATA" --index_csv "validation_trace_index.csv" --prefix "validation"
  python validate_any_index.py --data_dir "F:\EM_DATA" --index_csv "baseline_trace_index.csv"   --prefix "baseline_100MHz"

Index CSV columns required:
  filename, center_freq_hz, samp_rate, gain_db, condition, repeat
Optional: phase, duration_s
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


def read_cfile(path: Path) -> np.ndarray:
    x = np.fromfile(str(path), dtype=np.complex64)
    if x.size == 0:
        raise ValueError(f"Empty file: {path}")
    return x


def psd_db(x: np.ndarray, fs: float, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    f, P = welch(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
        detrend="constant", return_onesided=True, scaling="density", average="mean"
    )
    P = P.astype(float)
    # mask DC ±5 kHz
    P[f <= 5000.0] = np.nan
    eps = 1e-18
    return f, 10 * np.log10(P + eps)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = ~np.isnan(p) & ~np.isnan(q)
    p = p[mask]
    q = q[mask]
    p = p / (p.sum() + 1e-18)
    q = q / (q.sum() + 1e-18)
    m = 0.5 * (p + q)

    def kl(a, b):
        a = np.clip(a, 1e-18, None)
        b = np.clip(b, 1e-18, None)
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def band_power_series(x: np.ndarray, chunk: int = 4096) -> np.ndarray:
    m = len(x) // chunk
    x = x[: m * chunk]
    p = np.abs(x.reshape(m, chunk)) ** 2
    return p.mean(axis=1)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2 + 1e-18))
    return float((a.mean() - b.mean()) / (sp + 1e-18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--index_csv", required=True, help="CSV filename inside data_dir (e.g., baseline_trace_index.csv)")
    ap.add_argument("--prefix", required=True, help="Prefix for output files (e.g., baseline_100MHz)")
    ap.add_argument("--nperseg", type=int, default=4096)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    idx_path = data_dir / args.index_csv
    if not idx_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {idx_path}")

    df = pd.read_csv(idx_path)
    required = {"filename", "center_freq_hz", "samp_rate", "gain_db", "condition", "repeat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Index CSV missing columns: {missing}")

    df["condition"] = df["condition"].astype(str).str.lower().str.strip()

    out_dir = data_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / f"{args.prefix}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cf, g in df.groupby("center_freq_hz"):
        fs = float(g["samp_rate"].iloc[0])
        idle_files = g[g["condition"] == "idle"]["filename"].tolist()
        load_files = g[g["condition"] == "load"]["filename"].tolist()
        if len(idle_files) == 0 or len(load_files) == 0:
            continue

        idle_psds, load_psds = [], []
        idle_bp, load_bp = [], []
        f_ref = None

        for fn in idle_files:
            x = read_cfile(data_dir / Path(fn))
            f, db = psd_db(x, fs, nperseg=args.nperseg)
            if f_ref is None:
                f_ref = f
            idle_psds.append(db)
            idle_bp.append(band_power_series(x))

        for fn in load_files:
            x = read_cfile(data_dir / Path(fn))
            f, db = psd_db(x, fs, nperseg=args.nperseg)
            load_psds.append(db)
            load_bp.append(band_power_series(x))

        idle_mean = np.nanmean(np.vstack(idle_psds), axis=0)
        load_mean = np.nanmean(np.vstack(load_psds), axis=0)
        delta = load_mean - idle_mean

        mean_abs_delta = float(np.nanmean(np.abs(delta)))

        idle_lin = 10 ** (idle_mean / 10.0)
        load_lin = 10 ** (load_mean / 10.0)
        jsd = js_divergence(idle_lin, load_lin)

        idle_bp = np.concatenate(idle_bp)
        load_bp = np.concatenate(load_bp)
        d = cohens_d(load_bp, idle_bp)  # load - idle

        rows.append({
            "center_freq_hz": int(cf),
            "center_freq_mhz": float(cf) / 1e6,
            "mean_abs_delta_psd_db": mean_abs_delta,
            "js_divergence": jsd,
            "cohens_d_bandpower": d,
            "n_idle_files": len(idle_files),
            "n_load_files": len(load_files),
        })

        # Save plots
        plt.figure()
        plt.plot(f_ref / 1e3, idle_mean, label="Idle")
        plt.plot(f_ref / 1e3, load_mean, label="Load")
        plt.xlabel("Baseband Frequency (kHz)")
        plt.ylabel("PSD (dB/Hz)")
        plt.title(f"PSD Overlay at Center {float(cf)/1e6:.0f} MHz (masked DC ±5 kHz)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"psd_overlay_cf_{int(cf/1e6)}MHz.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(f_ref / 1e3, delta)
        plt.xlabel("Baseband Frequency (kHz)")
        plt.ylabel("ΔPSD (Load - Idle) [dB]")
        plt.title(f"ΔPSD at Center {float(cf)/1e6:.0f} MHz ({args.prefix})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"delta_psd_cf_{int(cf/1e6)}MHz.png", dpi=200)
        plt.close()

    summary = pd.DataFrame(rows).sort_values("mean_abs_delta_psd_db", ascending=False).reset_index(drop=True)
    out_csv = out_dir / f"{args.prefix}_summary.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Plots: {plot_dir}")


if __name__ == "__main__":
    main()
