#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-2 Validation Analysis (Top-3 bands) — LNA-free EM monitoring
Reads:  <data_dir>/validation_trace_index.csv   (downloaded from ChatGPT and placed there)
Writes: <data_dir>/analysis/validation_summary.csv
        <data_dir>/analysis/validation_plots/*.png

Outputs:
- Per-center-frequency separability metrics between Idle and Load:
  * mean_abs_delta_db (ΔPSD magnitude)
  * js_divergence (Jensen–Shannon divergence between normalized PSDs)
  * cohens_d (effect size on band power time-series, chunked)
- Publication-ready plots:
  * overlay PSD (idle vs load) in dB
  * ΔPSD curve

Run:
  cd /d F:\EM_DATA
  python validate_lna_free_bands.py --data_dir "F:\EM_DATA"
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
    f, P = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg//2,
                detrend="constant", return_onesided=True, scaling="density", average="mean")
    # mask DC ±5 kHz
    P = P.astype(float)
    P[f <= 5000.0] = np.nan
    eps = 1e-18
    return f, 10*np.log10(P + eps)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # p,q are nonnegative and sum to 1 (ignore nans)
    mask = ~np.isnan(p) & ~np.isnan(q)
    p = p[mask]
    q = q[mask]
    p = p / (p.sum() + 1e-18)
    q = q / (q.sum() + 1e-18)
    m = 0.5*(p+q)
    def kl(a,b):
        a = np.clip(a, 1e-18, None)
        b = np.clip(b, 1e-18, None)
        return float(np.sum(a*np.log2(a/b)))
    return 0.5*kl(p,m) + 0.5*kl(q,m)


def band_power_series(x: np.ndarray, chunk: int = 4096) -> np.ndarray:
    # chunked mean power in time: |x|^2 averaged per chunk
    m = len(x)//chunk
    x = x[:m*chunk]
    p = np.abs(x.reshape(m, chunk))**2
    return p.mean(axis=1)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float); b = b.astype(float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = a.var(ddof=1); vb = b.var(ddof=1)
    sp = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2 + 1e-18))
    return float((a.mean() - b.mean()) / (sp + 1e-18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--nperseg", type=int, default=4096)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    idx = data_dir / "validation_trace_index.csv"
    if not idx.exists():
        raise FileNotFoundError(f"Put validation_trace_index.csv in {data_dir}")

    df = pd.read_csv(idx)
    out_plot = data_dir / "analysis" / "validation_plots"
    out_plot.mkdir(parents=True, exist_ok=True)

    rows = []
    for cf, g in df.groupby("center_freq_hz"):
        fs = float(g["samp_rate"].iloc[0])

        idle_files = g[g["condition"].str.lower()=="idle"]["filename"].tolist()
        load_files = g[g["condition"].str.lower()=="load"]["filename"].tolist()

        # PSD averages
        idle_psds = []
        load_psds = []
        f_ref = None

        # time-series for Cohen's d
        idle_bp = []
        load_bp = []

        for fn in idle_files:
            x = read_cfile(data_dir / Path(fn))
            f, db = psd_db(x, fs, nperseg=args.nperseg)
            if f_ref is None: f_ref = f
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

        # JS divergence on linear PSDs (convert back from dB carefully)
        # Use exp10 to get proportional to PSD; nans remain
        idle_lin = 10**(idle_mean/10.0)
        load_lin = 10**(load_mean/10.0)
        jsd = js_divergence(idle_lin, load_lin)

        # Cohen's d on band power time-series
        idle_bp = np.concatenate(idle_bp)
        load_bp = np.concatenate(load_bp)
        d = cohens_d(load_bp, idle_bp)  # load - idle

        rows.append({
            "center_freq_hz": int(cf),
            "center_freq_mhz": float(cf)/1e6,
            "mean_abs_delta_psd_db": mean_abs_delta,
            "js_divergence": jsd,
            "cohens_d_bandpower": d,
            "n_idle_files": len(idle_files),
            "n_load_files": len(load_files),
        })

        # Plot overlay PSD
        plt.figure()
        plt.plot(f_ref/1e3, idle_mean, label="Idle")
        plt.plot(f_ref/1e3, load_mean, label="Load")
        plt.xlabel("Baseband Frequency (kHz)")
        plt.ylabel("PSD (dB/Hz)")
        plt.title(f"PSD Overlay at Center {float(cf)/1e6:.0f} MHz (masked DC ±5 kHz)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_plot / f"psd_overlay_cf_{int(cf/1e6)}MHz.png", dpi=200)
        plt.close()

        # Plot delta
        plt.figure()
        plt.plot(f_ref/1e3, delta)
        plt.xlabel("Baseband Frequency (kHz)")
        plt.ylabel("ΔPSD (Load - Idle) [dB]")
        plt.title(f"ΔPSD at Center {float(cf)/1e6:.0f} MHz (validation)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_plot / f"delta_psd_validation_cf_{int(cf/1e6)}MHz.png", dpi=200)
        plt.close()

    summary = pd.DataFrame(rows).sort_values("mean_abs_delta_psd_db", ascending=False)
    out_csv = data_dir / "analysis" / "validation_summary.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== Validation summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Plots: {out_plot}")


if __name__ == "__main__":
    main()
