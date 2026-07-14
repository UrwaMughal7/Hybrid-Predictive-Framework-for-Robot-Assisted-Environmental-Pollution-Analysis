"""
evaluation/reconstruction.py
============================
Hybrid forecast reconstruction and output saving:
  - save_all_predictions   : write per-model CSVs
  - save_all_metrics       : write per-metric CSVs + summary
  - write_system_report    : plain-text report
  - final_summary          : print table + heatmap
  - spatial_hotspot        : rank stations by mean PM2.5
  - run_cascade            : multi-sensor cascade simulation
"""

import ast
import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm     import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats

from config.settings import (
    SEED, LOOKBACK, TEST_FRAC,
    THR_SAFE, THR_MODERATE, THR_HAZARDOUS,
    SENSOR_CASCADE, ALL_DIRS,
    RESULTS_DIR, DIR_PRED, DIR_MET, DIR_FIG, DIR_ROBOT,
    STATION_COLORS, PAL, PLOT_STYLE,
    STATION_COORDS,
)
from utils.helpers import savefig, compute_metrics

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False

try:
    from tabulate import tabulate as _tabulate
except ImportError:
    def _tabulate(rows, headers=None, tablefmt=None):
        out = []
        if headers:
            out.append("  ".join(f"{h:<16}" for h in headers))
            out.append("-" * (17 * len(headers)))
        for r in rows:
            out.append("  ".join(f"{str(v):<16}" for v in r))
        return "\n".join(out)


# ── Output saving ──────────────────────────────────────────────────────────────
def save_all_predictions(arimax_results: dict, lstm_results: dict,
                         hybrid_results: dict) -> None:
    """Save prediction CSVs for all models → DIR_PRED."""
    print("\n  Saving prediction CSVs ...")
    os.makedirs(DIR_PRED, exist_ok=True)

    for sta, r in arimax_results.items():
        pd.DataFrame({
            "date":           r["test_index"],
            "actual":         r["actual"],
            "arimax_forecast":r["forecast"],
            "error":          r["actual"] - r["forecast"],
        }).set_index("date").to_csv(
            os.path.join(DIR_PRED, f"arimax_{sta.replace(' ','_')}.csv")
        )

    for sta, r in lstm_results.items():
        pd.DataFrame({
            "date":          r["test_index"],
            "actual":        r["actual"],
            "lstm_forecast": r["forecast"],
            "error":         r["actual"] - r["forecast"],
        }).set_index("date").to_csv(
            os.path.join(DIR_PRED, f"lstm_{sta.replace(' ','_')}.csv")
        )

    for sta, r in hybrid_results.items():
        pd.DataFrame({
            "date":             r["test_index"],
            "actual":           r["actual"],
            "hybrid_forecast":  r["final_fc"],
            "arimax_component": r["arimax_pred"],
            "lstm_correction":  r["lstm_correction"],
            "error":            r["actual"] - r["final_fc"],
        }).set_index("date").to_csv(
            os.path.join(DIR_PRED, f"hybrid_{sta.replace(' ','_')}.csv")
        )

    print(f"  ✓ Predictions → {DIR_PRED}")


def save_all_metrics(arimax_m: dict, lstm_m: dict, hybrid_m: dict) -> None:
    """Save one metrics CSV per model + a combined summary → DIR_MET."""
    os.makedirs(DIR_MET, exist_ok=True)
    model_map = {
        "arimax": arimax_m,
        "lstm":   lstm_m,
        "hybrid": hybrid_m,
    }
    summary = []
    for mname, mdict in model_map.items():
        if not mdict:
            continue
        df = pd.DataFrame(mdict).T
        df.to_csv(os.path.join(DIR_MET, f"{mname}_metrics.csv"))
        for mk in df.columns:
            summary.append({"model": mname, "metric": mk,
                             "mean":  df[mk].mean(), "std": df[mk].std()})
    pd.DataFrame(summary).to_csv(
        os.path.join(DIR_MET, "summary_all_models.csv"), index=False
    )
    print(f"  ✓ Metrics → {DIR_MET}")


def write_system_report(all_metrics: dict, comparison_df, ranked: list) -> None:
    """Write a plain-text summary → RESULTS_DIR/system_report.txt."""
    path = os.path.join(RESULTS_DIR, "system_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ARIMAX + LSTM + HYBRID ARIMA-LSTM — ROBOT-ASSISTED AIR QUALITY SYSTEM\n")
        f.write("Beijing Multi-Site PM2.5 Forecasting (Hourly)\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        f.write("HYBRID MODEL PERFORMANCE\n" + "-" * 40 + "\n")
        for sta, m in all_metrics.items():
            f.write(f"\n{sta}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.5f}\n")
        f.write("\n\nSTATION PATROL ORDER (highest PM2.5 first)\n" + "-" * 40 + "\n")
        for rk, (s, info) in enumerate(ranked, 1):
            f.write(f"  {rk}. {s:<25} mean={info['mean_pm25']:.2f} µg/m³\n")
        f.write("\n\nOUTPUT FOLDER STRUCTURE\n" + "-" * 40 + "\n")
        for d in ALL_DIRS:
            f.write(f"  {os.path.relpath(d, RESULTS_DIR) if d != RESULTS_DIR else '.'}\n")
    print(f"  ✓ System report → {path}")


def final_summary(all_metrics: dict) -> None:
    """Print summary table + save final metrics heatmap → DIR_FIG."""
    MK = ["MAE", "MSE", "RMSE", "R2"]
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  FINAL HYBRID MODEL SUMMARY                                      ║")
    print("╚" + "═" * 68 + "╝\n")
    rows = [[s] + [f"{all_metrics[s][k]:.4f}" for k in MK if k in all_metrics[s]]
            for s in all_metrics]
    print(_tabulate(rows, headers=["Station"] + MK, tablefmt="fancy_grid"))

    stas = list(all_metrics.keys())
    n = len(stas)
    mk_avail = [k for k in MK if k in next(iter(all_metrics.values()))]
    df_h = pd.DataFrame(all_metrics).T[mk_avail].astype(float)
    norm = df_h.copy()
    for c in mk_avail:
        mn, mx = df_h[c].min(), df_h[c].max()
        norm[c] = (df_h[c] - mn) / (mx - mn) if mx > mn else 0.5

    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * n + 1.5)))
    if _SNS:
        sns.heatmap(norm, annot=df_h.round(3), fmt="g", cmap="RdYlGn_r", ax=ax,
                    linewidths=0.5, annot_kws={"size": 8},
                    cbar_kws={"label": "Normalised (green=best)"})
    else:
        im = ax.imshow(norm.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(mk_avail)))
        ax.set_xticklabels(mk_avail, fontsize=10, fontweight="bold")
        ax.set_yticks(range(n)); ax.set_yticklabels(stas, fontsize=9)
        for i in range(n):
            for j, mk in enumerate(mk_avail):
                if mk in df_h.columns:
                    ax.text(j, i, f"{df_h.iloc[i][mk]:.3f}", ha="center",
                            va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="Normalised (green=best)")
    ax.set_title("Final Metrics Heatmap — Hybrid ARIMA-LSTM",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "final_metrics_heatmap.png", DIR_FIG)
    print(f"\n  ✓ Final heatmap → {DIR_FIG}")


# ── Spatial hotspot ────────────────────────────────────────────────────────────
def spatial_hotspot(results: dict, all_dec: dict) -> tuple:
    """
    Rank stations by mean predicted PM2.5.
    Saves robot_decisions/patrol_order.csv + figures/spatial_hotspot_map.png.
    Returns (ranked_list, sta_info_dict).
    """
    print("\n" + "═" * 65)
    print("  PART — SPATIAL HOTSPOT ANALYSIS")
    print("═" * 65)

    sta_info = {}
    for i, (sta, r) in enumerate(results.items()):
        coord = next(
            (v for k, v in STATION_COORDS.items()
             if k.lower() in sta.lower() or sta.lower() in k.lower()),
            (39.70 + i * 0.06, 116.10 + i * 0.08),
        )
        dec = all_dec[sta]
        sta_info[sta] = {
            "lat":       coord[0],
            "lon":       coord[1],
            "mean_pm25": float(np.mean(r["final_fc"])),
            "max_pm25":  float(np.max(r["final_fc"])),
            "pct_haz":   float((dec["category"] == "HAZARDOUS").mean() * 100),
        }

    ranked = sorted(sta_info.items(),
                    key=lambda x: x[1]["mean_pm25"], reverse=True)

    print(f"\n  {'Rank':<5}{'Station':<25}{'Mean PM2.5':>12}  Priority")
    print("  " + "─" * 55)
    for rk, (s, info) in enumerate(ranked, 1):
        pr = ("🔴 HIGH" if info["mean_pm25"] > THR_MODERATE else
              ("🟡 MED"  if info["mean_pm25"] > THR_SAFE    else "🟢 LOW"))
        print(f"  {rk:<5}{s:<25}{info['mean_pm25']:>12.2f}  {pr}")

    os.makedirs(DIR_ROBOT, exist_ok=True)
    pd.DataFrame([
        {"rank": rk, "station": s,
         "mean_pm25": info["mean_pm25"],
         "lat": info["lat"], "lon": info["lon"]}
        for rk, (s, info) in enumerate(ranked, 1)
    ]).to_csv(os.path.join(DIR_ROBOT, "patrol_order.csv"), index=False)

    norm_s = Normalize(vmin=0, vmax=THR_HAZARDOUS)
    cmap_s = plt.cm.RdYlGn_r
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 9))
        for sta, info in sta_info.items():
            pm = info["mean_pm25"]
            ax.scatter(info["lon"], info["lat"],
                       s=200 + pm * 2.5,
                       color=cmap_s(norm_s(pm)),
                       edgecolors="white", lw=1.5, zorder=5, alpha=0.85)
            ax.annotate(sta[:12], (info["lon"], info["lat"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        path = [(sta_info[s]["lon"], sta_info[s]["lat"]) for s, _ in ranked]
        for i in range(len(path) - 1):
            ax.annotate("", xy=path[i + 1], xytext=path[i],
                        arrowprops=dict(arrowstyle="-|>", color="navy",
                                        lw=2, alpha=0.6))

        sm = ScalarMappable(cmap=cmap_s, norm=norm_s)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Mean PM2.5 (µg/m³)")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(
            "Spatial Pollution Map & Patrol Route\n"
            "Bubble size ∝ mean PM2.5  |  Arrows = patrol order",
            fontweight="bold",
        )
        plt.tight_layout()
        savefig(fig, "spatial_hotspot_map.png", DIR_FIG)

    print(f"  ✓ Spatial map + patrol order → {DIR_FIG}")
    return ranked, sta_info


# ── Cascade Simulator ────────────────────────────────────────────────────────
class CascadeSimulator:
    """
    When PM2.5 rises, additional sensors activate progressively.
    Each activation refines the PM2.5 forecast and diagnoses the source.
    """
    _THR = {"SO2": 25, "NO2": 40, "CO": 1.0,
            "O3": 60, "PM10": 75, "WSPM": 3.0}

    _SOURCES = {
        (True,  True,  True,  False, True,  False): "Coal combustion + heavy traffic",
        (True,  False, True,  False, True,  False): "Industrial / power-plant emission",
        (False, True,  True,  False, False, False): "Dense urban traffic",
        (False, False, False, True,  False, True ): "Photochemical smog",
        (True,  True,  False, False, True,  False): "Mixed industry + traffic",
        (False, False, False, False, True,  True ): "Dust storm / sand transport",
        (False, False, False, False, True,  False): "Road dust / construction",
    }

    def _cat(self, pm: float) -> str:
        if pm < THR_SAFE:      return "SAFE"
        if pm < THR_MODERATE:  return "MODERATE"
        if pm < THR_HAZARDOUS: return "UNHEALTHY"
        return "HAZARDOUS"

    def _active(self, cat: str) -> list:
        sensors = ["PM2.5"]
        for c in ["SAFE", "MODERATE", "UNHEALTHY", "HAZARDOUS"]:
            sensors += SENSOR_CASCADE.get(c, [])
            if c == cat:
                break
        return list(dict.fromkeys(sensors))

    def _diagnose(self, row) -> str:
        flags = (
            row.get("SO2",  0) > self._THR["SO2"],
            row.get("NO2",  0) > self._THR["NO2"],
            row.get("CO",   0) > self._THR["CO"],
            row.get("O3",   0) > self._THR["O3"],
            row.get("PM10", 0) > self._THR["PM10"],
            row.get("WSPM", 9) < self._THR["WSPM"],
        )
        return self._SOURCES.get(flags, "Complex mixture — multiple sources")

    def _refine(self, pm: float, row, active: list) -> float:
        adj = 0.0
        if "PM10" in active and row.get("PM10", 0) > self._THR["PM10"]:
            adj += row.get("PM10", 0) * 0.08
        if "SO2"  in active and row.get("SO2",  0) > self._THR["SO2"]:
            adj += row.get("SO2",  0) * 0.12
        if "NO2"  in active and row.get("NO2",  0) > self._THR["NO2"]:
            adj += row.get("NO2",  0) * 0.10
        if "CO"   in active and row.get("CO",   0) > self._THR["CO"]:
            adj += row.get("CO",   0) * 5.0
        if "WSPM" in active and row.get("WSPM", 2) > 4.0:
            adj -= pm * 0.15
        return float(np.clip(pm + adj, 0, 800))

    def simulate_station(self, sta: str, results: dict,
                          sensor_df) -> pd.DataFrame:
        log = []
        r = results[sta]
        for date, pm in zip(r["test_index"], r["final_fc"]):
            srow   = (sensor_df.loc[date]
                      if date in sensor_df.index else pd.Series(dtype=float))
            cat    = self._cat(pm)
            active = self._active(cat)
            new    = [s for s in active if s != "PM2.5"]
            ref    = self._refine(pm, srow, active)
            ev = {
                "date":            str(date)[:19],
                "station":         sta,
                "pm25_predicted":  round(pm, 2),
                "pm25_refined":    round(ref, 2),
                "aqi_category":    cat,
                "sensors_active":  active,
                "newly_activated": new,
                "source":          self._diagnose(srow),
                "delta":           round(ref - pm, 2),
            }
            for s in new:
                v = srow.get(s, np.nan)
                ev[f"reading_{s}"] = (round(float(v), 3)
                                      if pd.notna(v) else None)
            log.append(ev)
        return pd.DataFrame(log)


def run_cascade(results: dict, sensor_data: dict) -> dict:
    """Run CascadeSimulator on all stations. Saves per-station + combined CSV."""
    print("\n" + "═" * 65)
    print("  PART — MULTI-SENSOR CASCADE SIMULATION")
    print("═" * 65)
    os.makedirs(DIR_ROBOT, exist_ok=True)

    sim  = CascadeSimulator()
    logs = {}
    for sta in results:
        if sta not in sensor_data:
            print(f"  Skipping {sta} — no sensor data")
            continue
        df_log = sim.simulate_station(sta, results, sensor_data[sta])
        df_log.to_csv(
            os.path.join(DIR_ROBOT, f"cascade_{sta.replace(' ','_')}.csv"),
            index=False,
        )
        logs[sta] = df_log
        n = len(df_log)
        for cat in ["SAFE", "MODERATE", "UNHEALTHY", "HAZARDOUS"]:
            c = (df_log["aqi_category"] == cat).sum()
            print(f"  {sta:<20} {cat:<12}: {c:>4}/{n} ({100*c/n:.1f}%)")

    if logs:
        pd.concat(logs.values()).to_csv(
            os.path.join(DIR_ROBOT, "cascade_all_stations.csv"), index=False
        )
    print(f"  ✓ Cascade logs → {DIR_ROBOT}")
    return logs


def plot_cascade(logs: dict, results: dict) -> None:
    """Four static cascade visualisation figures → DIR_FIG/cascade/."""
    out = os.path.join(DIR_FIG, "cascade")
    os.makedirs(out, exist_ok=True)
    stas = list(logs.keys())
    cols = STATION_COLORS[:len(stas)]
    nc = 3; nr = math.ceil(len(stas) / nc)
    sensors_ord = ["PM2.5", "PM10", "WSPM", "SO2", "NO2",
                   "TEMP", "CO", "O3", "PRES", "DEWP", "RAIN"]

    with plt.style.context(PLOT_STYLE):

        fig, axes = plt.subplots(nr, nc, figsize=(20, 5 * nr))
        axes = np.array(axes).flatten()
        for ax, sta, c in zip(axes, stas, cols):
            log = logs[sta].copy()
            log["date_dt"] = pd.to_datetime(log["date"])
            ax.plot(log["date_dt"], log["pm25_predicted"],
                    color=PAL["muted"], lw=1.3, ls=":", label="Original", alpha=0.7)
            ax.plot(log["date_dt"], log["pm25_refined"],
                    color=c, lw=2.2, ls="--", label="Refined")
            ax.axhspan(0, THR_SAFE,           color="#16A34A", alpha=0.06)
            ax.axhspan(THR_SAFE, THR_MODERATE, color="#D97706", alpha=0.06)
            ax.axhspan(THR_MODERATE, THR_HAZARDOUS, color="#DC2626", alpha=0.06)
            events = log[log["newly_activated"].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else (
                    len(ast.literal_eval(str(x))) > 0))]
            ax.scatter(events["date_dt"], events["pm25_refined"],
                       s=30, color=PAL["danger"], zorder=5, alpha=0.7)
            ax.set_title(sta, fontsize=8, fontweight="bold")
            ax.legend(fontsize=6)
        for ax in axes[len(stas):]: ax.set_visible(False)
        fig.suptitle(
            "Cascade: Original vs Refined PM2.5 Forecast\n"
            "(red dots = new sensor activations)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        savefig(fig, "cascade_forecast_refined.png", out)

        fig, axes = plt.subplots(nr, nc, figsize=(18, 5 * nr))
        axes = np.array(axes).flatten()
        for ax, sta in zip(axes, stas):
            log = logs[sta]
            ev  = log[log["newly_activated"].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else (
                    len(ast.literal_eval(str(x))) > 0))]
            if len(ev) == 0:
                ax.text(0.5, 0.5, "No cascade events",
                        ha="center", va="center", fontsize=10)
                ax.set_title(sta, fontweight="bold")
                continue
            sc = ev["source"].value_counts()
            ax.pie(sc.values,
                   labels=[s[:30] for s in sc.index],
                   autopct="%1.0f%%", startangle=90,
                   colors=STATION_COLORS[:len(sc)],
                   wedgeprops={"edgecolor": "white", "linewidth": 1.2})
            ax.set_title(f"{sta}\nSource Diagnosis", fontsize=8, fontweight="bold")
        for ax in axes[len(stas):]: ax.set_visible(False)
        fig.suptitle("Pollution Source Diagnosis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        savefig(fig, "cascade_source_pies.png", out)

        fig, axes = plt.subplots(nr, nc, figsize=(18, 4 * nr))
        axes = np.array(axes).flatten()
        for ax, sta, c in zip(axes, stas, cols):
            d = logs[sta]["delta"].dropna()
            mu, sg = d.mean(), d.std()
            ax.hist(d, bins=40, color=c, edgecolor="white",
                    lw=0.4, density=True, alpha=0.8)
            x = np.linspace(mu - 4 * sg, mu + 4 * sg, 200)
            ax.plot(x, stats.norm.pdf(x, mu, sg), "r-", lw=2,
                    label=f"μ={mu:.1f} σ={sg:.1f}")
            ax.axvline(0, color="black", lw=1.5, ls="--")
            ax.set_title(sta, fontweight="bold")
            ax.legend(fontsize=7)
        for ax in axes[len(stas):]: ax.set_visible(False)
        fig.suptitle(
            "Cascade Refinement Delta (positive = PM2.5 raised by sensors)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        savefig(fig, "cascade_delta_dist.png", out)

        frac = {
            sta: {
                s: logs[sta]["sensors_active"].apply(
                    lambda x: s in (x if isinstance(x, list) else ast.literal_eval(str(x)))
                ).mean()
                for s in sensors_ord
            }
            for sta in stas
        }
        df_f = pd.DataFrame(frac).T
        fig, ax = plt.subplots(
            figsize=(max(12, len(sensors_ord) * 1.1),
                     max(4, len(stas) * 0.6 + 1.5))
        )
        if _SNS:
            sns.heatmap(df_f, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                        linewidths=0.5, annot_kws={"size": 8},
                        cbar_kws={"label": "Fraction of days active"})
        else:
            im = ax.imshow(df_f.values, cmap="Blues",
                           aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(sensors_ord)))
            ax.set_xticklabels(sensors_ord, fontsize=9)
            ax.set_yticks(range(len(stas)))
            ax.set_yticklabels(stas, fontsize=8)
            plt.colorbar(im, ax=ax, label="Fraction of days active")
        ax.set_title("Fraction of Hours Each Sensor Was Active",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig(fig, "cascade_sensor_fraction.png", out)

    print(f"  ✓ Cascade figures → {out}")
