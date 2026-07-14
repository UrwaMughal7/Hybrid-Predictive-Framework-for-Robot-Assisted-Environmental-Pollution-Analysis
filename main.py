"""
main.py
=======
Entry point for the ARIMAX + LSTM + Hybrid ARIMA-LSTM PM2.5 Prediction Pipeline.

Models from pm2.5.py integrated into modular pipeline:
  1  Data loading (hourly, 27 features incl. wind direction)
  2  EDA
  3  ARIMAX — ARIMA(1,0,1) + exogenous features
  4  LSTM — 2-layer LSTM on all features
  5  Hybrid ARIMA-LSTM — ARIMAX + LSTM residual correction
  6  Model comparison
  7  Robot decision module
  8  Spatial hotspot analysis
  9  Multi-sensor cascade simulation
  10 Animations and video export
  11 Final summary & save outputs
"""

import os, random, warnings
warnings.filterwarnings("ignore")

os.environ["PYTHONHASHSEED"]         = "42"
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import (
    SEED, RESULTS_DIR, ALL_DIRS,
    LOOKBACK, NUMERIC_SENSORS,
    PLOT_STYLE, PAL, DIR_FIG,
)
from utils.helpers        import set_all_seeds, ensure_dirs, savefig
from data.loader          import load_hourly_data, load_sensor_data
from models               import fit_arimax, fit_lstm, fit_hybrid
from evaluation           import (
    spatial_hotspot, CascadeSimulator, run_cascade, plot_cascade,
    save_all_predictions, save_all_metrics,
    write_system_report, final_summary,
)
from robot                import (
    run_robot_decisions, run_realtime_simulation,
    compute_robotics_metrics, save_robotics_outputs,
)
from visualization        import (
    plot_eda,
    plot_cross_station_dashboard,
    plot_model_comparison,
    plot_robot_dashboards, plot_architecture,
    animate_cascade, animate_robot_full_dashboard,
    generate_all_animations,
)


def main():
    print("╔" + "═" * 78 + "╗")
    print("║  ARIMAX + LSTM + HYBRID ARIMA-LSTM  |  Beijing Air Quality  ║")
    print("║  Robot-Assisted Environmental Monitoring System  [Hourly]    ║")
    print("╚" + "═" * 78 + "╝")

    set_all_seeds(SEED)
    ensure_dirs()

    # ── PART 1: Data Loading (hourly, 27 features) ──────────────────────────
    hourly_data = load_hourly_data()
    sensor_data = load_sensor_data(features=NUMERIC_SENSORS)

    # Daily PM2.5 wide DataFrame for EDA
    df_daily = pd.DataFrame({
        sta: hourly_data[sta]["PM2.5"].resample("D").mean()
        for sta in hourly_data
    })
    df_daily = df_daily.replace(0, np.nan).ffill(limit=3).interpolate(method="time").clip(lower=0)

    # ── PART 2: EDA ─────────────────────────────────────────────────────────
    plot_eda(df_daily)

    # ── PART 3: ARIMAX ──────────────────────────────────────────────────────
    arimax_results = fit_arimax(hourly_data)
    arimax_metrics = {s: arimax_results[s]["metrics"] for s in arimax_results}

    # ── PART 4: LSTM ────────────────────────────────────────────────────────
    lstm_results = fit_lstm(hourly_data)
    lstm_metrics = {s: lstm_results[s]["metrics"] for s in lstm_results}

    # ── PART 5: Hybrid ARIMA-LSTM ───────────────────────────────────────────
    hybrid_results = fit_hybrid(hourly_data, arimax_results)
    hybrid_metrics = {s: hybrid_results[s]["metrics"] for s in hybrid_results}

    # ── PART 6: Model Comparison ────────────────────────────────────────────
    plot_model_comparison(arimax_metrics, lstm_metrics, hybrid_metrics)
    plot_cross_station_dashboard(hybrid_metrics, tag="hybrid")

    # ── PART 7: Robot Decision Module ───────────────────────────────────────
    all_dec = run_robot_decisions(hybrid_results)
    plot_robot_dashboards(hybrid_results, all_dec)

    # ── PART 8: Real-Time Simulation ────────────────────────────────────────
    sim_log, robot_sim = run_realtime_simulation(hybrid_results, all_dec)

    # ── PART 9: Robotics Metrics ────────────────────────────────────────────
    robotics_metrics = compute_robotics_metrics(hybrid_results, all_dec, sim_log)

    # ── PART 10: Adaptive Sampling Summary ──────────────────────────────────
    print("\n  Adaptive sampling summary figure ...")
    stas = list(hybrid_results.keys()); n_sta = len(stas)
    nc = 3; nr = math.ceil(n_sta / nc)
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(nr, nc, figsize=(18, 3 * nr))
        axes = np.array(axes).flatten()
        for ax, sta in zip(axes, stas):
            dec = all_dec[sta]
            ax.step(dec["timestamp"], dec["sampling_rate"], where="post",
                    color=PAL["primary"], lw=1.5)
            ax.fill_between(dec["timestamp"], dec["sampling_rate"],
                            step="post", alpha=0.3, color=PAL["primary"])
            ax.set_yscale("log"); ax.set_title(sta, fontsize=8, fontweight="bold")
            ax.set_ylabel("Samples/hr")
        for ax in axes[n_sta:]: ax.set_visible(False)
        fig.suptitle("Adaptive Sampling Rate — All Stations",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        savefig(fig, "adaptive_sampling_all_stations.png", DIR_FIG)

    # ── PART 11: Spatial Hotspot ────────────────────────────────────────────
    ranked, sta_info = spatial_hotspot(hybrid_results, all_dec)

    # ── PART 12: Cascade Simulation ─────────────────────────────────────────
    cascade_logs = run_cascade(hybrid_results, sensor_data)
    plot_cascade(cascade_logs, hybrid_results)

    # ── PART 13: Architecture Diagram + Animations ──────────────────────────
    plot_architecture()
    generate_all_animations(hybrid_results, all_dec, ranked, sta_info,
                            cascade_logs, sim_log=sim_log)

    # ── PART 14: Final Summary ──────────────────────────────────────────────
    final_summary(hybrid_metrics)

    # ── Save all outputs ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SAVING ALL OUTPUTS")
    print("=" * 65)
    save_all_predictions(arimax_results, lstm_results, hybrid_results)
    save_all_metrics(arimax_metrics, lstm_metrics, hybrid_metrics)
    write_system_report(hybrid_metrics, None, ranked)
    save_robotics_outputs(sim_log, robot_sim)

    # ── File listing ────────────────────────────────────────────────────────
    print("\n" + "✅ " * 20)
    print("  PIPELINE COMPLETE")
    print(f"  Root → {RESULTS_DIR}\n")
    total_files = 0
    for folder in ALL_DIRS:
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if files:
            tag = os.path.relpath(folder, RESULTS_DIR) if folder != RESULTS_DIR else "."
            print(f"  📁 {tag}/  ({len(files)} files)")
            for f in sorted(files):
                kb = os.path.getsize(os.path.join(folder, f)) // 1024
                print(f"      {f:<55} {kb:>6} KB")
            total_files += len(files)
    print(f"\n  Total files generated: {total_files}")
    print("✅ " * 20)

    return {
        "arimax_results":   arimax_results,
        "lstm_results":     lstm_results,
        "hybrid_results":   hybrid_results,
        "arimax_metrics":   arimax_metrics,
        "lstm_metrics":     lstm_metrics,
        "hybrid_metrics":   hybrid_metrics,
        "all_dec":          all_dec,
        "ranked":           ranked,
        "sta_info":         sta_info,
        "cascade_logs":     cascade_logs,
        "sim_log":          sim_log,
        "robot_sim":        robot_sim,
        "robotics_metrics": robotics_metrics,
    }


if __name__ == "__main__":
    outputs = main()
