"""
visualization/plots.py
======================
All plotting functions for the ARIMAX + LSTM + Hybrid pipeline:
  - plot_eda
  - plot_hybrid_per_station
  - plot_cross_station_dashboard
  - plot_model_comparison
  - plot_robot_dashboards
  - plot_architecture
  - animate_sampling
  - animate_cascade
  - animate_inter_station
  - animate_robot_full_dashboard
  - generate_all_animations
"""

import ast
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec  as gridspec
import matplotlib.dates     as mdates
import matplotlib.patches   as mpatches
import matplotlib.animation as animation
from matplotlib.cm     import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats

from config.settings import (
    THR_SAFE, THR_MODERATE, THR_HAZARDOUS,
    STATION_COLORS, PAL, CAT_COLORS, PLOT_STYLE,
    DIR_EDA, DIR_FIG, DIR_BASE, DIR_VID, DIR_MET,
    STATION_COORDS,
    BATTERY_LOW, SENSOR_CASCADE,
)
from utils.helpers import savefig, compute_metrics, save_animation, make_frames
from robot.decision_module import RobotDecisionModule

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False


# ── EDA ────────────────────────────────────────────────────────────────────────
def plot_eda(df: pd.DataFrame) -> None:
    """Four EDA figures → DIR_EDA."""
    print("\n  EDA plots ...")
    stas  = list(df.columns)
    cols  = STATION_COLORS[:len(stas)]
    ncols = 3; nrows = math.ceil(len(stas) / ncols)

    with plt.style.context(PLOT_STYLE):
        # 1 — Time series (per station, 30-day MA)
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows), sharex=True)
        axes = np.array(axes).flatten()
        for ax, sta, c in zip(axes, stas, cols):
            ma = df[sta].rolling(30, min_periods=1).mean()
            ax.fill_between(df.index, df[sta], alpha=0.12, color=c)
            ax.plot(df.index, df[sta], color=c, lw=0.6, alpha=0.4)
            ax.plot(df.index, ma, color=c, lw=2.0, label="30-day MA")
            ax.set_title(sta, fontsize=9, fontweight="bold")
            ax.set_ylabel("PM2.5 (µg/m³)", fontsize=8)
            for thr, lc in [(THR_SAFE, "#16A34A"), (THR_MODERATE, "#D97706"),
                            (THR_HAZARDOUS, "#DC2626")]:
                ax.axhline(thr, color=lc, lw=0.8, linestyle=":", alpha=0.7)
        for ax in axes[len(stas):]: ax.set_visible(False)
        fig.suptitle("Daily PM2.5 — All Beijing Stations (2013–2017)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        savefig(fig, "01_timeseries_all_stations.png", DIR_EDA)

        # 2 — Correlation heatmap
        corr = df.corr(); n = len(stas)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="Pearson r")
        ax.set_title("Station PM2.5 Correlation Matrix", fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig(fig, "02_correlation_heatmap.png", DIR_EDA)

        # 3 — Seasonal box (month)
        df2 = df.copy(); df2["month"] = df2.index.month
        fig, ax = plt.subplots(figsize=(16, 6))
        mdata = [df2[df2["month"] == m][stas].values.flatten() for m in range(1, 13)]
        mdata = [x[~np.isnan(x)] for x in mdata]
        bp = ax.boxplot(mdata, patch_artist=True, medianprops=dict(color="red", lw=2))
        for p, c in zip(bp["boxes"], plt.cm.RdYlGn_r(np.linspace(0, 1, 12))):
            p.set_facecolor(c)
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"])
        ax.set_title("Monthly PM2.5 Distribution — All Stations", fontsize=13, fontweight="bold")
        ax.set_ylabel("PM2.5 (µg/m³)")
        for thr, lc in [(THR_SAFE, "#16A34A"), (THR_MODERATE, "#D97706"),
                        (THR_HAZARDOUS, "#DC2626")]:
            ax.axhline(thr, color=lc, lw=1, ls=":", alpha=0.8)
        plt.tight_layout()
        savefig(fig, "03_seasonal_boxplot.png", DIR_EDA)

        # 4 — Yearly heatmap
        df2["year"] = df2.index.year
        yearly = df2.groupby("year")[stas].mean()
        df2.drop(columns=["month","year"], inplace=True)
        fig, ax = plt.subplots(figsize=(max(8, len(yearly) * 1.2),
                                        max(5, len(stas) * 0.5 + 1)))
        if _SNS:
            sns.heatmap(yearly.T, annot=True, fmt=".0f", cmap="YlOrRd",
                        ax=ax, linewidths=0.5, annot_kws={"size": 8},
                        cbar_kws={"label": "Mean PM2.5 (µg/m³)"})
        else:
            im2 = ax.imshow(yearly.T.values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(yearly))); ax.set_xticklabels(yearly.index, fontsize=9)
            ax.set_yticks(range(len(stas))); ax.set_yticklabels(stas, fontsize=8)
            plt.colorbar(im2, ax=ax, label="Mean PM2.5 (µg/m³)")
        ax.set_title("Station × Year Mean PM2.5 Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        savefig(fig, "05_station_year_heatmap.png", DIR_EDA)

    print(f"  ✓ EDA figures → {DIR_EDA}")


# ── Hybrid per-station ─────────────────────────────────────────────────────────
def plot_hybrid_per_station(results: dict) -> None:
    """4-panel result figure per station → DIR_FIG/hybrid/ (ARIMAX+LSTM components)."""
    out = os.path.join(DIR_FIG, "hybrid"); os.makedirs(out, exist_ok=True)
    for sta, col in zip(results, STATION_COLORS):
        r = results[sta]; m = r["metrics"]; ti = r["test_index"]
        with plt.style.context(PLOT_STYLE):
            fig = plt.figure(figsize=(18, 14))
            gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.32)
            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(ti, r["actual"],   color="#264653", lw=1.5, label="Actual")
            ax0.plot(ti, r["final_fc"], color=col, lw=2.2, ls="--", label="Hybrid Forecast")
            ax0.fill_between(ti, np.clip(r["final_fc"] - m["RMSE"], 0, None),
                              r["final_fc"] + m["RMSE"], color=col, alpha=0.12, label="±RMSE")
            for thr, lc in [(THR_SAFE,"#16A34A"),(THR_MODERATE,"#D97706"),(THR_HAZARDOUS,"#DC2626")]:
                ax0.axhline(thr, color=lc, lw=0.9, ls=":")
            ax0.set_title(f"{sta} — Hybrid Forecast vs Actual", fontsize=12, fontweight="bold")
            ax0.set_ylabel("PM2.5 (µg/m³)"); ax0.legend(fontsize=9)

            ax1 = fig.add_subplot(gs[1, 0])
            ax1.scatter(r["actual"], r["final_fc"], alpha=0.35, s=12, color=col, edgecolors="none")
            lim = [min(r["actual"].min(), r["final_fc"].min()) - 3,
                   max(r["actual"].max(), r["final_fc"].max()) + 3]
            ax1.plot(lim, lim, "k--", lw=1.5)
            ax1.set_xlabel("Actual PM2.5"); ax1.set_ylabel("Predicted PM2.5")
            ax1.set_title("Scatter: Actual vs Predicted", fontweight="bold")
            ax1.text(0.05, 0.92, f"MAE={m['MAE']:.2f}", transform=ax1.transAxes,
                     fontsize=11, color="navy", fontweight="bold")

            ax2 = fig.add_subplot(gs[1, 1])
            err = r["actual"] - r["final_fc"]; mu, sg = err.mean(), err.std()
            ax2.hist(err, bins=60, color=col, edgecolor="white", lw=0.4, density=True, alpha=0.8)
            x = np.linspace(mu - 4 * sg, mu + 4 * sg, 200)
            ax2.plot(x, stats.norm.pdf(x, mu, sg), "r-", lw=2, label=f"μ={mu:.2f} σ={sg:.2f}")
            ax2.axvline(0, color="black", lw=1.5, ls="--")
            ax2.set_title("Error Distribution", fontweight="bold"); ax2.legend(fontsize=9)

            ax3 = fig.add_subplot(gs[2, 0])
            mk = ["MAE","MSE","RMSE","R2"]; mv = [m[k] for k in mk]
            bars = ax3.bar(mk, mv, color=[PAL["primary"],PAL["secondary"],PAL["warning"],PAL["success"]],
                           edgecolor="white", lw=0.8)
            for b, v in zip(bars, mv):
                ax3.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax3.set_title("Error Metrics", fontweight="bold")

            ax4 = fig.add_subplot(gs[2, 1])
            nn = min(150, len(ti))
            ax4.stackplot(ti[:nn],
                          [r["arimax_pred"][:nn],
                           r["lstm_correction"][:nn]],
                          labels=["ARIMAX Component","LSTM Correction"],
                          colors=["#E76F51","#8338EC"], alpha=0.75)
            ax4.set_title("Component Contributions (first 150 hours)", fontweight="bold")
            ax4.legend(fontsize=9)

            fig.suptitle(f"Hybrid ARIMA-LSTM | {sta}",
                         fontsize=14, fontweight="bold", y=1.005)
            savefig(fig, f"hybrid_{sta.replace(' ','_')}.png", out)
    print(f"  ✓ Hybrid per-station figures → {out}")


# ── Cross-station dashboard ────────────────────────────────────────────────────
def plot_cross_station_dashboard(all_metrics: dict, tag: str = "hybrid") -> None:
    """6-panel bar + heatmap → DIR_FIG."""
    stas     = list(all_metrics.keys()); n = len(stas)
    cols     = STATION_COLORS[:n]
    met_keys = ["MAE","MSE","RMSE","R2"]

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10)); axes = axes.flatten()
        for ax, mk in zip(axes, met_keys):
            vals = [all_metrics[s][mk] for s in stas]
            bars = ax.bar(stas, vals, color=cols, edgecolor="white", lw=0.8)
            ax.set_title(mk, fontsize=12, fontweight="bold")
            ax.set_xticklabels(stas, rotation=35, ha="right", fontsize=7.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        fig.suptitle(f"Cross-Station Metrics — {tag.upper()}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        savefig(fig, f"crossstation_{tag}_bars.png", DIR_FIG)

        df_h = pd.DataFrame(all_metrics).T[met_keys].astype(float)
        norm = df_h.copy()
        for c in met_keys:
            mn, mx = df_h[c].min(), df_h[c].max()
            norm[c] = (df_h[c] - mn) / (mx - mn) if mx > mn else 0.5
        fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * n + 1.5)))
        if _SNS:
            sns.heatmap(norm, annot=df_h.round(3), fmt="g", cmap="RdYlGn_r", ax=ax,
                        linewidths=0.5, annot_kws={"size": 8},
                        cbar_kws={"label": "Normalised (green=best)"})
        else:
            im = ax.imshow(norm.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(met_keys))); ax.set_xticklabels(met_keys, fontsize=10, fontweight="bold")
            ax.set_yticks(range(n)); ax.set_yticklabels(stas, fontsize=9)
            plt.colorbar(im, ax=ax, label="Normalised (green=best)")
        ax.set_title(f"Metrics Heatmap — {tag.upper()}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig(fig, f"crossstation_{tag}_heatmap.png", DIR_FIG)
    print(f"  ✓ Cross-station dashboard ({tag})")


# ── Model comparison ───────────────────────────────────────────────────────────
def plot_model_comparison(arimax_m, lstm_m, hybrid_m) -> None:
    """Grouped bar charts comparing 3 models across stations → DIR_BASE."""
    print("\n  Model comparison figures ...")
    models = {"ARIMAX": arimax_m, "LSTM": lstm_m, "Hybrid ARIMA-LSTM": hybrid_m}

    stas   = sorted(set.intersection(*[set(v.keys()) for v in models.values()]))
    n      = len(stas); x = np.arange(n); w = 0.8 / len(models)
    mcolors= [PAL["primary"], PAL["warning"], PAL["danger"]]

    with plt.style.context(PLOT_STYLE):
        for mkey in ["MAE","RMSE","R2"]:
            fig, ax = plt.subplots(figsize=(max(14, n * 1.5), 6))
            for i, (mname, mdict) in enumerate(models.items()):
                vals   = [mdict[s][mkey] for s in stas]
                offset = (i - len(models) / 2 + 0.5) * w
                bars   = ax.bar(x + offset, vals, w, label=mname,
                                color=mcolors[i], alpha=0.85, edgecolor="white")
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
            ax.set_xticks(x); ax.set_xticklabels(stas, rotation=30, ha="right")
            ax.legend(fontsize=8, ncol=3); ax.set_ylabel(mkey)
            ax.set_title(f"All-Model {mkey} Comparison", fontsize=13, fontweight="bold")
            plt.tight_layout()
            savefig(fig, f"model_comparison_{mkey.replace('²','2')}.png", DIR_BASE)

    rows = []
    for mname, mdict in models.items():
        row = {"model": mname}
        for mk in ["MAE","MSE","RMSE","R2"]:
            vals = [mdict[s][mk] for s in stas if s in mdict and mk in mdict[s]]
            if vals: row[f"avg_{mk}"] = round(float(np.mean(vals)), 5)
        rows.append(row)
    pd.DataFrame(rows).set_index("model").to_csv(
        os.path.join(DIR_BASE, "all_models_summary.csv"))
    print(f"  ✓ Model comparison → {DIR_BASE}")


# ── Robot dashboard ───────────────────────────────────────────────────────────
def plot_robot_dashboards(results: dict, all_dec: dict) -> None:
    """5-row robot dashboard per station → DIR_FIG/robot/."""
    out = os.path.join(DIR_FIG, "robot"); os.makedirs(out, exist_ok=True)
    sc = [PAL["success"],PAL["warning"],PAL["danger"],PAL["secondary"]]

    for sta, col in zip(results, STATION_COLORS):
        r   = results[sta]; dec = all_dec[sta]
        ti  = r["test_index"]
        fig = plt.figure(figsize=(20, 22))
        gs  = gridspec.GridSpec(5, 2, hspace=0.5, wspace=0.3)
        ax0 = fig.add_subplot(gs[0, :])
        for lo, hi, bc in [(0,THR_SAFE,"#16A34A"),(THR_SAFE,THR_MODERATE,"#D97706"),
                           (THR_MODERATE,THR_HAZARDOUS,"#DC2626"),
                           (THR_HAZARDOUS,max(r["actual"].max(),r["final_fc"].max())*1.1,"#7C3AED")]:
            ax0.axhspan(lo, hi, color=bc, alpha=0.07)
        ax0.plot(ti, r["actual"], color=PAL["dark"], lw=1.5, label="Actual")
        ax0.plot(ti, r["final_fc"], color=col, lw=2.2, ls="--", label="Forecast")
        ax0.set_title(f"PM2.5 Forecast with AQI Zones — {sta}", fontweight="bold")
        ax0.legend(fontsize=9)

        # Robot state timeline
        ax1 = fig.add_subplot(gs[1, :])
        sm = {"PATROL":0,"MONITOR":1,"ALERT":2,"EMERGENCY":3}
        state_vals = dec["robot_state"].map(sm).values
        ax1.fill_between(ti[:len(state_vals)], state_vals, alpha=0.6, color=col)
        ax1.set_yticks([0,1,2,3]); ax1.set_yticklabels(["PATROL","MONITOR","ALERT","EMERGENCY"])
        ax1.set_title("Robot State Timeline", fontweight="bold")

        # Sampling rate
        ax2 = fig.add_subplot(gs[2, :])
        ax2.step(dec["timestamp"], dec["sampling_rate"], where="post",
                 color=PAL["primary"], lw=2)
        ax2.fill_between(dec["timestamp"], dec["sampling_rate"],
                         step="post", alpha=0.3, color=PAL["primary"])
        ax2.set_yscale("log"); ax2.set_title("Adaptive Sampling Rate (samples/hr)", fontweight="bold")

        # Category distribution
        ax3 = fig.add_subplot(gs[3, 0])
        cats = ["SAFE","MODERATE","UNHEALTHY","HAZARDOUS"]
        counts = [(dec["category"] == c).sum() for c in cats]
        ax3.pie(counts, labels=cats, autopct="%1.1f%%", startangle=90,
                colors=[CAT_COLORS[c] for c in cats],
                wedgeprops={"edgecolor":"white","linewidth":1.5})
        ax3.set_title("AQI Category Distribution", fontweight="bold")

        # Metrics bar
        ax4 = fig.add_subplot(gs[3, 1])
        m = r["metrics"]
        mk_keys = ["MAE","MSE","RMSE","R2"]
        ax4.bar(mk_keys, [m[k] for k in mk_keys],
                color=[PAL["primary"],PAL["warning"],PAL["danger"],PAL["secondary"]],
                edgecolor="white")
        ax4.set_title("Forecast Metrics", fontweight="bold")

        fig.suptitle(f"Robot Dashboard — {sta}", fontsize=14, fontweight="bold")
        savefig(fig, f"robot_dashboard_{sta.replace(' ','_')}.png", out)
    print(f"  ✓ Robot dashboards → {out}")


# ── Architecture diagram ───────────────────────────────────────────────────────
def plot_architecture() -> None:
    """Static system architecture diagram → DIR_FIG."""
    fig, ax = plt.subplots(figsize=(22, 14), facecolor="#0F172A")
    ax.set_xlim(0, 22); ax.set_ylim(0, 13); ax.axis("off")
    ax.set_facecolor("#0F172A")
    boxes = [
        (0.3,11.2,2.6,1.2,"INPUT\nBeijing CSV\n12 Stations","#0369A1"),
        (3.1,11.2,2.8,1.2,"DATA LOAD\n& EDA\nHourly + wd","#0369A1"),
        (6.2,11.2,2.8,1.2,"ARIMAX\nARIMA(1,0,1)\n+ 26 Exog Features","#7C3AED"),
        (9.3,11.2,2.8,1.2,"LSTM\n2-layer LSTM(16)\n27 Features","#6D28D9"),
        (12.3,10.25,0.8,0.8,"+","#374151"),
        (13.3, 9.1,2.8,1.5,"HYBRID\nARIMA-LSTM\nResidual Correction","#DC2626"),
        (0.3, 6.8,2.8,1.2,"ARIMAX\nStandalone\nForecast","#4B5563"),
        (3.4, 6.8,2.8,1.2,"LSTM\nStandalone\nForecast","#4B5563"),
        (6.5, 6.8,2.8,1.2,"HYBRID\nARIMA-LSTM\nŷ = ARIMAX + LSTM","#0D9488"),
        (9.9, 6.8,4.8,1.2,"MODEL COMPARE\nMAE  MSE  RMSE  R²","#374151"),
        (0.3, 4.5,3.5,1.2,"ROBOT MODULE\nPATROL/MONITOR\nALERT/EMGCY","#D97706"),
        (4.1, 4.5,3.5,1.2,"ADAPTIVE\nSAMPLING\n1-60/hr","#0D9488"),
        (7.9, 4.5,3.5,1.2,"SPATIAL MAP\n+HOTSPOT\n+PATROL ROUTE","#B91C1C"),
        (11.7,4.5,3.5,1.2,"CASCADE SIM\nSensor activation\n+Source diag","#7C2D12"),
        (15.5,4.5,5.8,1.2,"ANIMATIONS\nSampling | Cascade\nInter-Station","#1E3A5F"),
        (0.3, 2.3,21.0,1.2,"OUTPUTS: predictions/ | metrics/ | figures/ | robot_decisions/ | videos/","#1E293B"),
        (0.3, 0.8,21.0,1.0,"ROBOT ACTIONS: PATROL → MONITOR → ALERT → EMERGENCY  "
                             "| Move to Hotspot | Adaptive Sampling | Cascade Sensor Activation","#0F172A"),
    ]
    for (x,y,w,h,lbl,c) in boxes:
        rect = plt.Rectangle((x,y),w,h,fc=c,ec="white",lw=1.5,alpha=0.90,zorder=3)
        ax.add_patch(rect)
        ax.text(x+w/2,y+h/2,lbl,ha="center",va="center",color="white",
                fontsize=7.5,fontweight="bold",zorder=4,linespacing=1.4)
    ax.set_title("System Architecture — Robot-Assisted Environmental Monitoring",
                 fontsize=14,fontweight="bold",color="white",pad=12)
    savefig(fig, "system_architecture.png", DIR_FIG)
    print(f"  ✓ Architecture diagram → {DIR_FIG}")


# ── Animations ─────────────────────────────────────────────────────────────────
def animate_sampling(sta: str, results: dict, all_dec: dict, fps: int = 6) -> None:
    """PM2.5 rolling plot + sampling gauge + state badge → DIR_VID."""
    r       = results[sta]; dec = all_dec[sta]
    actual  = r["actual"]; pred = r["final_fc"]
    dates   = r["test_index"]; rates = dec["sampling_rate"].values
    cats    = dec["category"].values; states = dec["robot_state"].values
    N       = min(len(actual), len(rates), len(dates))
    frames, _ = make_frames(N)
    gauge_h = [1, 4, 12, 60]
    sc_map  = {"PATROL":PAL["success"],"MONITOR":PAL["warning"],
               "ALERT":PAL["danger"],"EMERGENCY":PAL["secondary"]}
    cat_num = {"SAFE":0,"MODERATE":1,"UNHEALTHY":2,"HAZARDOUS":3}

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    fig.suptitle(f"Robot Sampling — {sta}", fontsize=13, fontweight="bold")
    gs   = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    ax_ts= fig.add_subplot(gs[0,:])
    ax_g = fig.add_subplot(gs[1,0])
    ax_s = fig.add_subplot(gs[1,1])

    ax_ts.set_xlim(dates[0], dates[N-1])
    ax_ts.set_ylim(0, max(actual.max(), pred.max()) * 1.15)
    for thr, lc in [(THR_SAFE,"#16A34A"),(THR_MODERATE,"#D97706"),(THR_HAZARDOUS,"#DC2626")]:
        ax_ts.axhline(thr, color=lc, lw=0.9, ls=":")
    la, = ax_ts.plot([], [], color=PAL["dark"], lw=1.5, label="Actual")
    lp, = ax_ts.plot([], [], color=PAL["primary"], lw=2.0, ls="--", label="Predicted")
    vl  = ax_ts.axvline(dates[0], color="red", lw=1.5, alpha=0.8)
    td  = ax_ts.text(0.01, 0.95, "", transform=ax_ts.transAxes, fontsize=9,
                     color="black", fontweight="bold", va="top")
    ax_ts.legend(fontsize=8)

    ax_g.bar(range(4), gauge_h,
             color=[CAT_COLORS[c] for c in ["SAFE","MODERATE","UNHEALTHY","HAZARDOUS"]],
             alpha=0.25, edgecolor="white", lw=1)
    hl = ax_g.bar([0], [1], color="none", edgecolor="gold", lw=3)
    ax_g.set_xticks(range(4))
    ax_g.set_xticklabels(["SAFE\n1/hr","MOD\n4/hr","UNH\n12/hr","HAZ\n60/hr"], fontsize=7)
    ax_g.set_yscale("log"); ax_g.set_title("Sampling Rate", fontweight="bold")

    ax_s.axis("off"); ax_s.set_title("Robot State", fontweight="bold")
    badge = ax_s.text(0.5, 0.55, "PATROL", ha="center", va="center", fontsize=22,
                      fontweight="bold", transform=ax_s.transAxes,
                      bbox=dict(boxstyle="round,pad=0.6", fc=PAL["success"], alpha=0.9))
    pm_t  = ax_s.text(0.5, 0.2, "", ha="center", va="center", fontsize=12,
                      transform=ax_s.transAxes)

    def update(fi):
        i = frames[fi]
        la.set_data(dates[:i+1], actual[:i+1])
        lp.set_data(dates[:i+1], pred[:i+1])
        vl.set_xdata([dates[i], dates[i]])
        td.set_text(str(dates[i].date()))
        cat = cats[i]; ci = list(cat_num.keys()).index(cat)
        hl[0].set_x(ci - 0.4); hl[0].set_width(0.8); hl[0].set_height(gauge_h[ci])
        st = states[i]
        badge.set_text(st); badge.get_bbox_patch().set_facecolor(sc_map.get(st, PAL["dark"]))
        pm_t.set_text(f"PM2.5={pred[i]:.1f} µg/m³\nRate={rates[i]}/hr")
        return la, lp, vl, td, badge, pm_t

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                    interval=1000 // fps, blit=False, repeat=False)
    save_animation(anim, os.path.join(DIR_VID, f"sampling_{sta.replace(' ','_')}"), fps=fps)
    plt.close(fig)


def animate_inter_station(ranked, sta_info, results, fps=5) -> None:
    """Robot navigating between stations based on PM2.5 priority → DIR_VID."""
    stations = [s for s, _ in ranked]
    n_days   = min(len(results[s]["final_fc"]) for s in stations)
    n_frames = min(n_days, 100); step = max(1, n_days // n_frames)
    frames   = list(range(0, n_days, step)); T = len(frames)
    ref_dates= results[stations[0]]["test_index"][:n_days]
    pm_mat   = {s: results[s]["final_fc"][:n_days] for s in stations}
    rdm      = RobotDecisionModule()

    cur = stations[0]; dwell = 0; MIN_D = 3
    pos_lat = [sta_info[cur]["lat"]]; pos_lon = [sta_info[cur]["lon"]]
    at_sta = [cur]; log = []
    for fi, ti in enumerate(frames):
        dwell += 1; cur_pm = pm_mat[cur][ti]; cur_cat = rdm.classify(cur_pm)
        move = None
        if dwell >= MIN_D:
            best_pm = cur_pm; best_ci = ["SAFE","MODERATE","UNHEALTHY","HAZARDOUS"].index(cur_cat)
            for s in stations:
                if s == cur: continue
                s_pm = pm_mat[s][ti]; s_ci = ["SAFE","MODERATE","UNHEALTHY","HAZARDOUS"].index(rdm.classify(s_pm))
                if s_ci > best_ci or (s_ci == best_ci and s_pm > best_pm * 1.1):
                    best_pm = s_pm; best_ci = s_ci; move = s
        if move: dwell = 0; cur = move
        pos_lat.append(sta_info[cur]["lat"]); pos_lon.append(sta_info[cur]["lon"])
        at_sta.append(cur)
        log.append({"frame":fi,"date":str(ref_dates[ti])[:10],"station":cur,
                    "pm25":round(cur_pm,2),"category":cur_cat,
                    "action":f"MOVE→{move}" if move else f"DWELL@{cur}"})

    norm_pm = Normalize(vmin=0, vmax=THR_HAZARDOUS); cmap_pm = plt.cm.RdYlGn_r
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    fig.suptitle("Robot Inter-Station Movement", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)
    ax_m= fig.add_subplot(gs[:,0:2]); ax_pm = fig.add_subplot(gs[0,2])
    ax_rk= fig.add_subplot(gs[1,2])

    sta_sc = {}
    for s in stations:
        c   = cmap_pm(norm_pm(np.mean(pm_mat[s])))
        sc2 = ax_m.scatter(sta_info[s]["lon"], sta_info[s]["lat"],
                           s=250, color=c, edgecolors="white", lw=1.5, zorder=4, alpha=0.75)
        ax_m.annotate(s[:12], (sta_info[s]["lon"], sta_info[s]["lat"]),
                      textcoords="offset points", xytext=(4,4), fontsize=7, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))
        sta_sc[s] = sc2
    rd, = ax_m.plot(sta_info[stations[0]]["lon"], sta_info[stations[0]]["lat"],
                    marker="^", color="blue", ms=18, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=10, label="Robot")
    rt, = ax_m.plot([], [], "b--", lw=1.5, alpha=0.4, zorder=8)
    ax_m.legend(fontsize=9, loc="lower right")
    sm2 = ScalarMappable(cmap=cmap_pm, norm=norm_pm); sm2.set_array([])
    plt.colorbar(sm2, ax=ax_m, label="Mean PM2.5 (µg/m³)", shrink=0.7)
    ax_m.set_title("Robot Position (▲=Robot)", fontweight="bold")

    ax_pm.set_title("Current Station PM2.5", fontweight="bold")
    lc, = ax_pm.plot([], [], color=PAL["primary"], lw=2)
    vlc = ax_pm.axvline(0, color="red", lw=1.5, alpha=0.7)
    ts_txt = ax_pm.text(0.05, 0.92, "", transform=ax_pm.transAxes, fontsize=9, fontweight="bold")
    ax_pm.set_xlim(0, n_days); ax_pm.set_ylim(0, max(pm_mat[s].max() for s in stations) * 1.1)

    sn_short = [s[:12] for s in stations]
    bh = [np.mean(pm_mat[s]) for s in stations]
    ax_rk.barh(sn_short, bh, color=[cmap_pm(norm_pm(v)) for v in bh], edgecolor="white", lw=0.8)
    hl_bar, = ax_rk.plot([], [], "yo", ms=12, zorder=10)
    ax_rk.set_title("Station Priority", fontweight="bold")
    dt = fig.text(0.5, 0.97, "", ha="center", fontsize=11, fontweight="bold")
    trlat = []; trlon = []

    def upd(fi):
        if fi >= len(pos_lat) - 1: fi = len(pos_lat) - 2
        lat = pos_lat[fi+1]; lon = pos_lon[fi+1]; cur2 = at_sta[fi+1]
        rd.set_data([lon], [lat]); trlat.append(lat); trlon.append(lon)
        rt.set_data(trlon, trlat)
        for s in stations: sta_sc[s].set_sizes([250]); sta_sc[s].set_edgecolors("white")
        if cur2 in sta_sc: sta_sc[cur2].set_sizes([500]); sta_sc[cur2].set_edgecolors("gold")
        pd_data = pm_mat[cur2][:frames[fi]+1 if fi < T else n_days]
        lc.set_data(list(range(len(pd_data))), pd_data)
        vlc.set_xdata([frames[fi] if fi < T else n_days-1] * 2)
        ts_txt.set_text(f"@ {cur2[:15]}")
        if cur2 in stations:
            idx_s = stations.index(cur2); hl_bar.set_data([bh[idx_s]], [idx_s])
        row = log[min(fi, len(log)-1)]
        dt.set_text(f"Date:{row['date']} PM2.5={row['pm25']:.1f} {row['category']} | {row['action']}")
        return rd, rt, lc, vlc, ts_txt, hl_bar, dt

    anim = animation.FuncAnimation(fig, upd, frames=T, interval=1000//fps,
                                    blit=False, repeat=False)
    save_animation(anim, os.path.join(DIR_VID, "robot_inter_station"), fps=fps)
    plt.close(fig)


def generate_all_animations(results, all_dec, ranked, sta_info,
                              logs, sim_log=None) -> None:
    """Wrapper that generates all animation types."""
    print("\n" + "═" * 65)
    print("  PART 15 — ANIMATIONS & VIDEOS  [UPGRADED]")
    print("═" * 65)
    os.makedirs(DIR_VID, exist_ok=True)

    for sta in results:
        try:
            animate_sampling(sta, results, all_dec)
        except Exception as e:
            print(f"  ⚠ sampling {sta}: {e}")

    for sta in (logs or {}):
        try:
            animate_cascade(sta, logs[sta])
        except Exception as e:
            print(f"  ⚠ cascade {sta}: {e}")

    try:
        animate_inter_station(ranked, sta_info, results)
    except Exception as e:
        print(f"  ⚠ inter-station: {e}")

    if sim_log is not None:
        try:
            animate_robot_full_dashboard(sim_log, sta_info)
        except Exception as e:
            print(f"  ⚠ full dashboard: {e}")

    print("\n  Video summary:")
    if os.path.isdir(DIR_VID):
        for f in sorted(os.listdir(DIR_VID)):
            kb = os.path.getsize(os.path.join(DIR_VID, f)) // 1024
            print(f"    📹 {f:<55} {kb} KB")


def animate_cascade(sta: str, log, fps: int = 4) -> None:
    """
    Cascade animation: PM2.5 time-series (original vs refined) on the left,
    live sensor activation panel on the right → DIR_VID/cascade_{sta}.mp4.

    Left panel shows the date marker advancing through the test period.
    Right panel lights up each sensor (coloured box) when it becomes active,
    and shows its live reading value alongside.
    A footer text shows the diagnosed pollution source for the current day.
    """
    log = log.copy()
    log["date_dt"] = pd.to_datetime(log["date"])
    N = len(log)
    frames, _ = make_frames(N, max_frames=100)

    # Colour palette for each sensor
    sc_map = {
        "PM2.5": "#264653", "PM10": "#E9C46A", "WSPM": "#2A9D8F",
        "SO2":   "#E76F51", "NO2":  "#E63946", "TEMP": "#457B9D",
        "CO":    "#8338EC", "O3":   "#06D6A0", "PRES": "#FFBE0B",
        "DEWP":  "#FB5607", "RAIN": "#3A86FF",
    }
    sensors_ord = list(sc_map.keys())

    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.suptitle(f"Cascade Simulation — {sta}", fontsize=13, fontweight="bold")
    gs     = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
    ax_ts  = fig.add_subplot(gs[0])
    ax_sen = fig.add_subplot(gs[1])

    # ── Left: PM2.5 time series ───────────────────────────────────────────────
    y_max = max(log["pm25_predicted"].max(), log["pm25_refined"].max()) * 1.15
    ax_ts.set_xlim(log["date_dt"].iloc[0], log["date_dt"].iloc[-1])
    ax_ts.set_ylim(0, y_max)
    ax_ts.axhspan(0, THR_SAFE,           color="#16A34A", alpha=0.07)
    ax_ts.axhspan(THR_SAFE, THR_MODERATE, color="#D97706", alpha=0.07)
    ax_ts.axhspan(THR_MODERATE, THR_HAZARDOUS, color="#DC2626", alpha=0.07)
    for thr, lc in [(THR_SAFE, "#16A34A"), (THR_MODERATE, "#D97706"),
                    (THR_HAZARDOUS, "#DC2626")]:
        ax_ts.axhline(thr, color=lc, lw=0.9, ls=":")
    lo, = ax_ts.plot([], [], color=PAL["muted"], lw=1.3, ls=":", label="Original", alpha=0.7)
    lr, = ax_ts.plot([], [], color=PAL["danger"], lw=2.2, ls="--", label="Refined")
    vl  = ax_ts.axvline(log["date_dt"].iloc[0], color="black", lw=1.5, alpha=0.6)
    ti_txt = ax_ts.text(
        0.01, 0.97, "", transform=ax_ts.transAxes,
        fontsize=8, va="top", fontweight="bold",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    ax_ts.legend(fontsize=8)
    ax_ts.set_title("PM2.5: Original vs Refined", fontweight="bold")

    # ── Right: Sensor panel ───────────────────────────────────────────────────
    ax_sen.axis("off")
    ax_sen.set_xlim(0, 1)
    ax_sen.set_ylim(-0.5, len(sensors_ord) - 0.5)
    ax_sen.set_title("Active Sensors", fontweight="bold")
    patches = {}; texts = {}
    for si, sen in enumerate(sensors_ord):
        y    = len(sensors_ord) - 1 - si
        rect = plt.Rectangle((0.02, y - 0.4), 0.96, 0.75,
                              fc="#E2E8F0", ec="white", lw=1.5, alpha=0.4)
        ax_sen.add_patch(rect)
        ax_sen.text(0.06, y + 0.05, sen,
                    fontsize=9, fontweight="bold", va="center")
        vt = ax_sen.text(0.92, y + 0.05, "—",
                         fontsize=8, va="center", ha="right", color="#94A3B8")
        patches[sen] = rect; texts[sen] = vt

    diag = fig.text(
        0.72, 0.04, "", fontsize=8, ha="center", style="italic",
        color=PAL["danger"],
        bbox=dict(boxstyle="round", fc="#FEF2F2", alpha=0.9),
    )

    def update(fi):
        i   = frames[fi]
        row = log.iloc[i]
        dts = log["date_dt"].iloc[:i + 1]
        lo.set_data(dts, log["pm25_predicted"].iloc[:i + 1])
        lr.set_data(dts, log["pm25_refined"].iloc[:i + 1])
        vl.set_xdata([row["date_dt"], row["date_dt"]])
        delta = row["delta"]
        ti_txt.set_text(
            f"{row['date']}  AQI: {row['aqi_category']}\n"
            f"PM2.5: {row['pm25_predicted']:.1f} → {row['pm25_refined']:.1f} "
            f"(Δ{delta:+.1f})"
        )
        active = (row["sensors_active"]
                  if isinstance(row["sensors_active"], list)
                  else ast.literal_eval(str(row["sensors_active"])))
        for sen in sensors_ord:
            on = sen in active
            patches[sen].set_facecolor(
                sc_map.get(sen, "#94A3B8") if on else "#E2E8F0"
            )
            patches[sen].set_alpha(0.85 if on else 0.25)
            rk = f"reading_{sen}"
            if on and rk in row and pd.notna(row.get(rk)):
                texts[sen].set_text(f"{row[rk]:.1f}")
                texts[sen].set_color("#0F172A")
            elif on:
                texts[sen].set_text("ON")
                texts[sen].set_color("#0F172A")
            else:
                texts[sen].set_text("—")
                texts[sen].set_color("#CBD5E1")
        diag.set_text(f"Source: {str(row['source'])[:55]}")
        return lo, lr, vl, ti_txt, diag

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 // fps, blit=False, repeat=False,
    )
    save_animation(
        anim,
        os.path.join(DIR_VID, f"cascade_{sta.replace(' ', '_')}"),
        fps=fps,
    )
    plt.close(fig)


def animate_robot_full_dashboard(sim_log, sta_info: dict, fps: int = 5) -> None:
    """
    Animated full robot dashboard → DIR_VID/robot_full_dashboard.mp4.

    Layout:
      Left (2/3 width): 2-D lat/lon map with station bubbles (colour = mean PM2.5)
                         and robot trail (blue dashed line, triangle marker).
      Top-right        : Battery level bar — colour shifts green → yellow → red.
      Bottom-right     : Sampling rate bar — colour matches current robot state.

    One frame per unique date in sim_log (capped at 120 frames).
    """
    if sim_log is None or sim_log.empty:
        print("  ⚠ animate_robot_full_dashboard: sim_log is empty — skipping")
        return

    dates       = sim_log["date"].unique()
    n_frames    = min(len(dates), 120)
    step        = max(1, len(dates) // n_frames)
    frame_dates = dates[::step]
    stations    = list(sta_info.keys())
    cmap_pm     = plt.cm.RdYlGn_r
    norm_pm     = Normalize(vmin=0, vmax=THR_HAZARDOUS)

    # Pre-compute mean PM2.5 per station for static bubble colour
    mean_pm_sta = {}
    for sta in stations:
        sub = sim_log[sim_log["station"] == sta]
        mean_pm_sta[sta] = sub["pm25_pred"].mean() if not sub.empty else 0.0

    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.suptitle("Robot Full Dashboard — Real-Time Simulation",
                 fontsize=13, fontweight="bold")
    gs      = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.35)
    ax_map  = fig.add_subplot(gs[:, 0:2])
    ax_batt = fig.add_subplot(gs[0, 2])
    ax_rate = fig.add_subplot(gs[1, 2])

    # Static station scatter (coloured by mean PM2.5)
    sta_scatters = {}
    for sta in stations:
        if sta not in sta_info:
            continue
        info = sta_info[sta]
        pm   = mean_pm_sta.get(sta, 0)
        sc   = ax_map.scatter(
            info["lon"], info["lat"], s=220,
            color=cmap_pm(norm_pm(pm)),
            edgecolors="white", lw=1.5, zorder=4, alpha=0.8,
        )
        ax_map.annotate(
            sta[:11], (info["lon"], info["lat"]),
            xytext=(3, 3), textcoords="offset points",
            fontsize=6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7),
        )
        sta_scatters[sta] = sc

    sm_cb = ScalarMappable(cmap=cmap_pm, norm=norm_pm)
    sm_cb.set_array([])
    plt.colorbar(sm_cb, ax=ax_map, label="Mean PM2.5 (µg/m³)", shrink=0.7)

    robot_dot, = ax_map.plot([], [], "^", color="blue", ms=16,
                              markeredgecolor="white", markeredgewidth=1.5,
                              zorder=10, label="Robot")
    trail_line, = ax_map.plot([], [], "b--", lw=1.2, alpha=0.4, zorder=8)
    ax_map.legend(fontsize=8, loc="lower right")
    ax_map.set_title("Robot Position (▲)", fontweight="bold")
    ax_map.set_xlabel("Longitude"); ax_map.set_ylabel("Latitude")

    # Battery bar
    batt_bar = ax_batt.bar(["Battery"], [100.0],
                            color=PAL["success"], width=0.4, edgecolor="white")
    ax_batt.set_ylim(0, 110)
    ax_batt.set_title("Battery %", fontweight="bold")
    ax_batt.axhline(BATTERY_LOW, color=PAL["danger"], lw=1.2, ls="--",
                    alpha=0.7, label=f"Low = {BATTERY_LOW:.0f}%")
    ax_batt.legend(fontsize=7)
    batt_txt = ax_batt.text(0, 105, "100%", ha="center",
                            fontsize=12, fontweight="bold")

    # Sampling rate bar
    rate_bar = ax_rate.bar(["Sampling"], [1.0],
                            color=PAL["primary"], width=0.4, edgecolor="white")
    ax_rate.set_ylim(0, 70)
    ax_rate.set_title("Samples / hr", fontweight="bold")
    rate_txt = ax_rate.text(0, 65, "1/hr", ha="center", fontsize=12)

    date_txt = fig.text(0.5, 0.97, "", ha="center",
                        fontsize=10, fontweight="bold")

    trail_lons, trail_lats = [], []
    state_colors = {
        "PATROL":    PAL["success"],
        "MONITOR":   PAL["warning"],
        "ALERT":     PAL["danger"],
        "EMERGENCY": PAL["secondary"],
    }

    def update(fi):
        d   = frame_dates[fi]
        sub = sim_log[sim_log["date"] == d]
        if sub.empty:
            return robot_dot, trail_line, batt_txt, rate_txt, date_txt

        row   = sub.iloc[0]
        lat   = float(row.get("robot_lat", 0))
        lon   = float(row.get("robot_lon", 0))
        batt  = float(row.get("battery", 100))
        sr    = float(sub["sampling_rate"].max())
        state = str(row.get("robot_state", "PATROL"))

        robot_dot.set_data([lon], [lat])
        trail_lons.append(lon); trail_lats.append(lat)
        trail_line.set_data(trail_lons, trail_lats)

        # Battery bar colour: green > 50%, yellow > 20%, red below
        batt_bar[0].set_height(batt)
        batt_bar[0].set_color(
            PAL["success"] if batt > 50 else
            PAL["warning"] if batt > BATTERY_LOW else PAL["danger"]
        )
        batt_txt.set_text(f"{batt:.0f}%")
        batt_txt.set_y(batt + 3)

        rate_bar[0].set_height(sr)
        rate_bar[0].set_color(state_colors.get(state, PAL["primary"]))
        rate_txt.set_text(f"{sr:.0f}/hr")

        robot_sta = str(row.get("robot_station", "—"))
        returning = bool(row.get("returning_base", False))
        status    = "← returning to base" if returning else f"@ {robot_sta[:16]}"
        date_txt.set_text(
            f"Date: {str(d)[:10]}  |  State: {state}  |  {status}  |  "
            f"Battery: {batt:.0f}%"
        )
        return robot_dot, trail_line, batt_txt, rate_txt, date_txt

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_dates),
        interval=1000 // fps, blit=False, repeat=False,
    )
    save_animation(anim, os.path.join(DIR_VID, "robot_full_dashboard"), fps=fps)
    plt.close(fig)
    print(f"  ✓ Full dashboard animation → {DIR_VID}")
