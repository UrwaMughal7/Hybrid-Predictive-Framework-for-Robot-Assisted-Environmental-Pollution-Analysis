"""
robot/decision_module.py
========================
4-state robot decision state machine and movement simulator:
  - RobotDecisionModule    : PATROL → MONITOR → ALERT → EMERGENCY
  - run_robot_decisions    : apply to all stations
  - RobotMovementSimulator : lat/lon movement with battery model
  - run_realtime_simulation: replay test period day by day
  - compute_robotics_metrics
  - save_robotics_outputs
"""

import math
import os

import numpy as np
import pandas as pd

from config.settings import (
    SEED, TEST_FRAC,
    THR_SAFE, THR_MODERATE, THR_HAZARDOUS,
    TREND_ESCALATE_RATE,
    MOVE_SPEED_DEG, DIST_COST,
    BATTERY_MOVE_COST, BATTERY_SAMPLE_COST,
    BATTERY_IDLE_CHARGE, BATTERY_LOW,
    BASE_STATION, STATION_COORDS,
    DIR_ROBOT, DIR_MOV, DIR_ENRG, DIR_MET,
)
from utils.helpers import compute_metrics


# ── 4-state decision machine ──────────────────────────────────────────────────
class RobotDecisionModule:
    """
    4-state machine: PATROL → MONITOR → ALERT → EMERGENCY.
    Adaptive sampling rates: 1 / 4 / 12 / 60 samples per hour.
    """
    _CATS = [
        ("SAFE",      0,             THR_SAFE),
        ("MODERATE",  THR_SAFE,      THR_MODERATE),
        ("UNHEALTHY", THR_MODERATE,  THR_HAZARDOUS),
        ("HAZARDOUS", THR_HAZARDOUS, np.inf),
    ]
    _RATES  = {"SAFE": 1, "MODERATE": 4, "UNHEALTHY": 12, "HAZARDOUS": 60}
    _STATES = {
        "SAFE":      "PATROL",
        "MODERATE":  "MONITOR",
        "UNHEALTHY": "ALERT",
        "HAZARDOUS": "EMERGENCY",
    }
    _ACTIONS = {
        "PATROL":    "Routine patrol | Normal sampling (1/hr) | Log data",
        "MONITOR":   "Increase sampling (4/hr) | Alert neighbours | Slow patrol",
        "ALERT":     "Move to source | Sample 12/hr | Public alert | Extra sensors",
        "EMERGENCY": "Stationary at hotspot | 60/hr | Emergency broadcast | All sensors ON",
    }

    def __init__(self):
        self.state    = None
        self.prev_pm  = None
        self.history  = []

    def classify(self, pm: float) -> str:
        for cat, lo, hi in self._CATS:
            if lo <= pm < hi:
                return cat
        return "HAZARDOUS"

    def trend(self, pm: float) -> str:
        if self.prev_pm is None:
            return "STABLE"
        d = pm - self.prev_pm
        return "RISING" if d > 5 else ("FALLING" if d < -5 else "STABLE")

    def sampling_rate(self, cat: str, trnd: str) -> int:
        base = self._RATES[cat]
        if trnd == "RISING":
            return min(base * 2, 60)
        if trnd == "FALLING" and self.state in ("EMERGENCY", "ALERT"):
            return max(base, self._RATES["UNHEALTHY"])
        return base

    def decide(self, pm: float, station: str, ts=None) -> dict:
        cat  = self.classify(pm)
        trnd = self.trend(pm)

        # Early-warning escalation: if rising faster than threshold, go up one level
        if trnd == "RISING" and self.prev_pm is not None:
            rate_of_change = pm - self.prev_pm
            cats_order = ["SAFE", "MODERATE", "UNHEALTHY", "HAZARDOUS"]
            ci = cats_order.index(cat)
            if rate_of_change > TREND_ESCALATE_RATE and ci < 3:
                cat = cats_order[ci + 1]

        nst  = self._STATES[cat]
        rate = self.sampling_rate(cat, trnd)
        rec  = {
            "timestamp":    ts,
            "station":      station,
            "pm25":         round(float(pm), 2),
            "category":     cat,
            "trend":        trnd,
            "robot_state":  nst,
            "sampling_rate":rate,
            "action":       self._ACTIONS[nst],
            "state_changed":nst != self.state,
        }
        self.history.append(rec)
        self.state   = nst
        self.prev_pm = pm
        return rec

    def process_series(self, pm_arr, station: str, dates=None) -> pd.DataFrame:
        self.history = []; self.state = None; self.prev_pm = None
        for i, pm in enumerate(pm_arr):
            self.decide(pm, station, dates[i] if dates is not None else i)
        return pd.DataFrame(self.history)


def run_robot_decisions(results: dict) -> dict:
    """Run 4-state RobotDecisionModule on all stations' forecasts."""
    print("\n" + "═" * 65)
    print("  PART 11 — ROBOT DECISION MODULE")
    print("═" * 65)
    os.makedirs(DIR_ROBOT, exist_ok=True)

    rdm     = RobotDecisionModule()
    all_dec = {}
    for sta, r in results.items():
        ti  = r["test_index"]
        dec = rdm.process_series(r["final_fc"], sta, ti)
        all_dec[sta] = dec
        dec.to_csv(
            os.path.join(DIR_ROBOT, f"robot_log_{sta.replace(' ','_')}.csv"),
            index=False,
        )
        n = len(dec)
        for cat in ["SAFE", "MODERATE", "UNHEALTHY", "HAZARDOUS"]:
            c = (dec["category"] == cat).sum()
            print(f"  {sta:<20} {cat:<12}: {c:>4}/{n} ({100*c/n:.1f}%)")

    pd.concat(all_dec.values()).to_csv(
        os.path.join(DIR_ROBOT, "all_stations_robot_log.csv"), index=False
    )
    print(f"  ✓ Robot logs → {DIR_ROBOT}")
    return all_dec


# ── Movement simulator ─────────────────────────────────────────────────────────
class RobotMovementSimulator:
    """
    Maintains robot position on the lat/lon grid across timesteps.

    Movement logic:
      - Score each station: score = pm25 − DIST_COST × distance_deg
      - Move at most MOVE_SPEED_DEG per timestep toward chosen target
      - If battery < BATTERY_LOW, head back to base station

    Battery model:
      - Moving   → −BATTERY_MOVE_COST per step
      - Idle     → +BATTERY_IDLE_CHARGE per step (recharging)
      - Sampling → −BATTERY_SAMPLE_COST × (sampling_rate / 60) per step
      - Clamped to [0, 100]
    """

    def __init__(self, base_station: str = BASE_STATION):
        coord = next(
            (v for k, v in STATION_COORDS.items()
             if base_station.lower() in k.lower() or k.lower() in base_station.lower()),
            (39.88, 116.42),
        )
        self.lat = float(coord[0]); self.lon = float(coord[1])
        self.base_lat = self.lat;   self.base_lon = self.lon
        self.target_lat = self.lat; self.target_lon = self.lon
        self.current_station  = base_station
        self.battery          = 100.0
        self.path             = [(self.lat, self.lon)]
        self.returning_to_base= False

    def _dist_deg(self, lat2, lon2):
        return math.sqrt((self.lat - lat2) ** 2 + (self.lon - lon2) ** 2)

    def _step_toward(self, tlat, tlon):
        dlat = tlat - self.lat; dlon = tlon - self.lon
        dist = math.sqrt(dlat ** 2 + dlon ** 2)
        if dist < MOVE_SPEED_DEG:
            self.lat, self.lon = tlat, tlon
        else:
            self.lat += MOVE_SPEED_DEG * dlat / dist
            self.lon += MOVE_SPEED_DEG * dlon / dist

    def _choose_target(self, pm_snapshot: dict):
        best_score = -1e9; best_sta = self.current_station
        best_coord = (self.target_lat, self.target_lon)
        for sta, pm in pm_snapshot.items():
            coord = next(
                (v for k, v in STATION_COORDS.items()
                 if k.lower() in sta.lower() or sta.lower() in k.lower()),
                None,
            )
            if coord is None: continue
            d     = self._dist_deg(coord[0], coord[1])
            score = pm - DIST_COST * d
            if score > best_score:
                best_score = score; best_sta = sta; best_coord = coord
        self.current_station      = best_sta
        self.target_lat, self.target_lon = float(best_coord[0]), float(best_coord[1])

    def _update_battery(self, sampling_rate: float):
        moving = (
            abs(self.lat - self.target_lat) > 1e-4 or
            abs(self.lon - self.target_lon) > 1e-4
        )
        self.battery += BATTERY_IDLE_CHARGE if not moving else -BATTERY_MOVE_COST
        self.battery -= BATTERY_SAMPLE_COST * (sampling_rate / 60.0)
        self.battery  = float(np.clip(self.battery, 0.0, 100.0))

    def update(self, pm_snapshot: dict, sampling_rate: float = 1.0):
        if self.battery < BATTERY_LOW:
            self.returning_to_base = True
            self.target_lat = self.base_lat
            self.target_lon = self.base_lon
            if (abs(self.lat - self.base_lat) < MOVE_SPEED_DEG and
                    abs(self.lon - self.base_lon) < MOVE_SPEED_DEG):
                self.battery = 100.0; self.returning_to_base = False
        else:
            self.returning_to_base = False
            self._choose_target(pm_snapshot)

        self._step_toward(self.target_lat, self.target_lon)
        self._update_battery(sampling_rate)
        self.path.append((self.lat, self.lon))
        return self.lat, self.lon


def run_realtime_simulation(results: dict, all_dec: dict) -> tuple:
    """
    Replay the test period one day at a time across all stations.
    Returns (sim_log DataFrame, RobotMovementSimulator).
    """
    print("\n" + "═" * 65)
    print("  PART NEW-A — REAL-TIME SIMULATION LOOP")
    print("═" * 65)

    robot_sim = RobotMovementSimulator()
    ref_sta   = max(results, key=lambda s: len(results[s]["test_index"]))
    all_dates = results[ref_sta]["test_index"]

    pm_by_date = {}
    for sta, r in results.items():
        for date, pm in zip(r["test_index"], r["final_fc"]):
            pm_by_date.setdefault(str(date)[:10], {})[sta] = float(pm)

    rows = []
    for date in all_dates:
        date_key = str(date)[:10]
        pm_snap  = pm_by_date.get(date_key, {})
        cur_sta  = robot_sim.current_station
        dec_sta  = all_dec.get(cur_sta, all_dec[ref_sta])
        mask     = dec_sta["timestamp"].astype(str).str[:10] == date_key
        if mask.any():
            row_dec  = dec_sta[mask].iloc[0]
            srate    = float(row_dec["sampling_rate"])
            category = row_dec["category"]
            state    = row_dec["robot_state"]
        else:
            srate = 1.0; category = "SAFE"; state = "PATROL"

        lat, lon = robot_sim.update(pm_snap, sampling_rate=srate)

        for sta, pm in pm_snap.items():
            dec_row = all_dec.get(sta, all_dec[ref_sta])
            m2 = dec_row["timestamp"].astype(str).str[:10] == date_key
            if m2.any():
                r2 = dec_row[m2].iloc[0]
                cat2, st2, sr2 = r2["category"], r2["robot_state"], r2["sampling_rate"]
            else:
                cat2, st2, sr2 = "SAFE", "PATROL", 1.0
            rows.append({
                "date":           date_key, "station":        sta,
                "pm25_pred":      round(pm, 2), "category":       cat2,
                "robot_state":    st2, "sampling_rate":  sr2,
                "robot_lat":      round(lat, 6), "robot_lon":      round(lon, 6),
                "robot_station":  robot_sim.current_station,
                "battery":        round(robot_sim.battery, 2),
                "returning_base": robot_sim.returning_to_base,
            })

    sim_log = pd.DataFrame(rows)
    sim_log.to_csv(os.path.join(DIR_ROBOT, "simulation_log.csv"), index=False)
    print(f"  ✓ Simulation log → {DIR_ROBOT}/simulation_log.csv  ({len(sim_log)} rows)")
    return sim_log, robot_sim


def compute_robotics_metrics(results: dict, all_dec: dict,
                              sim_log: pd.DataFrame) -> pd.DataFrame:
    """
    Per-station robotics metrics:
      1. detection_accuracy  — TP / (TP + FN) for high-pollution days
      2. avg_response_days   — lag from PM2.5 spike to ALERT/EMERGENCY
      3. station_coverage    — fraction of stations visited
      4. energy_efficiency   — mean PM2.5 intercepted per 1% battery used
    """
    print("\n" + "═" * 65)
    print("  PART NEW-C — ROBOTICS-SPECIFIC METRICS")
    print("═" * 65)
    os.makedirs(DIR_MET, exist_ok=True)
    rows = []

    if sim_log is not None and "robot_station" in sim_log.columns:
        global_coverage = round(
            sim_log["robot_station"].nunique() / max(len(results), 1), 4
        )
    else:
        global_coverage = float("nan")

    for sta in results:
        r   = results[sta]
        dec = all_dec[sta]

        # Detection accuracy
        actual_high = (r["actual"] > THR_MODERATE).astype(int)
        pred_alert  = dec["robot_state"].isin(["ALERT", "EMERGENCY"]).astype(int).values
        n  = min(len(actual_high), len(pred_alert))
        tp = int(np.sum((actual_high[:n] == 1) & (pred_alert[:n] == 1)))
        fn = int(np.sum((actual_high[:n] == 1) & (pred_alert[:n] == 0)))
        det_acc = round(tp / (tp + fn), 4) if (tp + fn) > 0 else float("nan")

        # Response time
        pm_arr = r["final_fc"]
        spikes = np.where(np.diff((pm_arr > THR_MODERATE).astype(int)) == 1)[0]
        resp_times = []
        states_arr = dec["robot_state"].values
        for sp in spikes:
            hits = np.where(np.isin(states_arr[sp:], ["ALERT", "EMERGENCY"]))[0]
            if hits.size > 0:
                resp_times.append(int(hits[0]))
        avg_rt = round(float(np.mean(resp_times)), 2) if resp_times else float("nan")

        # Energy efficiency
        if sim_log is not None and "battery" in sim_log.columns:
            sta_log   = sim_log[sim_log["station"] == sta]
            batt_used = max(0.1, 100.0 - sta_log["battery"].min())
            mean_pm   = sta_log["pm25_pred"].mean() if "pm25_pred" in sta_log.columns else float("nan")
            energy_eff = round(mean_pm / batt_used, 4)
        else:
            energy_eff = float("nan")

        rows.append({
            "station":            sta,
            "detection_accuracy": det_acc,
            "avg_response_days":  avg_rt,
            "station_coverage":   global_coverage,
            "energy_efficiency":  energy_eff,
            "true_positives":     tp,
            "false_negatives":    fn,
        })
        print(f"  {sta:<22}  det={det_acc:.3f}  resp={avg_rt}d  "
              f"cov={global_coverage:.2f}  eff={energy_eff:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIR_MET, "robotics_metrics.csv"), index=False)
    print(f"\n  ✓ Robotics metrics → {DIR_MET}/robotics_metrics.csv")
    return df


def save_robotics_outputs(sim_log: pd.DataFrame,
                           robot_sim: RobotMovementSimulator) -> None:
    """Save robot path CSV and energy log CSV."""
    print("\n  Saving robotics outputs ...")

    os.makedirs(DIR_MOV, exist_ok=True)
    if robot_sim is not None and robot_sim.path:
        path_df = pd.DataFrame(robot_sim.path, columns=["lat", "lon"])
        path_df.index.name = "step"
        path_df.to_csv(os.path.join(DIR_MOV, "robot_path.csv"))
        print(f"  ✓ Robot path  → {DIR_MOV}/robot_path.csv  ({len(path_df)} positions)")

    os.makedirs(DIR_ENRG, exist_ok=True)
    if sim_log is not None and "battery" in sim_log.columns:
        cols = [c for c in
                ["date", "station", "battery", "robot_state",
                 "returning_base", "robot_lat", "robot_lon"]
                if c in sim_log.columns]
        energy_df = sim_log[cols].drop_duplicates(subset=["date"])
        energy_df.to_csv(os.path.join(DIR_ENRG, "energy_log.csv"), index=False)
        batt = sim_log["battery"]
        print(f"  ✓ Energy log  → {DIR_ENRG}/energy_log.csv  ({len(energy_df)} rows)")
        print(f"     Battery min={batt.min():.1f}%  mean={batt.mean():.1f}%  max={batt.max():.1f}%")
