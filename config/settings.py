"""
config/settings.py
==================
Central configuration for the ARIMAX + LSTM + Hybrid ARIMA-LSTM pipeline.
Edit DATA_DIR and RESULTS_DIR to match your machine.
"""

import os

# ── Reproducibility ────────────────────────────────────────────────────────────
os.environ["PYTHONHASHSEED"]         = "42"
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"

SEED = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "PRSA_Data_20130301-20170228")
)
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)

# ── Output sub-directories ─────────────────────────────────────────────────────
DIR_EDA   = os.path.join(RESULTS_DIR, "eda")
DIR_ARIMA = os.path.join(RESULTS_DIR, "arima")
DIR_TRAIN = os.path.join(RESULTS_DIR, "training")
DIR_PRED  = os.path.join(RESULTS_DIR, "predictions")
DIR_MET   = os.path.join(RESULTS_DIR, "metrics")
DIR_FIG   = os.path.join(RESULTS_DIR, "figures")
DIR_BASE  = os.path.join(RESULTS_DIR, "baseline")
DIR_ROBOT = os.path.join(RESULTS_DIR, "robot_decisions")
DIR_VID   = os.path.join(RESULTS_DIR, "videos")
DIR_MOV   = os.path.join(RESULTS_DIR, "movement")
DIR_ENRG  = os.path.join(RESULTS_DIR, "energy")

ALL_DIRS = [
    RESULTS_DIR, DIR_EDA, DIR_ARIMA, DIR_TRAIN,
    DIR_PRED, DIR_MET, DIR_FIG, DIR_BASE,
    DIR_ROBOT, DIR_VID, DIR_MOV, DIR_ENRG,
]

# ── Data parameters (hourly, matching pm2.5.py) ───────────────────────────────
LOOKBACK      = 1
TEST_FRAC     = 0.2
VAL_FRAC      = 0.1        # validation split from training portion (LSTM)

SEASONAL_PERIOD = int(365 * 24 / 2)  # 4380 hours — matches pm2.5.py

# ── ARIMAX parameters ─────────────────────────────────────────────────────────
ARIMA_ORDER    = [1, 0, 1]
ARIMA_PVALUE_THR = 0.05
ARIMA_MIN_FEATURES = 3

# ── LSTM parameters ────────────────────────────────────────────────────────────
LSTM_UNITS    = 16
LSTM_DROPOUT  = 0.2
LSTM_LR       = 1e-3
LSTM_LOSS     = "mean_squared_error"
LSTM_EPOCHS   = 50
LSTM_BATCH    = 128
ES_PATIENCE   = 10
ES_MIN_DELTA  = 1e-5

# ── Numeric sensor columns (from original CSV before one-hot encoding wd) ──────
NUMERIC_SENSORS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                   "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# ── Robot parameters ──────────────────────────────────────────────────────────
MOVE_SPEED_DEG      = 0.05
DIST_COST           = 0.8
TREND_ESCALATE_RATE = 10.0
BATTERY_MOVE_COST   = 2.0
BATTERY_SAMPLE_COST = 1.0
BATTERY_IDLE_CHARGE = 0.5
BATTERY_LOW         = 20.0
BASE_STATION        = "Tiantan"

# ── PM2.5 thresholds (WHO-based, µg/m³ daily mean) ────────────────────────────
THR_SAFE      = 35
THR_MODERATE  = 75
THR_HAZARDOUS = 150

SENSOR_CASCADE = {
    "SAFE":      [],
    "MODERATE":  ["PM10", "WSPM"],
    "UNHEALTHY": ["PM10", "SO2", "NO2", "TEMP"],
    "HAZARDOUS": ["PM10", "SO2", "NO2", "CO", "O3", "WSPM",
                  "TEMP", "PRES", "DEWP", "RAIN"],
}

# ── Colour palettes ───────────────────────────────────────────────────────────
STATION_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653",
    "#8338EC", "#FB5607", "#3A86FF", "#FFBE0B", "#06D6A0", "#EF476F",
]
PAL = {
    "primary":   "#2563EB",
    "secondary": "#7C3AED",
    "success":   "#16A34A",
    "warning":   "#D97706",
    "danger":    "#DC2626",
    "dark":      "#1E293B",
    "muted":     "#64748B",
}
CAT_COLORS = {
    "SAFE":      "#16A34A",
    "MODERATE":  "#D97706",
    "UNHEALTHY": "#DC2626",
    "HAZARDOUS": "#7C3AED",
}
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# ── Station GPS coordinates (Beijing, approximate) ────────────────────────────
STATION_COORDS = {
    "Aotizhongxin": (39.993, 116.407),
    "Changping":    (40.221, 116.231),
    "Dingling":     (40.293, 116.228),
    "Dongsi":       (39.952, 116.427),
    "Guanyuan":     (39.957, 116.363),
    "Gucheng":      (39.935, 116.184),
    "Huairou":      (40.316, 116.632),
    "Nongzhanguan": (39.963, 116.478),
    "Shunyi":       (40.137, 116.654),
    "Tiantan":      (39.882, 116.418),
    "Wanliu":       (39.972, 116.290),
    "Wanshouxigong":(39.878, 116.356),
}
