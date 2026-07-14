import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from config.settings import (
    SEED, LOOKBACK, TEST_FRAC, VAL_FRAC,
    LSTM_UNITS, LSTM_DROPOUT, LSTM_LR, LSTM_LOSS,
    LSTM_EPOCHS, LSTM_BATCH,
    ES_PATIENCE, ES_MIN_DELTA,
    PLOT_STYLE, DIR_TRAIN, DIR_FIG,
    STATION_COLORS, PAL,
)
from utils.helpers import set_all_seeds, savefig, compute_metrics


def build_lstm(lookback, n_features):
    model = Sequential()
    model.add(LSTM(LSTM_UNITS, dropout=LSTM_DROPOUT, return_sequences=True,
                   input_shape=(lookback, n_features)))
    model.add(LSTM(LSTM_UNITS, dropout=LSTM_DROPOUT))
    model.add(Dense(1))
    model.compile(loss=LSTM_LOSS, optimizer=Adam(learning_rate=LSTM_LR))
    return model


def make_dataset(data, duration):
    X, y = [], []
    for i in range(len(data) - duration):
        X.append(data[i: i + duration])
        y.append(data[i + duration])
    X = np.array(X).reshape(len(X), duration, 1)
    y = np.array(y).reshape(len(y), 1)
    return X, y


def fit_lstm(hourly_data):
    print("\n" + "\u2550" * 65)
    print("  LSTM MODEL  |  pm2.5.py implementation")
    print("\u2550" * 65)

    results = {}
    for sta, df in hourly_data.items():
        print(f"\n  [{sta}]")

        arr = df.values
        n_features = arr.shape[1]
        n_total = len(arr)
        n_train = max(1, int(n_total * (1 - TEST_FRAC)))

        scaler = StandardScaler()
        scaler.fit(arr[:n_train])
        scaled = scaler.transform(arr)
        scaled_df = pd.DataFrame(scaled)

        x = None; y = None; is_first = True
        for i in range(n_features):
            xx, yy = make_dataset(scaled_df.iloc[:, i], LOOKBACK)
            if is_first:
                x, y = xx, yy; is_first = False
            else:
                x = np.concatenate([x, xx], axis=2)

        total_size = len(scaled_df) - LOOKBACK
        train_size = int(total_size * (1 - TEST_FRAC))

        x_train, y_train = x[:train_size], y[:train_size]
        x_test_rest, y_test_rest = x[train_size - 1:], y[train_size - 1:]

        val_split = 0.5
        test_size = int((len(x_test_rest) - LOOKBACK) * val_split)
        x_valid, y_valid = x_test_rest[:test_size], y_test_rest[:test_size]
        x_test, y_test = x_test_rest[test_size - 1:], y_test_rest[test_size - 1:]

        print(f"    Train={len(x_train)}  Val={len(x_valid)}  Test={len(x_test)}  "
              f"Features={n_features}")

        if len(x_train) < 10 or len(x_test) < 10:
            print(f"    \u2717 Insufficient data")
            continue

        set_all_seeds(SEED)
        model = build_lstm(LOOKBACK, n_features)
        model.summary()

        cbs = [EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                             min_delta=ES_MIN_DELTA, restore_best_weights=True)]
        hist = model.fit(
            x_train, y_train,
            validation_data=(x_valid, y_valid),
            epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
            callbacks=cbs, verbose=1, shuffle=False,
        )

        test_predictions = model.predict(x_test, verbose=0)

        dummy = np.zeros((len(test_predictions), n_features))
        dummy[:, 0] = test_predictions.flatten()
        y_pred = scaler.inverse_transform(dummy)[:, 0]
        dummy[:, 0] = y_test.flatten()
        y_actual = scaler.inverse_transform(dummy)[:, 0]
        y_pred = np.clip(y_pred, 0, None)

        test_start_idx = LOOKBACK + train_size + test_size - 2
        test_index = df.index[test_start_idx:test_start_idx + len(y_actual)]

        metrics = compute_metrics(y_actual, y_pred, label=f"{sta} LSTM")

        results[sta] = {
            "actual": y_actual,
            "forecast": y_pred,
            "predicted": y_pred,
            "test_index": test_index,
            "metrics": metrics,
        }

        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hist.history["loss"], color="#E63946", lw=2, label="Train Loss")
            ax.plot(hist.history["val_loss"], color="#457B9D", lw=2, label="Val Loss")
            ax.set_title(f"LSTM Training History \u2014 {sta}", fontweight="bold")
            ax.set_yscale("log")
            ax.legend()
            plt.tight_layout()
            savefig(fig, f"lstm_history_{sta.replace(' ', '_')}.png", DIR_TRAIN)

        with plt.style.context(PLOT_STYLE):
            fig, axes = plt.subplots(2, 1, figsize=(16, 8))
            ax = axes[0]
            ax.plot(test_index, y_actual, color="#264653", lw=0.8, label="Test actual")
            ax.plot(test_index, y_pred, color="#E9C46A", lw=1.2, ls="-.", label="Test forecast")
            ax.set_title(f"LSTM \u2014 {sta}", fontweight="bold")
            ax.legend(fontsize=8)
            ax1 = axes[1]
            err = y_actual - y_pred
            ax1.hist(err, bins=50, color="#8338EC", edgecolor="white",
                     lw=0.4, density=True, alpha=0.8)
            ax1.set_title(f"Error Distribution  MAE={metrics['MAE']:.2f}  "
                          f"RMSE={metrics['RMSE']:.2f}  "
                          f"R\u00b2={metrics['R2']:.3f}",
                          fontweight="bold")
            plt.tight_layout()
            savefig(fig, f"lstm_pred_{sta.replace(' ', '_')}.png", DIR_TRAIN)

    return results


def fit_hybrid(hourly_data, arimax_results):
    print("\n" + "\u2550" * 65)
    print("  HYBRID ARIMA-LSTM  |  pm2.5.py implementation")
    print("\u2550" * 65)

    results = {}
    for sta, df in hourly_data.items():
        if sta not in arimax_results:
            print(f"  Skipping {sta} \u2014 no ARIMAX results")
            continue

        print(f"\n  [{sta}]")

        ar = arimax_results[sta]
        arr = df.values
        n_features = arr.shape[1]
        n_total = len(arr)
        n_train = max(1, int(n_total * (1 - TEST_FRAC)))

        scaler = StandardScaler()
        scaler.fit(arr[:n_train])
        scaled = scaler.transform(arr)

        scaled_arimax = pd.DataFrame(scaled, columns=df.columns).copy()
        scaled_arimax_arr = scaled_arimax.values

        y_scaled_arimax = scaled_arimax_arr[:, 0]
        X_scaled_arimax = pd.DataFrame(scaled_arimax_arr[:, 1:])
        drop_idx = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        drop_idx = [c for c in drop_idx if c < X_scaled_arimax.shape[1]]
        X_sel = X_scaled_arimax.drop(X_scaled_arimax.columns[drop_idx], axis=1)

        y_train_scaled = y_scaled_arimax[:n_train]
        X_train_sel = X_sel.iloc[:n_train]
        try:
            arima_model = ARIMA(endog=y_train_scaled, exog=X_train_sel, order=[1, 0, 1]).fit()
            predict_train_arr = arima_model.predict(exog=X_train_sel, dynamic=True).values.flatten()
            arima_residuals = y_train_scaled - predict_train_arr

            X_test_sel = X_sel.iloc[n_train - 1:]
            predict_test_arr = arima_model.predict(
                start=len(predict_train_arr) - 1,
                end=len(predict_train_arr) - 1 + len(X_test_sel[1:]),
                exog=X_test_sel[1:], dynamic=True
            ).values.flatten()
        except Exception as e:
            print(f"    ARIMAX recompute failed: {e}")
            continue

        scaled_df = pd.DataFrame(scaled)
        x_hybrid = None; is_first = True
        for i in range(n_features):
            xx, _ = make_dataset(scaled_df.iloc[:, i], LOOKBACK)
            if is_first:
                x_hybrid = xx; is_first = False
            else:
                x_hybrid = np.concatenate([x_hybrid, xx], axis=2)

        num_residual = n_train - 1
        x_res = x_hybrid[:num_residual]
        y_res = arima_residuals[1:].reshape(-1, 1)

        res_total = len(x_res)
        train_r = int(res_total * (1 - TEST_FRAC))
        x_r_train = x_res[:train_r]
        y_r_train = y_res[:train_r]
        x_r_valid_test = x_res[train_r - 1:]
        y_r_valid_test = y_res[train_r - 1:]
        test_r = int((len(x_r_valid_test) - LOOKBACK) * 0.5)
        x_r_valid = x_r_valid_test[:test_r]
        y_r_valid = y_r_valid_test[:test_r]

        print(f"    Residual LSTM: Train={len(x_r_train)}  Val={len(x_r_valid)}  "
              f"Test={len(x_r_valid_test) - test_r}")

        if len(x_r_train) < 10:
            print(f"    \u2717 Insufficient training data")
            continue

        set_all_seeds(SEED)
        model = build_lstm(LOOKBACK, n_features)

        cbs = [EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                             min_delta=ES_MIN_DELTA, restore_best_weights=True)]
        model.fit(
            x_r_train, y_r_train,
            validation_data=(x_r_valid, y_r_valid),
            epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
            callbacks=cbs, verbose=1, shuffle=False,
        )

        res_pred_train_full = model.predict(x_hybrid[:n_train - 1], verbose=0)
        res_pred_test_full = model.predict(x_hybrid[n_train - 1:], verbose=0)

        hybrid_pred_scaled = np.zeros(len(scaled_df))
        hybrid_pred_scaled[0] = y_scaled_arimax[0]
        hybrid_pred_scaled[1:n_train] = predict_train_arr[1:] + res_pred_train_full.flatten()
        hybrid_pred_scaled[n_train:] = predict_test_arr[1:] + res_pred_test_full.flatten()

        dummy_pred = np.zeros((len(hybrid_pred_scaled[n_train:]), n_features))
        dummy_pred[:, 0] = hybrid_pred_scaled[n_train:]
        hybrid_test_inv = scaler.inverse_transform(dummy_pred)[:, 0]
        dummy_act = np.zeros((len(y_scaled_arimax[n_train:]), n_features))
        dummy_act[:, 0] = y_scaled_arimax[n_train:]
        actual_test_inv = scaler.inverse_transform(dummy_act)[:, 0]
        hybrid_test_inv = np.clip(hybrid_test_inv, 0, None)

        n_hybrid = min(len(ar["forecast"]), len(hybrid_test_inv))
        hybrid_final = hybrid_test_inv[:n_hybrid]
        actual_final = actual_test_inv[:n_hybrid]
        arimax_final = ar["forecast"][:n_hybrid]
        test_idx = ar["test_index"][:n_hybrid]

        metrics = compute_metrics(actual_final, hybrid_final,
                                   label=f"{sta} Hybrid ARIMA-LSTM")

        results[sta] = {
            "actual": actual_final,
            "final_fc": hybrid_final,
            "arimax_pred": arimax_final,
            "lstm_correction": hybrid_final - arimax_final,
            "test_index": test_idx,
            "metrics": metrics,
        }

        with plt.style.context(PLOT_STYLE):
            fig, axes = plt.subplots(2, 1, figsize=(16, 8))
            ax = axes[0]
            ax.plot(test_idx, actual_final, color="#264653", lw=0.8, label="Test actual")
            ax.plot(test_idx, hybrid_final, color="#E76F51", lw=1.2, ls="-.", label="Hybrid forecast")
            ax.set_title(f"Hybrid ARIMA-LSTM \u2014 {sta}", fontweight="bold")
            ax.legend(fontsize=8)
            ax1 = axes[1]
            err = actual_final - hybrid_final
            ax1.hist(err, bins=50, color="#8338EC", edgecolor="white",
                     lw=0.4, density=True, alpha=0.8)
            ax1.set_title(f"Error Distribution  MAE={metrics['MAE']:.2f}  "
                          f"RMSE={metrics['RMSE']:.2f}  "
                          f"R\u00b2={metrics['R2']:.3f}",
                          fontweight="bold")
            plt.tight_layout()
            savefig(fig, f"hybrid_pred_{sta.replace(' ', '_')}.png", DIR_FIG)

    return results
