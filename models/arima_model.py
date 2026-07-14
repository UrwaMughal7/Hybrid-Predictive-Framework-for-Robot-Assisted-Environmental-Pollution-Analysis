import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from config.settings import (
    SEED, TEST_FRAC, SEASONAL_PERIOD, ARIMA_ORDER,
    STATION_COLORS, PLOT_STYLE, DIR_ARIMA,
)
from utils.helpers import set_all_seeds, savefig, compute_metrics


def _remove_seasonality(df, period):
    out = df.copy()
    for col in df.columns:
        try:
            decomp = seasonal_decompose(df[col], model="additive", period=period)
            out[col] = df[col] - decomp.seasonal
        except Exception:
            pass
    return out


def _fit_arimax_station(sta, df, order):
    n_total = len(df)
    n_train = max(1, int(n_total * (1 - TEST_FRAC)))

    scaler = StandardScaler()
    scaler.fit(df.iloc[:n_train])
    scaled = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

    deseas = _remove_seasonality(scaled, SEASONAL_PERIOD)

    try:
        adf_result = adfuller(deseas.iloc[:, 0].dropna(), autolag="AIC", maxlag=1)
        print(f"    ADF: stat={adf_result[0]:.4f}  p={adf_result[1]:.6f}  " +
              ("\u2713 STATIONARY" if adf_result[1] < 0.05 else "\u2717 NON-STATIONARY"))
    except Exception:
        print("    ADF test skipped")

    deseas = deseas.dropna()
    y_all = deseas.iloc[:, 0]
    X_all = deseas.iloc[:, 1:]
    y_train = y_all[:n_train]
    X_train_all = X_all.iloc[:n_train]

    ARIMA(endog=y_train, exog=X_train_all, order=order).fit()

    drop_idx = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    drop_idx = [c for c in drop_idx if c < X_all.shape[1]]
    X_selected = X_all.drop(X_all.columns[drop_idx], axis=1)

    X_train_sel = X_selected.iloc[:n_train]
    X_test_sel = X_selected.iloc[n_train - 1:]

    final_model = ARIMA(endog=y_train, exog=X_train_sel, order=order).fit()
    print(f"    ARIMA{tuple(order)}  features={X_train_sel.shape[1]}  AIC={final_model.aic:.2f}")

    predict_train = final_model.predict(exog=X_train_sel, dynamic=True)
    predict_test = final_model.predict(
        start=len(predict_train) - 1,
        end=len(predict_train) - 1 + len(X_test_sel[1:]),
        exog=X_test_sel[1:], dynamic=True
    )

    pm25_scaler = StandardScaler()
    pm25_scaler.fit(df.iloc[:n_train, 0].values.reshape(-1, 1))

    train_pred_inv = pm25_scaler.inverse_transform(predict_train.values.reshape(-1, 1)).flatten()
    test_pred_inv = pm25_scaler.inverse_transform(predict_test.values.reshape(-1, 1)).flatten()
    y_train_inv = pm25_scaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_actual = y_all.iloc[n_train - 1:]
    y_test_inv = pm25_scaler.inverse_transform(y_test_actual.values.reshape(-1, 1)).flatten()
    test_pred_inv = np.clip(test_pred_inv, 0, None)

    n_align = min(len(test_pred_inv), len(y_test_inv))
    aligned_pred = test_pred_inv[:n_align]
    aligned_actual = y_test_inv[:n_align]
    aligned_idx = df.index[n_train - 1:n_train - 1 + n_align]

    return {
        "actual": aligned_actual,
        "forecast": aligned_pred,
        "predicted": aligned_pred,
        "test_index": aligned_idx,
        "train_actual": y_train_inv,
        "train_forecast": train_pred_inv,
        "residuals": y_train_inv - train_pred_inv,
    }, final_model, df.index, y_train_inv, train_pred_inv


def fit_arimax(hourly_data):
    print("\n" + "\u2550" * 65)
    print("  ARIMAX MODEL  |  pm2.5.py implementation")
    print("\u2550" * 65)

    results = {}
    for sta, df in hourly_data.items():
        print(f"\n  [{sta}]")
        try:
            res, model, full_idx, y_tr, pred_tr = \
                _fit_arimax_station(sta, df, ARIMA_ORDER)

            metrics = compute_metrics(res["actual"], res["forecast"],
                                      label=f"{sta} ARIMAX")
            res["metrics"] = metrics

            results[sta] = res

            with plt.style.context(PLOT_STYLE):
                fig, axes = plt.subplots(2, 1, figsize=(16, 8))
                ax = axes[0]
                ax.plot(full_idx[:len(y_tr)], y_tr,
                        color="#264653", lw=0.4, alpha=0.7, label="Train actual")
                ax.plot(full_idx[:len(pred_tr)], pred_tr,
                        color="#E76F51", lw=0.6, ls="--", label="Train fitted")
                ax.plot(res["test_index"], res["actual"],
                        color="#2A9D8F", lw=0.8, label="Test actual")
                ax.plot(res["test_index"], res["forecast"],
                        color="#E9C46A", lw=1.2, ls="-.", label="Test forecast")
                ax.set_title(f"ARIMAX \u2014 {sta}", fontweight="bold")
                ax.legend(fontsize=8)

                ax1 = axes[1]
                err = res["actual"] - res["forecast"]
                ax1.hist(err, bins=50, color="#8338EC", edgecolor="white",
                         lw=0.4, density=True, alpha=0.8)
                ax1.set_title(f"Error Distribution  MAE={metrics['MAE']:.2f}  "
                              f"RMSE={metrics['RMSE']:.2f}  "
                              f"R\u00b2={metrics['R2']:.3f}",
                              fontweight="bold")
                plt.tight_layout()
                savefig(fig, f"arimax_{sta.replace(' ', '_')}.png", DIR_ARIMA)

        except Exception as e:
            print(f"    \u2717 FAILED: {e}")

    return results
