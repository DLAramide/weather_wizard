from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from utils.daily_nasa_data import fetch_nasa_data
from utils.monthly_nasa_data import fetch_nasa_monthly_data
app = FastAPI(title="Rainfall Prediction API")

# ------------------------
# Load model and scaler
# ------------------------
model = load_model("model/rainfall_predictor.h5", compile=False)
scaler = joblib.load("model/scaler.pkl")
model_monthly = load_model("model/rainfall_monthly_predictor.h5", compile = False)
scaler_monthly = joblib.load("model/scaler_monthly.pkl")

FEATURES = [
    'RH2M', 'WS10M', 'T2M', 'WD10M',
    'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PS',
    'QV2M', 'T2M_RANGE', 'TS', 'CLRSKY_SFC_SW_DWN'
]

TARGET_COL = "PRECTOTCORR"

# ------------------------
# Helper function for inverse
# ------------------------
def inverse_transform_prediction(pred_scaled, scaler, last_scaled_row):
    """
    Convert scaled log1p prediction back to original units safely
    """
    row = last_scaled_row.copy()
    row[0, -1] = pred_scaled  # replace target only
    inv_scaled = scaler.inverse_transform(row)[0, -1]
    return np.expm1(inv_scaled)  # inverse of log1p

# ------------------------
# Prediction endpoint
# ------------------------
@app.get("/predict_daily")
def predict_rainfall(days: int = 1, latitude: float = 6.585, longitude: float = 3.983):
    """
    Predict rainfall for 'days' ahead using NASA POWER data.
    Example: /predict?days=3
    """

    # 1️⃣ Fetch recent NASA weather data
    df = fetch_nasa_data(latitude, longitude, days=120)

    # 2️⃣ Ensure all required features exist
    for col in FEATURES + [TARGET_COL]:
        if col not in df.columns:
            return {"error": f"Missing feature: {col}"}

    # 3️⃣ Fill missing, clip negatives
    for col in FEATURES + [TARGET_COL]:
        df[col] = df[col].fillna(0)
        df[col] = df[col].clip(lower=0)

    # 4️⃣ Log-transform safely
    df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
    for col in FEATURES:
        df[f"log_{col}"] = np.log1p(df[col])

    # 5️⃣ Lag and rolling features
    for lag in [1, 3, 7]:
        df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)
    df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=7).mean()
    df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=7).std()
    df = df.dropna().copy()

    # 6️⃣ Check if enough data
    if len(df) < 15:
        return {"error": "Not enough data to make prediction."}

    # 7️⃣ Prepare features
    final_features = [f"log_{col}" for col in FEATURES] + [
        "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
        "rain_rolling_mean", "rain_rolling_std"
    ]

    # 8️⃣ Start with last 15 days
    window = df[final_features + ["log_PRECTOTCORR"]].tail(15).copy()
    forecasts = []

    for day in range(1, days + 1):
        # Scale window
        scaled = scaler.transform(window)
        X_input = np.expand_dims(scaled, axis=0)

        # Predict
        pred_scaled = model.predict(X_input)
        rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, scaled[-1:].reshape(1, -1))

        # Clip negative predictions
        rainfall_mm = max(0, rainfall_mm)

        forecasts.append({
            "day": day,
            "predicted_rainfall_mm": round(float(rainfall_mm), 3)
        })

        # --- Prepare new row for next prediction ---
        new_row = {}
        # Carry forward features
        for col in final_features:
            if col != "log_PRECTOTCORR":
                new_row[col] = window[col].iloc[-1]

        # Update lag features dynamically
        lag_values = [window["log_PRECTOTCORR"].iloc[-lag] for lag in [1, 3, 7]]
        new_row["log_PRECTOTCORR_lag1"] = lag_values[0]
        new_row["log_PRECTOTCORR_lag3"] = lag_values[1]
        new_row["log_PRECTOTCORR_lag7"] = lag_values[2]

        # Update rolling features
        rolling_window = list(window["log_PRECTOTCORR"].iloc[-6:]) + [np.log1p(rainfall_mm)]
        new_row["rain_rolling_mean"] = np.mean(rolling_window)
        new_row["rain_rolling_std"] = np.std(rolling_window)

        # Add predicted log rainfall itself
        new_row["log_PRECTOTCORR"] = np.log1p(rainfall_mm)

        # Append new row and keep last 15 days
        window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True)
        window = window.tail(15)

    return {
        "forecasts": forecasts,
        "location": {"latitude": latitude, "longitude": longitude}
    }


@app.get("/predict_monthly")
def predict_monthly(months: int = 1, latitude: float = 6.585, longitude: float = 3.983):
    """
    Predict rainfall for 'months' ahead using NASA POWER monthly data.
    Example: /predict_monthly?months=3
    """

    # 1️⃣ Fetch recent NASA monthly weather data
    df = fetch_nasa_monthly_data(latitude, longitude)

    # 2️⃣ Basic cleanup — remove annual rows, map months, build date
    df = df[df['MONTH'] != 'ANN'].copy()
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    df['MONTH_NUM'] = df['MONTH'].map(month_map)
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH_NUM'].astype(str) + '-01')
    df = df.sort_values('DATE').reset_index(drop=True)

    # 3️⃣ Ensure all required features exist
    for col in FEATURES + [TARGET_COL]:
        if col not in df.columns:
            df[col] = 0

    # 4️⃣ Fill missing, clip negatives
    for col in FEATURES + [TARGET_COL]:
        df[col] = df[col].fillna(0)
        df[col] = df[col].clip(lower=0)

    # 5️⃣ Log-transform
    df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
    for col in FEATURES:
        df[f"log_{col}"] = np.log1p(df[col])

    # 6️⃣ Lag and rolling features (for monthly)
    for lag in [1, 2, 3]:
        df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)
    df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=3).mean()
    df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=3).std()
    df = df.dropna().copy()

    # 7️⃣ Check data sufficiency
    if len(df) < 6:
        return {"error": "Not enough data to make monthly prediction."}

    # 8️⃣ Final feature list
    final_features = [f"log_{col}" for col in FEATURES] + [
        "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag2", "log_PRECTOTCORR_lag3",
        "rain_rolling_mean", "rain_rolling_std"
    ]

    # 9️⃣ Use last 6 months as window
    window = df[final_features + ["log_PRECTOTCORR"]].tail(6).copy()
    forecasts = []

    for month in range(1, months + 1):
        scaled = scaler_monthly.transform(window)
        X_input = np.expand_dims(scaled, axis=0)

        # Predict rainfall
        pred_scaled = model_monthly.predict(X_input)
        rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, scaled[-1:].reshape(1, -1))
        rainfall_mm = max(0, rainfall_mm)

        forecasts.append({
            "month_ahead": month,
            "predicted_rainfall_mm": round(float(rainfall_mm), 3)
        })

        # --- Update for next iteration ---
        new_row = {}
        for col in final_features:
            if col != "log_PRECTOTCORR":
                new_row[col] = window[col].iloc[-1]

        # Update lags
        lag_values = [window["log_PRECTOTCORR"].iloc[-lag] for lag in [1, 2, 3]]
        new_row["log_PRECTOTCORR_lag1"] = lag_values[0]
        new_row["log_PRECTOTCORR_lag2"] = lag_values[1]
        new_row["log_PRECTOTCORR_lag3"] = lag_values[2]

        # Rolling updates
        rolling_window = list(window["log_PRECTOTCORR"].iloc[-2:]) + [np.log1p(rainfall_mm)]
        new_row["rain_rolling_mean"] = np.mean(rolling_window)
        new_row["rain_rolling_std"] = np.std(rolling_window)

        # Add new prediction
        new_row["log_PRECTOTCORR"] = np.log1p(rainfall_mm)

        window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True)
        window = window.tail(6)

    return {
        "monthly_forecasts": forecasts,
        "location": {"latitude": latitude, "longitude": longitude}
    }
