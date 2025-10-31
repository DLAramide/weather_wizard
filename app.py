from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from utils.nasa_data import fetch_nasa_data

app = FastAPI(title="Rainfall Prediction API")

# Load model and scaler
model = load_model("model/rainfall_predictor.h5", compile=False)
scaler = joblib.load("model/scaler.pkl")

FEATURES = [
    'RH2M', 'WS10M', 'T2M', 'WD10M',
    'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PS',
    'QV2M', 'T2M_RANGE', 'TS', 'CLRSKY_SFC_SW_DWN'
]

@app.get("/predict")
def predict_rainfall(days: int = 1):
    """
    Predict rainfall for 'days' ahead using NASA POWER data.
    Example: /predict?days=3
    """

    # 1️⃣ Fetch recent NASA weather data
    df = fetch_nasa_data(days=120)

    # 2️⃣ Ensure all required features exist
    for col in FEATURES:
        if col not in df.columns:
            return {"error": f"Missing feature: {col}"}

    df["PRECTOTCORR"] = df["PRECTOTCORR"].ffill()

    # 3️⃣ Log-transform
    df["log_PRECTOTCORR"] = np.log1p(df["PRECTOTCORR"])
    for col in FEATURES:
        df[f"log_{col}"] = np.log1p(df[col])

    # 4️⃣ Lag and rolling features
    for lag in [1, 3, 7]:
        df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)
    df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=7).mean()
    df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=7).std()

    df = df.dropna().copy()

    # 5️⃣ Check if enough data
    if len(df) < 15:
        return {"error": "Not enough data to make prediction."}

    # 6️⃣ Prepare features
    final_features = [f"log_{col}" for col in FEATURES] + [
        "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
        "rain_rolling_mean", "rain_rolling_std"
    ]

    # Start with last 15 days
    window = df[final_features + ["log_PRECTOTCORR"]].tail(15).copy()
    forecasts = []

    for day in range(1, days + 1):
        # Scale and reshape
        scaled = scaler.transform(window)
        X_input = np.expand_dims(scaled, axis=0)

        # Predict
        pred_scaled = model.predict(X_input)
        dummy = np.zeros((1, len(final_features) + 1))
        dummy[0, -1] = pred_scaled
        inv = scaler.inverse_transform(dummy)[0, -1]
        rainfall_mm = np.expm1(inv)

        forecasts.append({
            "day": day,
            "predicted_rainfall_mm": round(float(rainfall_mm), 3)
        })

        # Prepare new row for next step
        new_row = {}

        # Keep other features the same (or fetch new NASA values if available)
        for col in final_features:
            if col.startswith("log_") and col != "log_PRECTOTCORR":
                # just carry forward the last known value
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

        # Append new row and keep only last 15 days
        window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True)
        window = window.tail(15)

    return {
        "forecasts": forecasts,
        "location": "Epe, Nigeria"
    }
