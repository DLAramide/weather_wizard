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
def inverse_transform_prediction(pred_scaled, scaler, last_row_scaled):
    """
    Reverse MinMax scaling and log1p transformation for the target.
    pred_scaled: output of model in scaled space
    scaler: fitted MinMaxScaler used on the features (including log_PRECTOTCORR)
    last_row_scaled: the last row of input scaled data (needed to preserve scaling shape)
    """
    # Copy last row and replace target column with prediction
    temp = last_row_scaled.copy()
    temp[0, -1] = pred_scaled  # Assuming target is last column

    # Inverse MinMax scaling
    temp_inv = scaler.inverse_transform(temp)
    pred_log = temp_inv[0, -1]  # Extract the target (log-transformed) column

    # Reverse log1p
    rainfall_mm = np.expm1(pred_log)
    return rainfall_mm


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
    # --- Step 1: Fetch NASA monthly data ---
    df = fetch_nasa_monthly_data(latitude, longitude)
    
    # Replace NASA missing flags and forward/backward fill
    df = df.replace(-999.0, np.nan).ffill().bfill()
    
    # Reset index for convenience
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'DATE'}, inplace=True)
    
    # Ensure all required features exist
    for col in FEATURES + [TARGET_COL]:
        if col not in df.columns:
            df[col] = 0

    # --- Step 2: Log-transform features ---
    df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
    for col in FEATURES:
        df[f"log_{col}"] = np.log1p(df[col])

    # --- Step 3: Create lag features ---
    for lag in [1, 3, 7]:
        df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)

    # --- Step 4: Rolling features ---
    df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=3).mean()
    df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=3).std()
    
    # Drop rows with NaNs due to lags/rolling
    df = df.dropna().copy()
    
    if len(df) < 7:
        return {"error": "Not enough data to compute lag7 for monthly prediction."}

    # --- Step 5: Prepare prediction window ---
    final_features = [f"log_{col}" for col in FEATURES] + [
        "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
        "rain_rolling_mean", "rain_rolling_std"
    ]
    window = df[final_features + ["log_PRECTOTCORR"]].tail(7).copy()
    
    forecasts = []
    for month in range(1, months + 1):
        # Scale features
        scaled = scaler_monthly.transform(window)
        X_input = np.expand_dims(scaled, axis=0)
        
        # Predict (scaled/log space)
        pred_scaled = model_monthly.predict(X_input)
        
        # --- Step 6: Inverse transform to mm ---
        rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, scaled[-1:].reshape(1, -1))
        rainfall_mm = max(0, rainfall_mm)

        forecasts.append({
            "month_ahead": month,
            "predicted_rainfall_mm": round(float(rainfall_mm), 3)
        })

        # --- Step 7: Update window for next month ---
        new_row = {col: window[col].iloc[-1] for col in final_features if col != "log_PRECTOTCORR"}
        lag_values = [window["log_PRECTOTCORR"].iloc[-lag] for lag in [1,3,7]]
        new_row.update({
            "log_PRECTOTCORR_lag1": lag_values[0],
            "log_PRECTOTCORR_lag3": lag_values[1],
            "log_PRECTOTCORR_lag7": lag_values[2],
            "rain_rolling_mean": np.mean(list(window["log_PRECTOTCORR"].iloc[-2:]) + [np.log1p(rainfall_mm)]),
            "rain_rolling_std": np.std(list(window["log_PRECTOTCORR"].iloc[-2:]) + [np.log1p(rainfall_mm)]),
            "log_PRECTOTCORR": np.log1p(rainfall_mm)
        })

        window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True).tail(7)

    return {"monthly_forecasts": forecasts, "location": {"latitude": latitude, "longitude": longitude}}

@app.get("/debug_prediction")
def debug_prediction(latitude: float = 6.585, longitude: float = 3.983):
    # 1️⃣ Fetch NASA monthly data
    df = fetch_nasa_monthly_data(latitude, longitude)
    df = df.replace(-999.0, np.nan).ffill().bfill()
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'DATE'}, inplace=True)

    # 2️⃣ Ensure all required features exist
    for col in FEATURES + [TARGET_COL]:
        if col not in df.columns:
            df[col] = 0

    # 3️⃣ Log-transform features
    df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
    for col in FEATURES:
        df[f"log_{col}"] = np.log1p(df[col])

    # 4️⃣ Create lag features (1,3,7)
    for lag in [1, 3, 7]:
        df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)

    # 5️⃣ Rolling features
    df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=3).mean()
    df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=3).std()

    # 6️⃣ Drop rows with NaNs from lag/rolling features
    df = df.dropna().copy()
    if len(df) < 7:
        return {"error": "Not enough data to compute lag7 for debug prediction."}

    # 7️⃣ Prepare feature set exactly matching scaler
    final_features_scaler = [f"log_{col}" for col in FEATURES] + [
        "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
        "rain_rolling_mean", "rain_rolling_std",
        "log_PRECTOTCORR"  # include target if scaler was fitted with it
    ]

    # Select last row for prediction
    last_row = df[final_features_scaler].tail(1)
    
    # 8️⃣ Scale
    scaled_features = scaler_monthly.transform(last_row.values)
    print("Scaled features:", scaled_features)

    # 9️⃣ Predict
    X_input = np.expand_dims(scaled_features, axis=0)
    pred_scaled = model_monthly.predict(X_input)
    print("Predicted scaled rainfall:", pred_scaled)

    # 10️⃣ Inverse-transform to mm
    rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, scaled_features)
    rainfall_mm = max(0, rainfall_mm)

    return {
        "predicted_scaled": pred_scaled.tolist(),
        "predicted_mm": round(float(rainfall_mm), 3),
        "scaled_features": scaled_features.tolist(),
        "location": {"latitude": latitude, "longitude": longitude}
    }
