import requests
import pandas as pd
from datetime import datetime

def fetch_nasa_monthly_data(latitude=6.585, longitude=3.983, years_back=3):
    """
    Fetch NASA POWER monthly weather data for the past `years_back` years.
    Returns a DataFrame indexed by month.
    """

    end_date = datetime.utcnow().date()
    start_year = end_date.year - years_back

    params = {
        "parameters": "T2M,RH2M,WS10M,WD10M,ALLSKY_SFC_SW_DWN,EVPTRNS,PS,QV2M,T2M_RANGE,TS,CLRSKY_SFC_SW_DWN,PRECTOTCORR",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": f"{start_year}01",
        "end": end_date.strftime("%Y%m"),
        "format": "JSON"
    }

    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Create DataFrame from JSON
    df = pd.DataFrame(data["properties"]["parameter"])
    df = pd.DataFrame({k: pd.Series(v) for k, v in df.items()})

    # The index is in YYYYMM format → convert to datetime
    df.index = pd.to_datetime(df.index, format="%Y%m")

    # Ensure all features exist and fill missing
    REQUIRED_FEATURES = [
        'T2M','RH2M','WS10M','WD10M','ALLSKY_SFC_SW_DWN',
        'EVPTRNS','PS','QV2M','T2M_RANGE','TS','CLRSKY_SFC_SW_DWN','PRECTOTCORR'
    ]
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[REQUIRED_FEATURES].ffill().bfill()

    # Reset index to have a DATE column like your training data
    df = df.reset_index().rename(columns={"index": "DATE"})
    return df
