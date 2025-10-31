import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_nasa_data(latitude=6.585, longitude=3.983, days=1):
    """
    Fetch latest NASA POWER weather data for the past N days.
    Default: Lagos, Nigeria
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    
    params = {
        "parameters": "T2M,RH2M,WS10M,WD10M,ALLSKY_SFC_SW_DWN,EVPTRNS,PS,QV2M,T2M_RANGE,TS,CLRSKY_SFC_SW_DWN,PRECTOTCORR",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    response = requests.get(url, params=params)
    data = response.json()

    # Extract data
    df = pd.DataFrame(data["properties"]["parameter"])
    df = pd.DataFrame({k: pd.Series(v) for k, v in df.items()})
    df.index = pd.to_datetime(df.index)
    return df
