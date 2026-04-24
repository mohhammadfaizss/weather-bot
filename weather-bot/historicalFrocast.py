import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from openmeteo_requests import OpenMeteoRequestsError
from pathlib import Path
from datetime import date
from variable import MODEL_ID_MAP , MODELS, cities
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

script_location = Path(__file__).resolve().parent

BASE_DIR = script_location / "mos_data"



today = date.today()

for city in cities :

    all_model_data = []
    


    for model in MODELS:
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "start_date": "2026-04-20",
            "end_date": today,
            "hourly": ["temperature_2m", "relative_humidity_2m", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "shortwave_radiation"],
            "models": model,
            "timezone": "GMT",
        }

        try:
            responses = openmeteo.weather_api(url, params=params)

            for response in responses:
                m_id = response.Model()
                base_name = MODEL_ID_MAP.get(m_id, f"model_id_{m_id}")

                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
                cloud_cover_low = hourly.Variables(2).ValuesAsNumpy()
                cloud_cover_mid = hourly.Variables(3).ValuesAsNumpy()
                cloud_cover_high = hourly.Variables(4).ValuesAsNumpy()
                wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
                shortwave_radiation = hourly.Variables(6).ValuesAsNumpy()

                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    ),
                    "city": f"{city["name"]}",
                    "model_max_temp": base_name,
                    "temperature_2m": hourly_temperature_2m,
                    # "model_max_humidity": base_name,
                    "relative_humidity_2m": relative_humidity_2m,
                    "shortwave_radiation" : shortwave_radiation,
                    "cloud_cover_low" : cloud_cover_low,
                    "cloud_cover_high" : cloud_cover_high,
                    "cloud_cover_mid" : cloud_cover_mid,
                    "wind_speed_10m" : wind_speed_10m
                }
                all_model_data.append(pd.DataFrame(hourly_data))

        except OpenMeteoRequestsError as e:
            print(f"Skipping model {model} because: {e}")
            continue

    if all_model_data:
        # 1. Combine all data into one long DataFrame
        df_long = pd.concat(all_model_data, ignore_index=True)

        # 2. Pivot the table: city and date stay as rows, model names become columns
        df_wide = df_long.pivot(index=['city', 'date'],
                                columns= ['model_max_temp'], 
                                values= ['temperature_2m' , 'relative_humidity_2m', 'shortwave_radiation', 'cloud_cover_low' , 'cloud_cover_mid', 'cloud_cover_high', 'wind_speed_10m']).reset_index()

        df_wide.columns = [
        '_'.join(str(i) for i in col if i).strip('_') 
        for col in df_wide.columns.values
    ]
        # 3. Flatten the index (pivoting sometimes creates a hierarchy in columns)
        df_wide.columns.name = None 

        filename = f"historical_{city["name"]}.csv"
        folder_path = BASE_DIR / filename
        # 4. Save to CSV
        # old_df = pd.read_csv(folder_path)
        # combined_df = pd.concat([old_df, df_wide]).drop_duplicates()

        df_wide.to_csv(folder_path, index=False, encoding="UTF-8")
        
        print(f"\n✅ Data for {city["name"]} successfully stored!")
        print(df_wide.head())
    else:
        print("\n❌ No data found.")