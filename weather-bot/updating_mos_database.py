import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from openmeteo_requests import OpenMeteoRequestsError
from pathlib import Path
from datetime import date
from variable import MODEL_ID_MAP , MODELS, cities
import csv
import requests
from io import StringIO
import os

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


script_location = Path(__file__).resolve().parent
BASE_DIR = script_location / "mos_data"

metar_data_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
historical_forcast_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

today = date.today()

class update_mos_database :

    def updating_forcast(self) :
        for city in cities :

            all_model_data = []
            


            for model in MODELS:
                params = {
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "start_date": today,
                    "end_date": today,
                    "hourly": ["temperature_2m", "relative_humidity_2m", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "shortwave_radiation"],
                    "models": model,
                    "timezone": "GMT",
                }

                try:
                    responses = openmeteo.weather_api(historical_forcast_url, params=params)

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
                old_df = pd.read_csv(folder_path)
                combined_df = pd.concat([old_df, df_wide]).drop_duplicates()

                combined_df.to_csv(folder_path, index=False, encoding="UTF-8")
                
                print(f"\n✅ Data for {city["name"]} successfully stored!")
                print(df_wide.head())
            else:
                print("\n❌ No data found.")
            

    def updating_metar(self) :
        for city in cities :

            params = {
                "network": city["network"],
                "station": city["station"],
                "data": ["tmpc", "dwpc", "relh", "skyc1", "sknt", "skyc2", "skyc3", "metar"],
                "year1": 2026,
                "month1": 4,
                "day1": 20,
                "year2": today.year,
                "month2": today.month,
                "day2": today.day,
                "tz": "Etc/UTC",
                "format": "onlycomma",
                "latlon": "no",
                "elev": "no",
                "missing": "null",
                "trace": "T",
                "direct": "no",
                "report_type": ["1", "3", "4"]
            }


            response = requests.get(metar_data_url, params=params)

            text = response.text.strip()
            if not text:
                continue

            new_df = pd.read_csv(StringIO(text))

            file_path = BASE_DIR / f"{city['station']}.csv"

            

            
            print(new_df["tmpc"])

            if os.path.exists(file_path):
                old_df = pd.read_csv(file_path)

                # Combine + remove duplicates
                combined_df = pd.concat([old_df, new_df]).drop_duplicates()

            else:
                combined_df = new_df

            combined_df.to_csv(file_path, index=False)

            print(f"[{city['station']}] File updated → {len(combined_df)} total rows")


get_the_update = update_mos_database()

get_the_update.updating_metar()
get_the_update.updating_forcast()