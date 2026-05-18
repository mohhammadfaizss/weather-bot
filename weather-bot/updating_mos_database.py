import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
from openmeteo_requests import OpenMeteoRequestsError
from pathlib import Path
from datetime import date, timedelta
import csv
import requests
from io import StringIO
import os
# from gettinghistoricalstuff import UpdateMosDatabase

cities = [
    {"name": "beijing", "station": "ZBAA", "timezone": "Asia/Shanghai", "lat": 40.0801, "lon": 116.5846, "network": "CN__ASOS"},
    {"name": "london", "station": "EGLC", "timezone": "Europe/London", "lat": 51.5085, "lon": -0.1257, "network": "GB__ASOS"},
    {"name": "tokyo", "station": "RJTT", "timezone": "Asia/Tokyo", "lat": 35.6895, "lon": 139.6917, "network": "JP__ASOS"},
    {"name": "lucknow", "station": "VILK", "timezone": "Asia/Kolkata", "lat": 26.74, "lon": 80.86, "network": "IN__ASOS"},
    {"name": "mexico-city", "station": "MMMX", "timezone": "America/Mexico_City", "lat": 19.44, "lon": -99.08, "network": "MX__ASOS"},
    {"name": "nyc", "station": "LGA", "timezone": "America/New_York", "lat": 40.76, "lon": -73.86, "network": "NY_ASOS"},
    {"name": "toronto", "station": "CYYZ", "timezone": "America/Toronto", "lat": 43.71, "lon": -79.66, "network": "CA_ON_ASOS"},
    {"name": "chicago", "station": "ORD", "timezone": "America/Chicago", "lat": 41.98, "lon": -87.91, "network": "IL_ASOS"},
    {"name": "atlanta", "station": "ATL", "timezone": "America/New_York", "lat": 33.64, "lon": -84.41, "network": "GA_ASOS"},
    {"name": "dallas", "station": "DAL", "timezone": "America/Chicago", "lat": 32.85, "lon": -96.87, "network": "TX_ASOS"},
    {"name": "denver", "station": "BKF", "timezone": "America/Denver", "lat": 39.7, "lon": -104.76, "network": "CO_ASOS"},
    {"name": "san-francisco", "station": "SFO", "timezone": "America/Los_Angeles", "lat": 37.62, "lon": -122.39, "network": "CA_ASOS"},
    {"name": "houston", "station": "HOU", "timezone": "America/Chicago", "lat": 29.63, "lon": -95.25, "network": "TX_ASOS"},
    {"name": "miami", "station": "MIA", "timezone": "America/New_York", "lat": 25.85, "lon": -80.24, "network": "FL_ASOS"},
    {"name": "los-angeles", "station": "LAX", "timezone": "America/Los_Angeles", "lat": 33.96, "lon": -118.4, "network": "CA_ASOS"},
    {"name": "austin", "station": "AUS", "timezone": "America/Chicago", "lat": 30.16, "lon": -97.69, "network": "TX_ASOS"},
    {"name": "seattle", "station": "SEA", "timezone": "America/Los_Angeles", "lat": 47.44, "lon": -122.3, "network": "WA_ASOS"},
    {"name": "panama-city", "station": "MPMG", "timezone": "America/Panama", "lat": 8.98, "lon": 79.56, "network": "PA__ASOS"},
    {"name": "sao-paulo", "station": "SBGR", "timezone": "America/Sao_Paulo", "lat": -23.42, "lon": -46.48, "network": "BR__ASOS"},
    {"name": "buenos-aires", "station": "SAEZ", "timezone": "America/Argentina/Buenos_Aires", "lat": -34.79, "lon": -58.52, "network": "AR__ASOS"},
    {"name": "wellington", "station": "NZWN", "timezone": "Pacific/Auckland", "lat": -41.32, "lon": 174.8, "network": "NF__ASOS"},
    {"name": "jakarta", "station": "WIHH", "timezone": "Asia/Jakarta", "lat": -6.26, "lon": 106.89, "network": "ID__ASOS"},
    {"name": "seoul", "station": "RKSI", "timezone": "Asia/Seoul", "lat": 37.49, "lon": 126.49, "network": "KR__ASOS"},
    {"name": "singapore", "station": "WSSS", "timezone": "Asia/Singapore", "lat": 1.35, "lon": 104, "network": "SG__ASOS"},
    {"name": "hong-kong", "station": "VHHH", "timezone": "Asia/Hong_Kong", "lat": 22.2783, "lon": 114.1747, "network": "HK__ASOS"},
    {"name": "shanghai", "station": "ZSPD", "timezone": "Asia/Shanghai", "lat": 31.15, "lon": 121.8, "network": "CN__ASOS"},
    {"name": "taipei", "station": "RCSS", "timezone": "Asia/Taipei", "lat": 25.06, "lon": 121.55, "network": "TW__ASOS"},
    {"name": "kuala-lumpur", "station": "WMKK", "timezone": "Asia/Kuala_Lumpur", "lat": 2.77, "lon": 101.7, "network": "MY__ASOS"},
    {"name": "chongqing", "station": "ZUCK", "timezone": "Asia/Shanghai", "lat": 29.72, "lon": 106.63, "network": "CN__ASOS"},
    {"name": "chengdu", "station": "ZUUU", "timezone": "Asia/Shanghai", "lat": 30.57, "lon": 103.96, "network": "CN__ASOS"},
    {"name": "busan", "station": "RKPK", "timezone": "Asia/Seoul", "lat": 35.18, "lon": 128.95, "network": "KR__ASOS"},
    {"name": "cape-town", "station": "FACT", "timezone": "Africa/Johannesburg", "lat": -33.97, "lon": 18.59, "network": "ZA__ASOS"},
    {"name": "lagos", "station": "DNMM", "timezone": "Africa/Lagos", "lat": 6.45, "lon": 3.39, "network": "NG__ASOS"},
    {"name": "jeddah", "station": "OEJN", "timezone": "Asia/Riyadh", "lat": 21.58, "lon": 39.16, "network": "SA__ASOS"},
    {"name": "tel-aviv", "station": "LLBG", "timezone": "Asia/Jerusalem", "lat": 32.0809, "lon": 34.7806, "network": "IL__ASOS"},
    {"name": "munich", "station": "EDDM", "timezone": "Europe/Berlin", "lat": 48.35, "lon": 11.79, "network": "DE__ASOS"},
    {"name": "paris", "station": "LFPB", "timezone": "Europe/Paris", "lat": 49.02, "lon": 2.59, "network": "FR__ASOS"},
    {"name": "ankara", "station": "LTAC", "timezone": "Europe/Istanbul", "lat": 40.24, "lon": 33.03, "network": "TR__ASOS"},
    {"name": "istanbul", "station": "LTFM", "timezone": "Europe/Istanbul", "lat": 41.0138, "lon": 28.9497, "network": "TR__ASOS"},
    {"name": "moscow", "station": "UUEE", "timezone": "Europe/Moscow", "lat": 55.7522, "lon": 37.6156, "network": "RU__ASOS"},
    {"name": "madrid", "station": "LEMD", "timezone": "Europe/Madrid", "lat": 40.45, "lon": -3.58, "network": "ES__ASOS"},
    {"name": "helsinki", "station": "EFHK", "timezone": "Europe/Helsinki", "lat": 60.32, "lon": 24.97, "network": "FI__ASOS"},
    {"name": "amsterdam", "station": "EHAM", "timezone": "Europe/Amsterdam", "lat": 52.31, "lon": 4.76, "network": "NL__ASOS"},
    {"name": "warsaw", "station": "EPWA", "timezone": "Europe/Warsaw", "lat": 52.17, "lon": 20.98, "network": "PL__ASOS"},  
    {"name": "milan", "station": "LIMC", "timezone": "Europe/Rome", "lat": 45.63, "lon": 8.7, "network": "IT__ASOS"}
]


MODELS = [
    "ecmwf_ifs",
    "ecmwf_ifs025",
    "gem_seamless",
    "gfs_seamless",
    "icon_seamless",
    "gfs_hrrr",
    "ukmo_seamless",
]

MODEL_ID_MAP = {
    30: "ecmwf_ifs",
    2:  "gfs_seamless",
    16: "gem_seamless",
    20: "icon_seamless",    
    4:  "gfs_hrrr",
    82: "ukmo_seamless",
    60: "ecmwf_ifs025"
}

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


script_location = Path(__file__).resolve().parent
BASE_DIR = script_location / "mos_data"

metar_data_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
historical_forcast_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
max_metar_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"

today = date.today()

class update_mos_database :

    def updating_forcast(self) :
        for city in cities :

            all_model_data = []
            


            for model in MODELS:
                params = {
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "start_date": "2026-05-02",
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
                child_folder = "forcast_data"
                folder_path = BASE_DIR / child_folder /filename
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
                "day1": 28,
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

            child_folder = "metar_data"
            file_path = BASE_DIR / child_folder / f"{city['station']}.csv"

            

            
            print(new_df["tmpc"])

            if os.path.exists(file_path):
                old_df = pd.read_csv(file_path)

                # Combine + remove duplicates
                combined_df = pd.concat([old_df, new_df]).drop_duplicates()

            else:
                combined_df = new_df

            combined_df.to_csv(file_path, index=False)

            print(f"[{city['station']}] File updated → {len(combined_df)} total rows")

    def updating_max_metar(self) :
        yesterday = today - timedelta(days=1)
        for city in cities :
                params = {
                    "network": city["network"],
                    "station": city["station"],
                    # "var": ["max_temp_f", "max_dewpoint_f", "precip_in", "max_rh", "max_feel", "max_wind_speed_kts", "skyc3", "metar"],
                    "year1": 2026,
                    "month1": 5,
                    "day1": 1,
                    "year2": yesterday.year,
                    "month2": yesterday.month,
                    "day2": yesterday.day,
                    "tz": "Etc/UTC",
                    "format": "onlycomma",
                    "latlon": "no",
                    "elev": "no",
                    "missing": "null",
                    "trace": "T",
                    "direct": "no",
                    "report_type": ["1", "3", "4"]
                }


                response = requests.get(max_metar_url, params=params)

                text = response.text.strip()
                if not text:
                    continue

                new_df = pd.read_csv(StringIO(text))

                child_folder = "max_metar"
                
                file_path = BASE_DIR  / child_folder / f"{city['station']}.csv"
                

                

                
                

                if os.path.exists(file_path):
                    old_df = pd.read_csv(file_path)

                    # Combine + remove duplicates
                    combined_df = pd.concat([old_df, new_df]).drop_duplicates()

                else:
                    combined_df = new_df

                combined_df.to_csv(file_path, index=False)

                print(f"[{city['station']}] File updated → {len(combined_df)} total rows")

class UpdateMosDatabase:

    MODELS = {
    "ecmwf_ifs":      "2022-01-01",   # ~3 years
    "ecmwf_ifs025":   "2024-11-01",   # ~6 months only
    "gem_seamless":   "2022-01-01",
    "gfs_seamless":   "2022-01-01",
    "icon_seamless":  "2022-01-01",
    "ukmo_seamless":  "2023-01-01",   # ~2 years
    }

    MODEL_ID_MAP = {
        30: "ecmwf_ifs",
        60: "ecmwf_ifs025",
        16: "gem_seamless",
        2:  "gfs_seamless",
        20: "icon_seamless",
        82: "ukmo_seamless",
    }

    # Base variables — the _previous_day1 suffix is added automatically below
    BASE_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "dew_point_2m",
        "wind_direction_10m",
        "wind_speed_10m",
        "shortwave_radiation",
    ]

    # The exact column order your CSV expects — used to reindex every output frame
    EXPECTED_COLUMNS = [
        "city", "date",
        "temperature_2m_ecmwf_ifs",      "temperature_2m_ecmwf_ifs025",
        "temperature_2m_gem_seamless",   "temperature_2m_gfs_seamless",
        "temperature_2m_icon_seamless",  "temperature_2m_ukmo_seamless",
        "relative_humidity_2m_ecmwf_ifs",     "relative_humidity_2m_ecmwf_ifs025",
        "relative_humidity_2m_gem_seamless",  "relative_humidity_2m_gfs_seamless",
        "relative_humidity_2m_icon_seamless", "relative_humidity_2m_ukmo_seamless",
        "shortwave_radiation_ecmwf_ifs",      "shortwave_radiation_ecmwf_ifs025",
        "shortwave_radiation_gem_seamless",   "shortwave_radiation_gfs_seamless",
        "shortwave_radiation_icon_seamless",  "shortwave_radiation_ukmo_seamless",
        "cloud_cover_ecmwf_ifs",     "cloud_cover_ecmwf_ifs025",
        "cloud_cover_gem_seamless",  "cloud_cover_gfs_seamless",
        "cloud_cover_icon_seamless", "cloud_cover_ukmo_seamless",
        "wind_speed_10m_ecmwf_ifs",      "wind_speed_10m_ecmwf_ifs025",
        "wind_speed_10m_gem_seamless",   "wind_speed_10m_gfs_seamless",
        "wind_speed_10m_icon_seamless",  "wind_speed_10m_ukmo_seamless",
        "dew_point_2m_ecmwf_ifs", "dew_point_2m_ecmwf_ifs025",
        "dew_point_2m_gem_seamless", "dew_point_2m_gfs_seamless",
        "dew_point_2m_icon_seamless", "dew_point_2m_ukmo_seamless",
        "wind_direction_10m_ecmwf_ifs", "wind_direction_10m_ecmwf_ifs025",
        "wind_direction_10m_gem_seamless", "wind_direction_10m_gfs_seamless",
        "wind_direction_10m_icon_seamless", "wind_direction_10m_ukmo_seamless",
        

    ]

    script_location = Path(__file__).resolve().parent
    BASE_DIR = script_location / "mos_data"
    FORECAST_DIR    = BASE_DIR / "forcast_data"
    API_URL         = "https://previous-runs-api.open-meteo.com/v1/forecast"

    cache_session  = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session  = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo      = openmeteo_requests.Client(session=retry_session)


    def __init__(self, start_date: str = None, end_date: str = None):
        """
        start_date / end_date: "YYYY-MM-DD" strings.
        Defaults to yesterday → yesterday (single day update mode).
        """
        today = date.today()
        # yesterday        = (date.today() - timedelta(days=1)).isoformat()
        today = today.isoformat()
        self.start_date  = start_date or today
        self.end_date    = end_date   or today

    def previous_day1_variables(self):
        return [f"{v}_previous_day1" for v in self.BASE_VARIABLES]
    
    def strip_previous_day1(self, col_name: str) -> str:
        """Remove the _previous_day1 suffix so column names match the CSV schema."""
        return col_name.replace("_previous_day1", "")
    
    def fetch_model(self, city: dict, model: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """
        Fetch previous_day1 data for one city + model combination.
        Returns a tidy long DataFrame with columns:
            date, city, variable, model, value
        Returns None if the model has no data for this date range.
        """
        params = {
            "latitude":     city["lat"],
            "longitude":    city["lon"],
            "hourly":       self.previous_day1_variables(),
            "models":       model,
            "start_date":   start_date,
            "end_date":     end_date,
            "timezone":     "GMT",
            # previous-runs API needs these to anchor which run we want
        }

        try:
            responses = self.openmeteo.weather_api(self.API_URL, params=params)
        except OpenMeteoRequestsError as e:
            print(f"  ⚠ {city['name']} / {model}: API error — {e}")
            return None

        response  = responses[0]
        m_id      = response.Model()
        base_name = self.MODEL_ID_MAP.get(m_id, f"model_id_{m_id}")

        hourly = response.Hourly()
        dates  = pd.date_range(
            start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
            end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq      = pd.Timedelta(seconds=hourly.Interval()),
            inclusive = "left",
        )

        rows = {"date": dates, "city": city["name"]}

        api_vars = self.previous_day1_variables()
        for idx, api_var in enumerate(api_vars):
            clean_col = f"{self.strip_previous_day1(api_var)}_{base_name}"
            rows[clean_col] = hourly.Variables(idx).ValuesAsNumpy()

        return pd.DataFrame(rows)

    def build_wide_frame(self, city_name: str, model_frames: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all per-model DataFrames on (city, date) into a single wide frame
        that matches EXPECTED_COLUMNS exactly. Missing models → NaN columns.
        """
        if not model_frames:
            return pd.DataFrame(columns=self.EXPECTED_COLUMNS)

        # Start from the first frame and outer-join the rest on date
        merged = model_frames[0]
        for frame in model_frames[1:]:
            merged = pd.merge(merged, frame, on=["city", "date"], how="outer")

        # Guarantee every expected column is present (fills NaN for absent models)
        for col in self.EXPECTED_COLUMNS:
            if col not in merged.columns:
                merged[col] = np.nan

        return merged[self.EXPECTED_COLUMNS]

    def save_to_csv(self, df: pd.DataFrame, city_name: str):
        """Append new rows to the existing CSV, deduplicating on (city, date)."""
        filepath = self.FORECAST_DIR / f"historical_{city_name}.csv"

        if filepath.exists():
            existing = pd.read_csv(filepath, parse_dates=["date"])

            # Ensure existing file also has all expected columns
            for col in self.EXPECTED_COLUMNS:
                if col not in existing.columns:
                    existing[col] = np.nan
            existing = existing[self.EXPECTED_COLUMNS]

            combined = pd.concat([existing, df], ignore_index=True)
        else:
            print(f"  ℹ No existing file for {city_name} — creating new.")
            combined = df

        # Deduplicate: keep the newest fetch for each (city, date) pair
        combined["date"] = pd.to_datetime(combined["date"], utc=True)
        combined = (
            combined
            .drop_duplicates(subset=["city", "date"], keep="last")
            .sort_values("date")
        )

        self.FORECAST_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(filepath, index=False, encoding="UTF-8")
        print(f"  ✅ Saved {len(combined)} total rows → {filepath.name}")


    def updating_forecast(self):
        print(f"\n📅 Fetching previous_day1 runs: {self.start_date} → {self.end_date}\n")

        for city in cities:
            print(f"\n🌍 {city['name'].upper()}")
            model_frames = []

            for model, model_start in self.MODELS.items():

                # Skip entirely if the requested window predates this model's archive
                if self.end_date < model_start:
                    print(f"  ⏭ {model}: no data before {model_start} — skipping")
                    continue

                # Clamp start date to the model's availability window
                effective_start = max(self.start_date, model_start)
                if effective_start != self.start_date:
                    print(f"  ⚠ {model}: clamping start to {effective_start} (archive begins {model_start})")

                frame = self.fetch_model(city, model, effective_start, self.end_date)
                if frame is not None:
                    model_frames.append(frame)
                    print(f"  ✔ {model}: {len(frame)} rows fetched")

            if not model_frames:
                print(f"  ❌ No data retrieved for {city['name']}")
                continue

            # If models returned different date ranges (due to clamping),
            # align them so every row has the same date index
            df_wide = self.build_wide_frame(city["name"], model_frames)
            self.save_to_csv(df_wide, city["name"])

if __name__ == "__main__":
    get_the_update = update_mos_database()
    updater = UpdateMosDatabase()



    # get_the_update.updating_max_metar()
    updater.updating_forecast()
    # get_the_update.updating_metar()

    # get_the_update.updating_forcast()