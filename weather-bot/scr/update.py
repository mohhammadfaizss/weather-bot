

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

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


script_location = Path(__file__).resolve().parent.parent
BASE_DIR = script_location / "Data"

metar_data_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"




today = date.today()

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

    script_location = Path(__file__).resolve().parent.parent
    BASE_DIR = script_location / "Data"
    FORECAST_DIR    = BASE_DIR / "forcast_data"
    API_URL         = "https://previous-runs-api.open-meteo.com/v1/forecast"


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

        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)


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


    def updating_forecast(self, city):
        print(f"\n📅 Updating forcast : {self.start_date} → {self.end_date}\n")

        
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



def updating_metar(city, lastdate) :

        
            params = {
                "network": city["network"],
                "station": city["station"],
                "data": ["tmpc", "dwpc", "relh", "skyc1", "sknt", "skyc2", "skyc3", "metar"],
                "year1": lastdate.year,
                "month1": lastdate.month,
                "day1": lastdate.day,
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
                return

            new_df = pd.read_csv(StringIO(text))

            child_folder = "metar_data"
            file_path = BASE_DIR / child_folder / f"{city['station']}.csv"

            

            
            # print(new_df["tmpc"])

            if os.path.exists(file_path):
                old_df = pd.read_csv(file_path)

                # Combine + remove duplicates
                combined_df = pd.concat([old_df, new_df]).drop_duplicates()

            else:
                combined_df = new_df

            combined_df.to_csv(file_path, index=False)

            print(f"[{city['station']}] File updated → {len(combined_df)} total rows")



def gettingcity(city):
    
    today = date.today()

    script_location = Path(__file__).resolve().parent.parent
    filelocation = BASE_DIR /"forcast_data" / f"historical_{city['name']}.csv"
    file = pd.read_csv(filelocation)

    metarlocation = BASE_DIR /"metar_data" / f"{city['station']}.csv"
    metar_file = pd.read_csv(metarlocation)
    metar_last_date = pd.to_datetime(metar_file['valid']).dt.normalize().iloc[-1].date()


    

    last_date = pd.to_datetime(file["date"]).dt.normalize().iloc[-1].date()
    start_date = last_date.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    yesterday = date.today() - timedelta(days=1)

    print("=" * 65)
    print(f"  Updating database   |  {city['name']} | {city['station']}")
    print("=" * 65)

    if(last_date == today and metar_last_date == yesterday):
        print("Metar and Forecast data is up to date")
        return


    updater = UpdateMosDatabase(start_date= start_date, end_date= end_date)
    updating_metar(city, metar_last_date)
    print("=" * 65)
    updater.updating_forecast(city)
    print("=" * 65)
    
