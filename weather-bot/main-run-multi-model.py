import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os
import sys
from datetime import datetime
from pathlib import Path

script_location = Path(__file__).resolve().parent



cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

locations = [
    {"name": "beijing", "lat": 40.0801, "lon": 116.5846},
    {"name": "london", "lat": 51.5085, "lon": -0.1257},
    {"name": "tokyo", "lat": 35.6895, "lon": 139.6917},
    {"name": "lucknow", "lat": 26.74, "lon": 80.86},
    {"name": "mexico-city", "lat": 19.44 , "lon": -99.08 },
    {"name": "nyc", "lat": 40.76, "lon": -73.86},
    {"name": "toronto", "lat": 43.71 , "lon": -79.66},
    {"name": "chicago", "lat": 41.98, "lon": -87.91},
    {"name": "atlanta", "lat": 33.64, "lon": -84.41},
    {"name": "dallas", "lat": 32.85, "lon": -96.87},
    {"name": "denver", "lat": 39.7, "lon": -104.76},
    {"name": "san-francisco", "lat": 37.62, "lon": -122.39},
    {"name": "houston", "lat": 29.63, "lon": -95.25},
    {"name": "miami", "lat": 25.85 , "lon": -80.24},
    {"name": "los-angeles", "lat": 33.96, "lon": -118.4},
    {"name": "austin", "lat": 30.16, "lon": -97.69},
    {"name": "seattle", "lat": 47.44, "lon": -122.3},
    {"name": "panama-city", "lat": 8.98, "lon": 79.56 },
    {"name": "sao-paulo", "lat": -23.42 , "lon": -46.48},
    {"name": "buenos-aires", "lat": -34.79 , "lon": -58.52},
    {"name": "wellington", "lat": -41.32, "lon": 174.8},
    {"name": "jakarta", "lat": -6.26, "lon": 106.89},
    {"name": "seoul", "lat": 37.49, "lon": 126.49},
    {"name": "singapore", "lat": 1.35, "lon": 104},
    {"name": "hong-kong", "lat": 22.2783, "lon": 114.1747},
    {"name": "shanghai", "lat": 31.15 , "lon": 121.8 },
    {"name": "taipei", "lat": 25.06, "lon": 121.55},
    {"name": "kuala-lumpur", "lat": 2.77, "lon": 101.7},
    {"name": "chongqing", "lat": 29.72 , "lon": 106.63},
    {"name": "chengdu", "lat": 30.57 , "lon": 103.96 },
    {"name": "busan", "lat": 35.18  , "lon": 128.95},
    {"name": "cape-town", "lat": -33.97, "lon": 18.59},
    {"name": "lagos", "lat": 6.45 , "lon": 3.39 },
    {"name": "jeddah", "lat": 21.58 , "lon": 39.16},
    {"name": "tel-aviv", "lat": 32.0809, "lon": 34.7806},
    {"name": "munich", "lat": 48.35  , "lon": 11.79 },
    {"name": "paris", "lat": 49.02  , "lon": 2.59  },
    {"name": "ankara", "lat": 40.24   , "lon": 33.03},
    {"name": "istanbul", "lat": 41.0138  , "lon": 28.9497 },
    {"name": "moscow", "lat": 55.7522  , "lon": 37.6156  },
    {"name": "madrid", "lat": 40.45 , "lon": -3.58 },
    {"name": "helsinki", "lat": 60.32   , "lon": 24.97},
    {"name": "amsterdam", "lat": 52.31  , "lon": 4.76},
    {"name": "warsaw", "lat": 52.17  , "lon": 20.98},
    {"name": "milan", "lat": 45.63, "lon": 8.7 }
]

model_list = ["gfs_seamless", "gem_seamless", "ecmwf_ifs", "gfs_hrrr", "icon_seamless"]


def validate_date(date_text):
    try:
        # This tells Python the expected format (YYYY-MM-DD)
        # You can change '%Y-%m-%d' to '%m/%d/%Y' or whatever you need
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        print("Incorrect format, should be YYYY-MM-DD")
        return False

while True:
    theDate = input("Enter date (YYYY-MM-DD): ")

    if validate_date(theDate):
        break
    
print(f"Validated Date: {theDate}")

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": [loc["lat"] for loc in locations],
    "longitude": [loc["lon"] for loc in locations],
    "daily": "temperature_2m_max",
    "models": model_list,
    "timezone": "auto",
    "start_date": theDate,
    "end_date": theDate,
}

print("Fetching data...")
responses = openmeteo.weather_api(url, params = params)

# 2. CORRECTED INTERNAL ID MAP
# Based on your chart vs CSV alignment:
MODEL_ID_MAP = {
    30: "ecmwf_ifs",    # This is 21.3 (Matches your Green line)
    2:  "gfs_seamless", # This is 20.1 (Matches your Blue line)
    16: "gem_seamless", # Usually GEM or specialized ECMWF
    20: "icon_seamless",
    43: "gem_global"
}

# 3. PROCESSING
total_responses = len(responses)
num_cities = len(locations)
responses_per_city = total_responses // num_cities

current_idx = 0
for loc in locations:
    city_name = loc['name']
    merged_data = {}
    
    print(f"Merging models for: {city_name}...")
    
    for i in range(responses_per_city):
        if current_idx >= total_responses:
            break
            
        res = responses[current_idx]
        current_idx += 1
        
        # Identify the model name
        m_id = res.Model()
        base_name = MODEL_ID_MAP.get(m_id, f"model_id_{m_id}")
        
        # SAFETY: If a city returns the same model name twice, don't overwrite
        m_name = base_name
        counter = 1
        while m_name in merged_data:
            m_name = f"{base_name}_{counter}"
            counter += 1

        daily = res.Daily()
        
        # Dates setup
        if "date" not in merged_data:
            merged_data["date"] = pd.date_range(
                start = pd.to_datetime(daily.Time() + res.UtcOffsetSeconds(), unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd() + res.UtcOffsetSeconds(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            )
            
        merged_data[m_name] = daily.Variables(0).ValuesAsNumpy()
    
    BASE_DIR = script_location / "Data" / city_name / theDate
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    file_name = "model_runs"
    # Save CSV
    if merged_data:
        df = pd.DataFrame(data=merged_data)
        file_path = os.path.join(BASE_DIR, f"{file_name}_Report.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")

print("\nDone! Columns now match the chart values.")