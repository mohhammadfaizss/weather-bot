import openmeteo_requests
from openmeteo_sdk.Variable import Variable
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime
from pathlib import Path

# --- 1. SETUP ---
# Anchoring the directory to where this script is saved
script_location = Path(__file__).resolve().parent
BASE_DIR = script_location / "Data"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Setup the API client with cache and retry logic
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

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

# --- 2. VALIDATION ---
while True:
    theDate = input("Enter date (YYYY-MM-DD): ")
    try:
        datetime.strptime(theDate, '%Y-%m-%d')
        break
    except ValueError:
        print("Incorrect format, should be YYYY-MM-DD")

# --- 3. EXECUTION ---
url = "https://ensemble-api.open-meteo.com/v1/ensemble"

for loc in locations:
    city_name = loc['name']
    print(f"\nProcessing: {city_name}...")

    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "daily": "temperature_2m_max",
        "models": "ecmwf_ifs025_ensemble",
        "timezone": "auto",
        "start_date": theDate,
        "end_date": theDate,
    }

    try:
        # Requesting data ONLY for this city
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0] 

        daily = response.Daily()
        daily_variables = [daily.Variables(i) for i in range(daily.VariablesLength())]
        
        # Filtering for temperature variables at 2m altitude
        daily_temperature_2m_max = filter(
            lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2, 
            daily_variables
        )   

        # Preparing the data structure for Pandas
        daily_data = {"date": [theDate]}

        for variable in daily_temperature_2m_max:
            member = variable.EnsembleMember()
            daily_data[f"member_{member}"] = variable.ValuesAsNumpy()

        # Create DataFrame
        df = pd.DataFrame(data=daily_data)

        # Building the Hierarchical Path: Data / city_name / date
        target_path = BASE_DIR / city_name / theDate
        target_path.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        file_path = target_path / "ensemble_members_forecast.csv"
        df.to_csv(file_path, index=False)
        
        print(f"✅ Success: {city_name} ensemble data saved.")

    except Exception as e:
        print(f"❌ Failed to fetch data for {city_name}: {e}")

print("\n--- All city ensemble forecasts are organized and saved! ---")