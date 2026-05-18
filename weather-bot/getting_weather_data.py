import requests_cache
from retry_requests import retry
import openmeteo_requests
from openmeteo_sdk.Variable import Variable
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime
from datetime import time
from pathlib import Path
import numpy as np
import requests
from datetime import datetime


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
# script location

script_location = Path(__file__).resolve().parent



class Weather_Data_Collection:
    

    def validate_date(self):
        try:
            return datetime.strptime(self.raw_input, '%Y-%m-%d')
        except ValueError:
            print("Incorrect format, should be YYYY-MM-DD")
            return None
    
    def date_input(self):
        # while True:
        #     self.theDate = input("Enter date (YYYY-MM-DD): ")
        #     self.folder_str = f"{self.theDate.strftime('%B')}-{self.theDate.day}-{self.theDate.year}".lower()
        #     try:
        #         datetime.strptime(self.theDate, '%Y-%m-%d')
        #         break
        #     except ValueError:
        #         print("Incorrect format, should be YYYY-MM-DD")
        while True:
            self.raw_input = input("Enter date (YYYY-MM-DD): ")
            date_obj = self.validate_date()
            if date_obj:
                break

        self.theDate = date_obj.strftime("%Y-%m-%d")
        self.folder_str = f"{date_obj.strftime('%B')}-{date_obj.day}-{date_obj.year}".lower()


    
    

    def ensemble_model(self):
        all_rows = []
        MODEL_ID_MAP = {
            2:  "ncep_gefs_seamless",
            60: "ecmwf_ifs025_ensemble",
            20: "icon_seamless_eps",
            85: "ukmo_global_ensemble_20km"
        }

        for loc in cities:
            city_name = loc['name']
            print(f"\nProcessing: {city_name}...")
            url = "https://ensemble-api.open-meteo.com/v1/ensemble"

            params = {
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "daily": "temperature_2m_max",
                "models": ["ncep_gefs_seamless", "ecmwf_ifs025_ensemble", "icon_seamless_eps", "ukmo_global_ensemble_20km"],
                "timezone": "auto",
                "start_date": self.theDate,
                "end_date": self.theDate,
            }

            try:
                responses = openmeteo.weather_api(url, params=params)

                for response in responses:
                    m_id = response.Model()
                    model_name = MODEL_ID_MAP.get(m_id, f"model_id_{m_id}")  # fallback if ID is unexpected

                    daily = response.Daily()
                    daily_variables = [daily.Variables(i) for i in range(daily.VariablesLength())]

                    daily_temperature_2m_max = filter(
                        lambda x: x.Variable() == Variable.temperature and x.Altitude() == 2,
                        daily_variables
                    )

                    daily_data = { "city": city_name, "model_name": model_name, "date": self.theDate}
                    for variable in daily_temperature_2m_max:
                        member = variable.EnsembleMember()
                        daily_data[f"member_{member}"] = variable.ValuesAsNumpy()[0]

                    all_rows.append(daily_data)
                    
                    print(f"✅ Success: {city_name} — {model_name} saved.")

            except Exception as e:
                if "limit exceeded" in str(e).lower():
                    print("Rate limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                    # Optionally: try to request this city again here
                else:
                    print(f"❌ Failed to fetch data for {city_name}: {e}")
        if all_rows:
            df = pd.DataFrame(all_rows)

            

            BASE_DIR = script_location / "Data"
            BASE_DIR.mkdir(parents=True, exist_ok=True)

            target_path = BASE_DIR /  self.theDate 
            target_path.mkdir(parents=True, exist_ok=True)

            # ✅ Each model gets its own file
            file_path = target_path / "ensemble-data.csv"
            df.to_csv(file_path, index=False)

            
            

    def main_run_multi_model(self):
        all_rows = []

        for loc in cities:
            city_name = loc['name']
            print(f"\nProcessing: {city_name}...")
            url = "https://api.open-meteo.com/v1/forecast"
            
            

            

            params = {
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "daily": "temperature_2m_max",
                "models": MODELS,
                "timezone": "auto",
                "start_date": self.theDate,
                "end_date": self.theDate,
            }

            try:
                responses = openmeteo.weather_api(url, params=params)
                
                # Dictionary for THIS specific city row
                merged_data = {
                    "city": city_name,
                    "date": self.theDate
                }
                
                for res in responses:
                    m_id = res.Model()
                    base_name = MODEL_ID_MAP.get(m_id, f"model_id_{m_id}")
                    
                    # Handle duplicate model names
                    m_name = base_name
                    counter = 1
                    while m_name in merged_data:
                        m_name = f"{base_name}_{counter}"
                        counter += 1

                    daily = res.Daily() 
                    
                    # --- SAFETY CHECK ---
                    # Verify that the model actually returned data for this location
                    if daily and daily.VariablesLength() > 0:
                        # Extract the first value (since start/end date are the same)
                        val = daily.Variables(0).ValuesAsNumpy()[0]
                        merged_data[m_name] = val
                    else:
                        # If no data, fill with NaN (Not a Number) so calculations don't break
                        print(f"⚠️ No data for model {m_name} at {city_name}")
                        merged_data[m_name] = np.nan

                # Append this city's completed data to our global list
                all_rows.append(merged_data)

            except Exception as e:
                print(f"❌ Failed to fetch data for {city_name}: {e}")

            # --- 2. FINAL SAVING (OUTSIDE THE LOOP) ---
            # This runs after all cities are processed
            if all_rows:
                # Create the DataFrame from the collected list of rows
                df = pd.DataFrame(all_rows)
                
                save_path = script_location / "Data" / self.theDate
                save_path.mkdir(parents=True, exist_ok=True)
                
                full_file_path = save_path / "model_runs_Report.csv"
                
                # Save the final report containing all cities
                df.to_csv(full_file_path, index=False)
                print(f"\n✅ All cities processed. Final report saved to: {full_file_path}")
            else:
                print("❌ No data was collected.")

    def market_price(self):

        BASE_DIR = script_location / "Data" / self.theDate 
        BASE_DIR.mkdir(parents=True, exist_ok=True)

        file_dir = BASE_DIR / f"market-{self.theDate}.csv"

        each_market_data = []
        for loc in cities :

            city_name = loc['name']
            url = f"https://gamma-api.polymarket.com/events/slug/highest-temperature-in-{city_name}-on-{self.folder_str}"

            

            try:
                response = requests.get(url, timeout=10)  # ✅ Fail after 10 seconds
                response.raise_for_status()               # ✅ Catch 4xx/5xx errors
                data = response.json()
            except requests.exceptions.Timeout:
                print(f"⏱️ Timeout for {city_name}, skipping...")
                continue
            except requests.exceptions.HTTPError as e:
                print(f"❌ HTTP error for {city_name}: {e.response.status_code}")
                continue
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed for {city_name}: {e}")
                continue


            market = data.get("markets", [])
        
            
            
            for mark in market:

                item = mark
                # yesprice = mark.get("outcomePrices[0]")
                # noprice = mark.get("outcomePrices[1]")
                entry = {       
                                "date" : self.theDate,
                                "city" : city_name,
                                "title": item.get("groupItemTitle") or "General Market",
                                "prices": item.get("outcomePrices") or "[No Price Data]",
                                "question": item.get("question") or "No question",
                                "resolutionSource": item.get("resolutionSource") or "No resolution source"
                            }
                each_market_data.append(entry)
                # time.sleep(0.5)
            print(f"Data of {city_name} has been saved")
                

        df = pd.DataFrame(each_market_data)
            

        df.to_csv(file_dir, index=False)



weather_data = Weather_Data_Collection()
weather_data.date_input()
weather_data.main_run_multi_model()
weather_data.market_price()
# weather_data.ensemble_model()