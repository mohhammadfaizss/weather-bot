import openmeteo_requests
from openmeteo_sdk.Variable import Variable
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime
from datetime import time
from pathlib import Path
import numpy as np
from variable import *

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
# script location

script_location = Path(__file__).resolve().parent



class Weather_Data_Collection:
    

        
    def date_input(self):
        while True:
            self.theDate = input("Enter date (YYYY-MM-DD): ")
            try:
                datetime.strptime(self.theDate, '%Y-%m-%d')
                break
            except ValueError:
                print("Incorrect format, should be YYYY-MM-DD")

    

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



weather_data = Weather_Data_Collection()
weather_data.date_input()
# weather_data.ensemble_model()
weather_data.main_run_multi_model()