import openmeteo_requests
from openmeteo_sdk.Variable import Variable
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime
from datetime import time
from pathlib import Path
import numpy as np




cities = [
        "beijing", "london", "tokyo", "lucknow", "mexico-city", "nyc", 
        "toronto", "chicago", "atlanta", "dallas", "denver", "san-francisco", 
        "houston", "miami", "los-angeles", "austin", "seattle", "panama-city", 
        "sao-paulo", "buenos-aires", "wellington", "jakarta", "seoul", 
        "singapore", "hong-kong", "shanghai", "taipei", "kuala-lumpur", 
        "chongqing", "chengdu", "busan", "cape-town", "lagos", "jeddah", 
        "tel-aviv", "munich", "paris", "ankara", "istanbul", "moscow", 
        "madrid", "helsinki", "amsterdam", "warsaw", "milan"
    ]

while True:
        theDate = input("Enter date (YYYY-MM-DD): ")
        try:
            datetime.strptime(theDate, '%Y-%m-%d')
            break
        except ValueError:
            print("Incorrect format, should be YYYY-MM-DD")


script_location = Path(__file__).resolve().parent
BASE_DIR = script_location / "Data" / theDate

print(f"Script location: {script_location}")
print(f"Base dir: {BASE_DIR}")

market_path = BASE_DIR / f"market-{theDate}.csv"
model_runs_path= BASE_DIR / f"model_runs_Report.csv"
ensemble_data_path = BASE_DIR / "ensemble-data.csv"

while True:
     city_name = input("Enter the city name : ")

     if(city_name in cities):
          break
     
     else :
          print("Invalid city name. Try again")



market_data = pd.read_csv(market_path)

market_data_filtered = market_data[market_data['city'] == city_name]

model_runs_data = pd.read_csv(model_runs_path)

model_runs_data_filtered = model_runs_data[model_runs_data["city"] == city_name]

ensemble_data = pd.read_csv(ensemble_data_path)


ensemble_data_filtered = ensemble_data[ensemble_data["city"] == city_name]



print(market_data_filtered)
print(model_runs_data_filtered)
# print(ensemble_data_filtered)



ensemble_data_filtered['mean']   = ensemble_data_filtered.filter(like='member_').mean(axis=1)
ensemble_data_filtered['median'] = ensemble_data_filtered.filter(like='member_').median(axis=1)
ensemble_data_filtered['std']    = ensemble_data_filtered.filter(like='member_').std(axis=1)
ensemble_data_filtered['p10']    = ensemble_data_filtered.filter(like='member_').quantile(0.10, axis=1)
ensemble_data_filtered['p25']    = ensemble_data_filtered.filter(like='member_').quantile(0.25, axis=1)
ensemble_data_filtered['p75']    = ensemble_data_filtered.filter(like='member_').quantile(0.75, axis=1)
ensemble_data_filtered['p90']    = ensemble_data_filtered.filter(like='member_').quantile(0.90, axis=1)

ensemble_data_filtered = ensemble_data_filtered.set_index('model_name')

print(ensemble_data_filtered['mean'])
print(ensemble_data_filtered['median'])
print(ensemble_data_filtered['std'])
print(ensemble_data_filtered['p10'])
print(ensemble_data_filtered['p25'])
print(ensemble_data_filtered['p75'])
print(ensemble_data_filtered['p90'])
