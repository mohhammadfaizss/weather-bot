import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import time


def validate_date(date_text):
    try:
        # We return the actual datetime object if it's valid
        return datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        print("Incorrect format, should be YYYY-MM-DD")
        return None # Return None instead of False

while True:
    raw_input = input("Enter date (YYYY-MM-DD): ")
    date_obj = validate_date(raw_input)

    if date_obj: # If it's not None, the date is valid
        break

theDate = date_obj.strftime("%Y-%m-%d")
folder_str = date_obj.strftime("%B-%d-%Y").lower()

print(theDate)
print(folder_str)

script_location = Path(__file__).resolve().parent
BASE_DIR = script_location / "Data" / theDate 
BASE_DIR.mkdir(parents=True, exist_ok=True)

file_dir = BASE_DIR / f"market-{theDate}.csv"


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





each_market_data = []
for city_name in cities :
    url = f"https://gamma-api.polymarket.com/events/slug/highest-temperature-in-{city_name}-on-{folder_str}"


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
                        "date" : theDate,
                        "city" : city_name,
                        "title": item.get("groupItemTitle") or "General Market",
                        "prices": item.get("outcomePrices") or "[No Price Data]",
                        "question": item.get("question") or "No question",
                        "resolutionSource": item.get("resolutionSource") or "No resolution source"
                    }
        each_market_data.append(entry)
        time.sleep(0.5)
    print(f"Data of {city_name} has been saved")
        

df = pd.DataFrame(each_market_data)


df.to_csv(file_dir, index=False)






