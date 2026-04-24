import requests
import csv
from pathlib import Path
import pandas as pd
from io import StringIO
import os
from variable import *

script_location = Path(__file__).resolve().parent
BASE_DIR = script_location / "mos_data"

url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


for city in cities :

    params = {
        "network": city["network"],
        "station": city["station"],
        "data": ["tmpc", "dwpc", "relh", "skyc1", "sknt", "skyc2", "skyc3", "metar"],
        "year1": 2026,
        "month1": 4,
        "day1": 23,
        "year2": 2026,
        "month2": 4,
        "day2": 24,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "null",
        "trace": "T",
        "direct": "no",
        "report_type": ["1", "3", "4"]
    }


    response = requests.get(url, params=params)

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