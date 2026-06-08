import sys
from config import cities
from pipeline import run_pipeline, BASE_DIR
from update import gettingcity
import pandas as pd
from pathlib import Path
from datetime import date

if __name__ == "__main__":

    all_city_names = [c["name"] for c in cities]

    city_name = sys.argv[1]
    while True:
        city = next((c for c in cities if c["name"] == city_name), None)
        if city_name in all_city_names:
            print(f"{city_name} found!")
            break
        else:
            print("City not found. Please try again.")
            city_name = input("Enter city name: ").strip().lower()




    gettingcity(city= city)
    

    result = run_pipeline(
        station             = city["station"],
        city                = city["name"],
        timezone            = city["timezone"],
        data_folder         = str( BASE_DIR /"Data"),
        initial_train_days  = 1400,
        run_walk_forward    = False,   # set True for full diagnostic (slow)
        corrector_seed_days = 30,      # Recommended is 30 for more stable seeding
        )