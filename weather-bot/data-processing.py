import json
import csv
from pathlib import Path
from datetime import datetime



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


def process_model_runs(path , city_name):
    if not Path(path).exists():
        print(f"Error: The file at {path} was not found.")
        return
    
    print(f"\n\n\n\n========== {city_name} ==========")
    with open(path, 'r', encoding='utf-8') as file:
        modeldata = csv.DictReader(file)

        models_to_track = [
            "ecmwf_ifs", 
            "gfs_seamless", 
            "gem_seamless", 
            "gfs_hrrr", 
            "icon_seamless"
        ]

        for row in modeldata:
            print(f"\n--- Data for {row.get('date', 'Unknown Date')} ---")
            for model in models_to_track:
                value = row.get(model)
                if value and value.strip(): # Check if value exists and isn't just whitespace
                    print(f"{model}: {value}")

def process_market_data(path , city_name):
    if not Path(path).exists():
        print(f"Error: The file at {path} was not found.")
        return


    
    try:
        with open(path, 'r') as file:
            full_data = json.load(file)

        markets_list = full_data.get("markets", [])

        print(f"\n--- Market Results for {Path(path).parent.name} ---")
        
        if not markets_list:
            print("No active markets found in this file.")
            return

        for current_market in markets_list:
            # Extracting specific fields with fallbacks
            title = current_market.get("groupItemTitle") or "General Market"
            
            # outcomePrices is usually a stringified JSON list like '["0.30", "0.70"]'
            prices = current_market.get("outcomePrices") or "[No Price Data]"
            
            # Additional useful data for your "Edge" analysis
            description = current_market.get("description") or "No description"
            
            print(f"{title}: {prices}")
            

    except json.JSONDecodeError:
        print("Error: The file is not a valid JSON. It might be empty or corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")








market_file_name = "market.json"
model_run_file_name = "model_runs_Report.csv"

script_location = Path(__file__).resolve().parent
base_path = script_location / "Data"




# --- 2. EXECUTION ---
if __name__ == "__main__":

    while True:
        theDate = input("Enter date (YYYY-MM-DD): ")
        try:
            datetime.strptime(theDate, '%Y-%m-%d')
            break
        except ValueError:
            print("Incorrect format, should be YYYY-MM-DD")

    decision = int(input("Enter 1 or True : if you want data of specific city\nEnter 0 or False : if you want data of all the cities :\n"))

    if decision == 1 :
        while(True):

            city_name = input("Please enter city name : ")
            if(city_name in cities):
                break
            else :
                print("This city is not available. Try another city")
        weather_file_path = base_path / city_name / theDate / market_file_name
        model_run_file_path = base_path /city_name / theDate / model_run_file_name

        process_model_runs(model_run_file_path)
        process_market_data(weather_file_path)
        
    elif decision == 0:

        for city_name in cities:

            weather_file_path = base_path / city_name / theDate / market_file_name
            model_run_file_path = base_path /city_name / theDate / model_run_file_name

            process_model_runs(model_run_file_path , city_name)
            process_market_data(weather_file_path , city_name)

    else :
        print("Undefined input")