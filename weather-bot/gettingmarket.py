from pathlib import Path
from datetime import datetime
import requests
import json

# --- 1. CONFIGURATION & DIRECTORY ANCHORING ---
# This finds the folder where this script is actually saved
script_location = Path(__file__).resolve().parent

# Define your pre-existing folder name here
# It will be located in the same directory as this .py file
BASE_DIR = script_location / "Data"

# Create BASE_DIR if it doesn't exist yet, just to be safe
BASE_DIR.mkdir(parents=True, exist_ok=True)

# def get_file_name(url):
#     """Extracts the date from the slug to create a filename."""
#     try:
#         date_part = url.split("-on-")[-1]
#         return f"{date_part}.json"
#     except IndexError:
#         return "market_data_fallback.json"

def getting_data(url, file_path):
    """Fetches JSON data from the API and saves it to the specific path."""
    print(f"Fetching data from Polymarket...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print("Data successfully saved.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

def get_target_path(url, cities, filename, folder_date_str):
    """Determines the folder structure inside BASE_DIR based on the city."""
    url_lower = url.lower()
    # Matches the city from the list or defaults to 'unknown'
    folder_name = next((c for c in cities if c in url_lower), "unknown")
    
    # Path logic: BASE_DIR / city_folder / filename.json
    return BASE_DIR / folder_name / folder_date_str /filename

# validates weather the date is in correct format or not

def validate_date(date_text):
    try:
        # We return the actual datetime object if it's valid
        return datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        print("Incorrect format, should be YYYY-MM-DD")
        return None # Return None instead of False
    
def allcode(theDate, cities, city_name, folder_date_str):
    url = f"https://gamma-api.polymarket.com/events/slug/highest-temperature-in-{city_name}-on-{theDate}"

    print(url)
    # Generate paths
    filename = "market.json"
    file_path = get_target_path(url, cities, filename, folder_date_str)

    # Ensure the city-specific subfolder exists inside BASE_DIRs
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch and Save
    getting_data(url, file_path)

    print(f"Absolute Path Used: {file_path.resolve()}")

    # --- 3. DATA PROCESSING ---
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)

    market = data.get("markets", [])

    # print("\n--- Market Results ---")
    # for current_market in market:
    #     # We use .get() with fallbacks to avoid NoneType errors during printing
    #     title = current_market.get("groupItemTitle") or "General Market"
    #     prices = current_market.get("outcomePrices") or "[No Price Data]"
        
    #     print(f"{title}: {prices}")
    # else:
    #     print("File was not created. Check your URL or permissions.")

# --- 2. EXECUTION ---
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

#45



while True:
    raw_input = input("Enter date (YYYY-MM-DD): ")
    date_obj = validate_date(raw_input)

    if date_obj: # If it's not None, the date is valid
        break

folder_date_str = date_obj.strftime("%Y-%m-%d")
theDate = date_obj.strftime("%B-%d-%Y").lower()

decision = int(input("Enter 1 or True : if you want data of specific city\nEnter 0 or False : if you want data of all the cities :\n"))

if decision == 1 :
    while(True):

        city_name = input("Please enter city name : ")
        if(city_name in cities):
            break
        else :
            print("This city is not available. Try another city")
    
    allcode(theDate, cities, city_name, folder_date_str)


elif decision == 0:
    for city_name in cities:
        allcode(theDate, cities, city_name, folder_date_str)

else :
    print("Undefined input")