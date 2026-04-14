# import requests

# url = "https://gamma-api.polymarket.com/events/slug/cricipl-mum-kol-2026-03-29"

# response = requests.get(url)

# print(response.text)

# import requests
# import json

# # Your Polymarket CLOB URL
# url = "https://clob.polymarket.com/price?token_id=9243859686424154476031393325319635706552449918713151700199742912076304706244"

# try:
#     # 1. Fetch the data
#     response = requests.get(url)
#     response.raise_for_status() # Check for HTTP errors

#     # 2. Convert response text to a Python Dictionary
#     data = response.json()

#     # 3. Save to a file with "Pretty Print" formatting
#     with open("market_price.json", "w") as f:
#         # indent=4 creates the newlines and tabs you want
#         json.dump(data, f, indent=4)

#     print("Success! Data stored in market_price.json")

# except Exception as e:
#     print(f"An error occurred: {e}")

# import requests
# import json
# from pathlib import Path

# # 1. The URL for the specific weather event
# url = "https://gamma-api.polymarket.com/events/slug/highest-temperature-in-madrid-on-march-30-2026"


# #finding the index of the data. We do this by finding the index of the last substring before our date


# date_part = url.split("-on-")[-1]
# filename = f"{date_part}.json"

# cityName = ["singapore", "madrid"]
# foundURl = (city for city in cityName if city in url)
# FilePath = Path(foundURl) / filename





# # 2. Fetch the data from Polymarket
# response = requests.get(url)

# # # # 3. Convert the raw text into a Python Dictionary (JSON object)
# data = response.json()

# # # # 4. Open a file and "dump" the dictionary into it
# with open( FilePath , "w") as f:
#     json.dump(data, f, indent=4)

# print("Data successfully saved to " + f"{FilePath}")

# # print(url[29:])


# from pathlib import Path
# import requests
# import json

# def get_file_name(url):
#     date_part = url.split("-on-")[-1]
#     filename = f"{date_part}.json"
#     return filename



# def get_target_path(url, cities, filename):
#     """Finds a city match in a URL and returns a organized Path object."""
#     url_lower = url.lower()
    
    
#     folder_name = next((c for c in cities if c in url_lower), "unknown")
    
    
#     return Path(folder_name) / filename

# def getting_data(url, file_path, filename):
#     response = requests.get(url)
#     data = response.json()

#     with open( file_path , "w") as f:
#         json.dump(data, f, indent=4)

# url = "https://gamma-api.polymarket.com/events/slug/highest-temperature-in-madrid-on-march-30-2026"
# cities = ["singapore", "madrid", "london"]
# filename = get_file_name(url)

# file_path = get_target_path(url, cities, filename)

# file_path.parent.mkdir(parents=True, exist_ok=True)

# getting_data(url, file_path, filename)

# print(f"Data saved to: {file_path}")

# from pathlib import Path
# import requests
# import json

# def get_file_name(url):
#     date_part = url.split("-on-")[-1]
#     return f"{date_part}.json"

# def getting_data(url, file_path): # Removed unnecessary filename arg
#     response = requests.get(url)
#     data = response.json()
    
#     # Now this will work because the folder exists!
#     with open(file_path, "w") as f:
#         json.dump(data, f, indent=4)

# def get_target_path(url, cities, filename):
#     url_lower = url.lower()
#     folder_name = next((c for c in cities if c in url_lower), "unknown")
#     return Path(folder_name) / filename

# # --- Execution ---
# url = "https://gamma-api.polymarket.com/events/slug/highest-temperature-in-madrid-on-april-14-2026"
# cities = ["singapore", "madrid", "london", "lucknow"]

# filename = get_file_name(url)
# file_path = get_target_path(url, cities, filename)

# # STEP 1: Create the folder FIRST
# file_path.parent.mkdir(parents=True, exist_ok=True)

# # STEP 2: Now it is safe to fetch and save data
# getting_data(url, file_path)

# print(f"Data saved to: {file_path}")


# with open(file_path, "r") as f:
#     data = json.load(f)

# market = data.get("markets", [])

# for current_market in market:
#     # Use the variable defined in the 'for' line
#     print(current_market.get("groupItemTitle") +" " + current_market.get("outcomePrices"))

# # first_market = market[0]

# # print(first_market.get("outcomePrices"))


from pathlib import Path
from datetime import datetime
import requests
import json

# --- 1. CONFIGURATION & DIRECTORY ANCHORING ---
# This finds the folder where this script is actually saved
script_location = Path(__file__).resolve().parent

# Define your pre-existing folder name here
# It will be located in the same directory as this .py file
BASE_DIR = script_location / "market-data"

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

def get_target_path(url, cities, filename):
    """Determines the folder structure inside BASE_DIR based on the city."""
    url_lower = url.lower()
    # Matches the city from the list or defaults to 'unknown'
    folder_name = next((c for c in cities if c in url_lower), "unknown")
    
    # Path logic: BASE_DIR / city_folder / filename.json
    return BASE_DIR / folder_name / filename

# validates weather the date is in correct format or not

def validate_date(date_text):
    try:
        # We return the actual datetime object if it's valid
        return datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        print("Incorrect format, should be YYYY-MM-DD")
        return None # Return None instead of False
    
def allcode(theDate, cities, city_name):
    url = f"https://gamma-api.polymarket.com/events/slug/highest-temperature-in-{city_name}-on-{theDate}"

    print(url)
    # Generate paths
    filename = f"{theDate}"
    file_path = get_target_path(url, cities, filename)

    # Ensure the city-specific subfolder exists inside BASE_DIR
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch and Save
    getting_data(url, file_path)

    print(f"Absolute Path Used: {file_path.resolve()}")

    # --- 3. DATA PROCESSING ---
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)

    market = data.get("markets", [])

    print("\n--- Market Results ---")
    for current_market in market:
        # We use .get() with fallbacks to avoid NoneType errors during printing
        title = current_market.get("groupItemTitle") or "General Market"
        prices = current_market.get("outcomePrices") or "[No Price Data]"
        
        print(f"{title}: {prices}")
    else:
        print("File was not created. Check your URL or permissions.")

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

theDate = date_obj.strftime("%B-%d-%Y").lower()

decision = int(input("Enter 1 or True : if you want data of specific city\nEnter 0 or False : if you want data of all the cities :\n"))

if decision == 1 :
    while(True):

        city_name = input("Please enter city name : ")
        if(city_name in cities):
            break
        else :
            print("This city is not available. Try another city")
    
    allcode(theDate, cities, city_name)


elif decision == 0:
    for city_name in cities:
        allcode(theDate, cities, city_name)

else :
    print("Undefined input")

# url = f"https://gamma-api.polymarket.com/events/slug/highest-temperature-in-{city_name}-on-{theDate}"

# print(url)
# # Generate paths
# filename = f"{theDate}"
# file_path = get_target_path(url, cities, filename)

# # Ensure the city-specific subfolder exists inside BASE_DIR
# file_path.parent.mkdir(parents=True, exist_ok=True)

# # Fetch and Save
# getting_data(url, file_path)

# print(f"Absolute Path Used: {file_path.resolve()}")

# # --- 3. DATA PROCESSING ---
# if file_path.exists():
#     with open(file_path, "r") as f:
#         data = json.load(f)

#     market = data.get("markets", [])

#     print("\n--- Market Results ---")
#     for current_market in market:
#         # We use .get() with fallbacks to avoid NoneType errors during printing
#         title = current_market.get("groupItemTitle") or "General Market"
#         prices = current_market.get("outcomePrices") or "[No Price Data]"
        
#         print(f"{title}: {prices}")
# else:
#     print("File was not created. Check your URL or permissions.")