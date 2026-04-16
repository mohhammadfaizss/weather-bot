import json
import csv
from pathlib import Path
from datetime import datetime



class data_processing:


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

    market_file_name = "market.json"
    model_run_file_name = "model_runs_Report.csv"
    json_file_name = "processed_data.json"

    script_location = Path(__file__).resolve().parent
    base_path = script_location / "Data"
    
    def save_to_master_json(self, city_name, key_name, data_to_add):
        """A helper function to handle any number of sources dynamically"""
        json_file_path = self.base_path / city_name / self.theDate / self.json_file_name
        
        # Ensure directory exists
        json_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        master_data = {}
        if json_file_path.exists():
            with open(json_file_path, 'r', encoding='utf-8') as f:
                try:
                    master_data = json.load(f)
                except json.JSONDecodeError:
                    master_data = {}

        # Update only the specific key
        master_data[key_name] = data_to_add

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=4)
        print(f"✅ Key '{key_name}' updated in {city_name} master JSON.")

    def process_model_runs(self, path, city_name):
        """Processes CSV model data"""
        if not Path(path).exists():
            print(f"Error: Model file at {path} not found.")
            return
        
        csv_data = []
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                csv_data.append(row)

        self.save_to_master_json(city_name, "weather_models", csv_data)

    def process_market_data(self, path, city_name):
        """Processes JSON market data"""
        if not Path(path).exists():
            print(f"Error: Market file at {path} not found.")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as file:
                full_data = json.load(file)

            markets_list = full_data.get("markets", [])
            market_results = []
            
            for item in markets_list:
                # Explicitly map keys to ensure we aren't just copying the whole object
                entry = {
                    "title": item.get("groupItemTitle") or "General Market",
                    "prices": item.get("outcomePrices") or "[No Price Data]",
                    "question": item.get("question") or "No question",
                    "resolutionSource": item.get("resolutionSource") or "No resolution source"
                }
                market_results.append(entry)

            self.save_to_master_json(city_name, "market_data", market_results)

        except Exception as e:
            print(f"An unexpected error occurred processing market data: {e}")



    def run(self):
        while True:
            self.theDate = input("Enter date (YYYY-MM-DD): ")
            try:
                datetime.strptime(self.theDate, '%Y-%m-%d')
                break
            except ValueError:
                print("Incorrect format, should be YYYY-MM-DD")

            
        decision = int(input("Enter 1 or True : if you want data of specific city\nEnter 0 or False : if you want data of all the cities :\n"))

        if decision == 1 :
            while(True):

                city_name = input("Please enter city name : ")
                if(city_name in self.cities):
                    break
                else :
                    print("This city is not available. Try another city")
            weather_file_path = self.base_path / city_name / self.theDate / self.market_file_name
            model_run_file_path = self.base_path /city_name / self.theDate / self.model_run_file_name

            self.process_model_runs(path=model_run_file_path, city_name=city_name)
            self.process_market_data(path=weather_file_path, city_name=city_name)
                
        elif decision == 0:

            for city_name in self.cities:

                weather_file_path = self.base_path / city_name / self.theDate / self.market_file_name
                model_run_file_path = self.base_path /city_name / self.theDate / self.model_run_file_name

                self.process_model_runs(path=model_run_file_path, city_name=city_name)
                self.process_market_data(path=weather_file_path, city_name=city_name)

        else :
            print("Undefined input")
    





# --- 2. EXECUTION ---
if __name__ == "__main__":

    processing = data_processing()
    processing.run()
    