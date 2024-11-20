# Visual Crossing API

import requests
import pandas as pd

def extract_weather_data():
    try:
        # Define the new URL with the updated date range
        url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/bakersfield%2C%20california/2023-01-01/2024-11-19"
        
        # Parameters for the request
        params = {
            "unitGroup": "metric",  # Unit for temperature in Celsius
            "include": "days",  # Include daily weather data
            "key": "WRDAEDX9SGX2GDFQ2SDZM5JHN",  # API key
            "contentType": "json"  # Request data in JSON format
        }

        # Make the request to the API
        response = requests.get(url, params=params)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the JSON response to extract the weather data
        data = response.json()
        
        if "days" in data:
            weather_data = data["days"]
            
            # Create DataFrame from the extracted weather data
            df = pd.DataFrame(weather_data)
            
            # Selecting relevant columns for transformation (adjust columns as needed)
            df = df[["datetime", "temp", "dew", "humidity", "precip", "windspeed", "conditions"]]
            
            # Convert the 'datetime' column to proper date-time format
            df["datetime"] = pd.to_datetime(df["datetime"])
            
            # Save the data to a CSV file
            output_file = "weather_bakersfield_2023_2024.csv"
            df.to_csv(output_file, index=False)
            
            print(f"Data successfully saved to {output_file}")
        else:
            print("No weather data available for the specified dates.")
    
    except requests.RequestException as e:
        print(f"Error during API request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Call the function to extract and save weather data
extract_weather_data()
