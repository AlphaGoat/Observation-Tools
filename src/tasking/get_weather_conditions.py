"""
Utilities for retrieving weather conditions at observation sites at given times.

Author: Peter Thomas
Date: 2025-10-13
"""
import requests
from datetime import datetime, timedelta


def get_weather_conditions(api_key: str, lat: float, lon: float, observation_time_utc: str) -> dict:
    """
    Retrieve weather conditions for a given location and time using the OpenWeatherMap API.

    Parameters:
    api_key (str): API key for OpenWeatherMap.
    lat (float): Latitude of the observation site.
    lon (float): Longitude of the observation site.
    observation_time_utc (str): Observation time in UTC (ISO format).

    Returns:
    dict: Weather conditions including temperature, humidity, wind speed, and cloud cover.
    """
    # Convert observation time to a datetime object
    obs_time = datetime.fromisoformat(observation_time_utc)

    # OpenWeatherMap provides historical weather data for paid accounts.
    # For free accounts, we can only get current weather data.
    # Here, we will fetch current weather data as a placeholder.
    
    url = f"http://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={api_key}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching weather data: {response.status_code}, {response.text}")
    
    weather_data = response.json()
    
    # Extract relevant weather information
    weather_conditions = {

        "temperature_celsius": weather_data["current"]["temp"],
        "humidity_percent": weather_data["current"]["humidity"],
        "wind_speed_mps": weather_data["current"]["wind_speed"],
        "wind_gust_mps": weather_data["current"].get("wind_gust", 0.),
        "cloud_cover_percent": weather_data["current"]["clouds"],
        "observation_time_utc": obs_time.isoformat()
    }
    
    return weather_conditions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrieve weather conditions for a given location and time.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for OpenWeatherMap.")
    parser.add_argument("--sensor", type=str, required=True, help="Path to sensors configuration file.")
    parser.add_argument("--observation_time_utc", type=str, required=True, help="Observation time in UTC (ISO format).")
    args = parser.parse_args()

    import json
    with open(args.sensor, 'r') as f:
        sensors = json.load(f)

    for sensor in sensors:
        lat = sensor["latitude"]
        lon = sensor["longitude"]
        weather = get_weather_conditions(args.api_key, lat, lon, args.observation_time_utc)
        print(f"Sensor: {sensor['name']}")
        print(weather)