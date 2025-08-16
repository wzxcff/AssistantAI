import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

user_info = {"city": None}
weather_timestamp = {}
weather_cache = {}


def fetch_weather(city):
    if city in weather_timestamp and city in weather_cache:
        if datetime.now() - weather_timestamp[city] < timedelta(minutes=15):
            print(f"Using cached result from {round((datetime.now() - weather_timestamp[city]).total_seconds() // 60)} minutes ago")
            return weather_cache[city]

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv('OPENWEATHER_API_KEY')}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        user_response = f"Weather for {city}:\n – {data['weather'][0]['main']}\n – Temp: {round(data["main"]["temp"])}\n – Feels like: {round(data["main"]['feels_like'])}\n – Humidity: {data['main']['humidity']}%"
        weather_timestamp[city] = datetime.now()
        weather_cache[city] = user_response
        return user_response
    else:
        return "Cannot fetch weather"


def weather():
    if user_info["city"]:
        print(f"I remember that last time you used: {user_info["city"]}, should I use it for this result?")
        decision = input("y/n: ").lower()
        if decision == "y":
            return fetch_weather(user_info["city"])

    print("Tell me a place, I will tell you a weather.")
    city = input("City: ").lower().capitalize()
    user_info["city"] = city

    return fetch_weather(city)


def reminder():
    print("Sure, I got you covered.\nAbout what do I need to remind you?")
    topic = input("Topic: ")
    print("For what time do I set the reminder?")
    time = input("Time: ")
    return f"\nSo that's your reminder:\nTopic: {topic}\nTime: {time}"


def math():
    print("What do I need to solve?")
    problem = input("Problem: ")
    return f"Here's your answer: {eval(problem)}"


