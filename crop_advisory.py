import requests
import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# Load trained ML model, scaler, encoder
model = joblib.load("cr_model.pkl")
scaler = joblib.load("cr_scaler.pkl")
encoder = joblib.load("cr_encoder.pkl")


# OpenWeatherMap API Setup
API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not API_KEY:
    raise Exception("API key not found. Please set OPENWEATHER_API_KEY in environment variables.")

CITY = "Bengaluru"

# Current weather
current_url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
current_data = requests.get(current_url).json()

if current_data.get("cod") != 200:
    raise Exception("Error fetching weather data:", current_data)

current_temp = current_data["main"]["temp"]
current_humidity = current_data["main"]["humidity"]
current_rain = current_data.get("rain", {}).get("1h", 0)

print(f"Current Weather → Temp: {current_temp}°C, Humidity: {current_humidity}%, Rainfall: {current_rain}mm")

# Forecast (5-day / 3-hour)
forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
forecast_data = requests.get(forecast_url).json()
if forecast_data.get("cod") != "200":
    raise Exception("Error fetching forecast data:", forecast_data)

# Take average temp & humidity over next 24 hours (first 8 entries = 24h forecast)
temps = [entry["main"]["temp"] for entry in forecast_data["list"][:8]]
humidities = [entry["main"]["humidity"] for entry in forecast_data["list"][:8]]
avg_temp = sum(temps) / len(temps)
avg_humidity = sum(humidities) / len(humidities)
print(f"Forecast Avg (Next 24h) → Temp: {avg_temp:.1f}°C, Humidity: {avg_humidity:.1f}%")


# Example soil input
# (N, P, K, temp, humidity, pH, rainfall)
soil_sample = np.array([[90, 42, 43, avg_temp, avg_humidity, 6.5, current_rain]])
soil_sample_scaled = scaler.transform(soil_sample)


# ML Model Prediction
predicted_crop_num = model.predict(soil_sample_scaled)[0]
predicted_crop = encoder.inverse_transform([predicted_crop_num])[0]
print("ML Suggested Crop:", predicted_crop)


# Rule-based Filter
def rule_based_filter_flexible(predicted_crop, temperature, humidity):
    crops_conditions = {
        "rice": {"temp": (20, 35), "humidity": (70, 90), "rainfall": (1000, 2000)},
        "maize": {"temp": (20, 32), "humidity": (60, 80), "rainfall": (500, 1200)},
        "jute": {"temp": (24, 38), "humidity": (75, 90), "rainfall": (1200, 2000)},
        "cotton": {"temp": (25, 35), "humidity": (50, 70), "rainfall": (500, 1200)},
        "coconut": {"temp": (25, 35), "humidity": (70, 90), "rainfall": (1500, 2500)},
        "papaya": {"temp": (25, 35), "humidity": (70, 90), "rainfall": (1200, 2000)},
        "orange": {"temp": (20, 35), "humidity": (50, 80), "rainfall": (800, 1500)},
        "apple": {"temp": (15, 25), "humidity": (60, 80), "rainfall": (700, 1200)},
        "muskmelon": {"temp": (25, 35), "humidity": (50, 70), "rainfall": (400, 800)},
        "watermelon": {"temp": (25, 35), "humidity": (50, 70), "rainfall": (400, 800)},
        "grapes": {"temp": (20, 30), "humidity": (50, 70), "rainfall": (600, 1200)},
        "mango": {"temp": (25, 35), "humidity": (60, 80), "rainfall": (750, 1200)},
        "banana": {"temp": (25, 35), "humidity": (70, 90), "rainfall": (1500, 2500)},
        "pomegranate": {"temp": (25, 35), "humidity": (40, 70), "rainfall": (400, 800)},
        "lentil": {"temp": (15, 25), "humidity": (50, 70), "rainfall": (400, 700)},
        "blackgram": {"temp": (25, 35), "humidity": (50, 80), "rainfall": (400, 1000)},
        "mungbean": {"temp": (25, 35), "humidity": (50, 80), "rainfall": (400, 1000)},
        "mothbeans": {"temp": (28, 38), "humidity": (30, 60), "rainfall": (200, 600)},
        "pigeonpeas": {"temp": (25, 35), "humidity": (50, 70), "rainfall": (600, 1000)},
        "kidneybeans": {"temp": (20, 30), "humidity": (50, 70), "rainfall": (600, 1200)},
        "chickpea": {"temp": (20, 30), "humidity": (40, 70), "rainfall": (400, 700)},
        "coffee": {"temp": (18, 25), "humidity": (70, 90), "rainfall": (1200, 2500)}
    }
    
    crop = predicted_crop.lower()
    if crop not in crops_conditions:
        return f"Recommended Crop: {predicted_crop}"

    cond = crops_conditions[crop]
    temp_ok = cond["temp"][0] <= temperature <= cond["temp"][1]
    humidity_ok = cond["humidity"][0] <= humidity <= cond["humidity"][1]

    if temp_ok and humidity_ok:
        return f"Recommended Crop: {predicted_crop}"
    else:
        return f"Predicted crop is {predicted_crop} which may not be ideal for upcoming weather. Consider alternatives."

final_advice = rule_based_filter_flexible(predicted_crop, avg_temp, avg_humidity)
print(final_advice)
