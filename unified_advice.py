import os
import joblib
import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load trained ML model + encoders
model = joblib.load("cr_model.pkl")  # trained crop recommendation model
scaler = joblib.load("cr_scaler.pkl")
encoder = joblib.load("cr_encoder.pkl")


# Fetch weather (OpenWeatherMap API)
def get_weather(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    #print(data)
    if response.status_code != 200 or "main" not in data:
        raise ValueError(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
    
    temp = data.get("main", {}).get("temp", None)
    humidity = data.get("main", {}).get("humidity", None)
    
    rainfall = None
    if "rain" in data:
        rainfall = data["rain"].get("1h") or data["rain"].get("3h")

    return {
        "temp": temp,
        "humidity": humidity,
        "rainfall": rainfall if rainfall is not None else 0
    }


# Fetch market prices (Mandi API)
def fetch_market_prices(state, district, crops, date=None):
    api_key = os.getenv("MANDI_PRICE_KEY")
    if not api_key:
        raise ValueError("MANDI_PRICE_KEY not found in environment variables.")

    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 1000
    }
    if date:
        params["filters[arrival_date]"] = date
    if state:
        params["filters[state]"] = state
    if district:
        params["filters[district]"] = district
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data["records"])
    #print("Columns from API:", df.columns.tolist())
    
    # Normalize column names
    df = df.rename(columns={
        "Modal_x0020_Price": "modal_price",
        "modal_price": "modal_price"
    })
    
    if "modal_price" not in df.columns:
        raise KeyError(f"'modal_price' column not found. Got: {df.columns.tolist()}")
    
    df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
    
    # Filter crops
    df["commodity"] = df["commodity"].str.lower()
    crops = [c.lower() for c in crops]
    df = df[df["commodity"].isin(crops)]
    prices = df.groupby("commodity")["modal_price"].mean().to_dict()
    
    return prices

# Weather-based crop requirements
crop_requirements = {
    "rice": {"temp": (20, 35), "humidity": (70, 90), "rainfall": (100, float("inf"))},
    "maize": {"temp": (18, 27), "humidity": (50, 70), "rainfall": (50, 100)},
    "jute": {"temp": (24, 37), "humidity": (70, 90), "rainfall": (150, float("inf"))},
    "cotton": {"temp": (21, 30), "humidity": (60, 80), "rainfall": (50, 100)},
    "coconut": {"temp": (20, 32), "humidity": (70, 90), "rainfall": (100, float("inf"))},
    "papaya": {"temp": (22, 30), "humidity": (60, 85), "rainfall": (100, 150)},
    "orange": {"temp": (15, 29), "humidity": (50, 70), "rainfall": (100, 120)},
    "apple": {"temp": (8, 22), "humidity": (30, 60), "rainfall": (0, 150)},
    "muskmelon": {"temp": (20, 30), "humidity": (50, 70), "rainfall": (40, 60)},
    "watermelon": {"temp": (20, 30), "humidity": (50, 70), "rainfall": (40, 60)},
    "grapes": {"temp": (15, 30), "humidity": (50, 70), "rainfall": (75, 85)},
    "mango": {"temp": (24, 30), "humidity": (50, 70), "rainfall": (0, 100)},
    "banana": {"temp": (26, 30), "humidity": (70, 90), "rainfall": (100, float("inf"))},
    "pomegranate": {"temp": (18, 35), "humidity": (40, 60), "rainfall": (50, 100)},
    "lentil": {"temp": (18, 30), "humidity": (40, 60), "rainfall": (40, 60)},
    "blackgram": {"temp": (25, 35), "humidity": (60, 80), "rainfall": (60, 80)},
    "mungbean": {"temp": (25, 35), "humidity": (60, 80), "rainfall": (60, 80)},
    "mothbeans": {"temp": (24, 30), "humidity": (50, 70), "rainfall": (50, 75)},
    "pigeonpeas": {"temp": (26, 30), "humidity": (60, 80), "rainfall": (60, 100)},
    "kidneybeans": {"temp": (18, 27), "humidity": (50, 70), "rainfall": (60, 120)},
    "chickpea": {"temp": (10, 30), "humidity": (40, 60), "rainfall": (40, 60)},
    "coffee": {"temp": (15, 28), "humidity": (70, 90), "rainfall": (150, float("inf"))},
}


def check_crop_suitability(crop, temp, humidity, rainfall):
    if crop not in crop_requirements:
        return f"No rules defined for {crop}"

    limits = crop_requirements[crop]
    reasons = []

    if not (limits["temp"][0] <= temp <= limits["temp"][1]):
        reasons.append(f"temperature={temp}°C (expected {limits['temp'][0]}–{limits['temp'][1]}°C)")
    if not (limits["rainfall"][0] <= rainfall <= limits["rainfall"][1]):
        reasons.append(f"rainfall={rainfall}mm (expected {limits['rainfall'][0]}–{limits['rainfall'][1]} mm)")
    if not (limits["humidity"][0] <= humidity <= limits["humidity"][1]):
        reasons.append(f"humidity={humidity}% (expected {limits['humidity'][0]}–{limits['humidity'][1]}%)")

    if reasons:
        return f"⚠️ Recommended crop {crop} is not suitable due to: {', '.join(reasons)}"
    else:
        return f"✅ Recommended crop {crop} is suitable under current weather conditions."


def give_advice(user_input, state, district):
    # Ensure correct feature order (same as model training)
    features_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    # Convert user_input to DataFrame
    if isinstance(user_input, dict):
        user_df = pd.DataFrame([user_input], columns=features_order)
    elif isinstance(user_input, (list, np.ndarray)):
        user_df = pd.DataFrame([user_input], columns=features_order)
    elif isinstance(user_input, pd.DataFrame):
        user_df = user_input[features_order]
    else:
        raise ValueError("❌ user_input must be dict, list, np.array, or DataFrame")

    # Scale soil data
    soil_scaled = scaler.transform(user_df)

    # Predict ML suggested crop
    ml_pred_idx = model.predict(soil_scaled)[0]
    ml_crop = encoder.inverse_transform([ml_pred_idx])[0]

    # Fetch weather data
    weather_data = get_weather(district)
    temp = weather_data["temp"]
    humidity = weather_data["humidity"]
    rainfall = weather_data["rainfall"]

    # Print current weather
    print('-'*50)
    print(f"\n🌦 Current Weather: Temp={temp}°C, Humidity={humidity}%, Rainfall={rainfall}mm\n")
    print('-'*50)
    print("\nCrop Recommended based on Soil:", ml_crop)

    # Check suitability of ML crop
    suitability_msg = check_crop_suitability(ml_crop, temp, humidity, rainfall)
    print(suitability_msg)

    # If ML crop unsuitable → suggest nearest suitable crop
    if "not suitable" in suitability_msg:
        print("\n🔄 Finding nearest suitable alternative crop...")
        time.sleep(3)
        nearest_crop = None
        min_diff = float("inf")

        for crop, limits in crop_requirements.items():
            # Compute difference for each weather parameter
            diff = 0
            if temp < limits["temp"][0]:
                diff += limits["temp"][0] - temp
            elif temp > limits["temp"][1]:
                diff += temp - limits["temp"][1]

            if humidity < limits["humidity"][0]:
                diff += limits["humidity"][0] - humidity
            elif humidity > limits["humidity"][1]:
                diff += humidity - limits["humidity"][1]

            if rainfall < limits["rainfall"][0]:
                diff += limits["rainfall"][0] - rainfall
            elif rainfall > limits["rainfall"][1]:
                diff += rainfall - limits["rainfall"][1]

            # Update nearest crop if this one is closer
            if diff < min_diff:
                min_diff = diff
                nearest_crop = crop

        final_crop = nearest_crop if nearest_crop else ml_crop
    else:
        final_crop = ml_crop

    # Fetch market prices
    market_prices = fetch_market_prices(
        state, district,
        ["Brinjal", "Tomato", "Raddish"],
        date=datetime.now().strftime("%Y-%m-%d")
    )

    if market_prices:
        best_crop = max(market_prices, key=market_prices.get)
        best_price = market_prices[best_crop]
    else:
        best_crop, best_price = final_crop, "N/A"

    # Final combined advice
    advice = f"""
    \n🌱 Based on soil, weather & market:
    - ML Suggested Crop: {ml_crop}
    - Final Recommended Crop: {final_crop}
    - Best Market Crop: {best_crop} (₹{best_price})
    """

    return advice.strip()


# Example usage
if __name__ == "__main__":
    user_input = {
        "N": 90, "P": 42, "K": 43,
        "temperature": 25, "humidity": 80,
        "ph": 6.5, "rainfall": 120
    }
    advice = give_advice(user_input, state="Karnataka", district="Bangalore")
    print('-'*50)
    print(advice)
    print('-'*50)
