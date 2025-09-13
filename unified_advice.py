import os
import joblib
import requests
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


def check_crop_suitability(crop, temp, humidity, rainfall):
    rules = {
        "rice": {"temp": (20, 35), "rainfall": (100, float("inf")), "humidity": (60, 85)},
        "maize": {"temp": (18, 27), "rainfall": (50, 100), "humidity": (50, 70)},
        "jute": {"temp": (24, 37), "rainfall": (150, float("inf")), "humidity": (65, 90)},
        "cotton": {"temp": (21, 30), "rainfall": (50, 100), "humidity": (50, 70)},
        "coconut": {"temp": (20, 32), "rainfall": (100, float("inf")), "humidity": (60, 80)},
        "papaya": {"temp": (22, 30), "rainfall": (100, 150), "humidity": (65, 85)},
        "orange": {"temp": (15, 29), "rainfall": (100, 120), "humidity": (50, 70)},
        "apple": {"temp": (8, 22), "rainfall": (0, 150), "humidity": (50, 70)},
        "muskmelon": {"temp": (20, 30), "rainfall": (40, 60), "humidity": (50, 65)},
        "watermelon": {"temp": (20, 30), "rainfall": (40, 60), "humidity": (50, 65)},
        "grapes": {"temp": (15, 30), "rainfall": (75, 85), "humidity": (50, 70)},
        "mango": {"temp": (24, 30), "rainfall": (0, 100), "humidity": (50, 70)},
        "banana": {"temp": (26, 30), "rainfall": (100, float("inf")), "humidity": (70, 90)},
        "pomegranate": {"temp": (18, 35), "rainfall": (50, 100), "humidity": (45, 65)},
        "lentil": {"temp": (18, 30), "rainfall": (40, 60), "humidity": (40, 60)},
        "blackgram": {"temp": (25, 35), "rainfall": (60, 80), "humidity": (50, 70)},
        "mungbean": {"temp": (25, 35), "rainfall": (60, 80), "humidity": (50, 70)},
        "mothbeans": {"temp": (24, 30), "rainfall": (50, 75), "humidity": (40, 60)},
        "pigeonpeas": {"temp": (26, 30), "rainfall": (60, 100), "humidity": (50, 70)},
        "kidneybeans": {"temp": (18, 27), "rainfall": (60, 120), "humidity": (50, 70)},
        "chickpea": {"temp": (10, 30), "rainfall": (40, 60), "humidity": (40, 60)},
        "coffee": {"temp": (15, 28), "rainfall": (150, float("inf")), "humidity": (70, 90)},
    }

    if crop not in rules:
        return f"No rules defined for {crop}"

    limits = rules[crop]
    reasons = []

    if not (limits["temp"][0] <= temp <= limits["temp"][1]):
        reasons.append(f"temperature={temp}°C (expected {limits['temp'][0]}–{limits['temp'][1]}°C)")
    if not (limits["rainfall"][0] <= rainfall <= limits["rainfall"][1]):
        reasons.append(f"rainfall={rainfall}mm (expected {limits['rainfall'][0]}–{limits['rainfall'][1]}mm)")
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
    print(f"\n🌦 Current Weather: Temp={temp}°C, Humidity={humidity}%, Rainfall={rainfall}mm")

    # Check suitability
    suitability_msg = check_crop_suitability(ml_crop, temp, humidity, rainfall)
    print(suitability_msg)

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
        best_crop, best_price = ml_crop, "N/A"

    # Final combined advice
    advice = f"""
    🌱 Based on soil, weather & market:
    - ML Suggested Crop: {ml_crop}
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
    print(advice)
