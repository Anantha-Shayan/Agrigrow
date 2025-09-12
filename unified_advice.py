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
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    # Extract safely
    temp = data.get("main", {}).get("temp", None)
    humidity = data.get("main", {}).get("humidity", None)
    
    # OpenWeather gives rainfall in "rain": {"1h": ..., "3h": ...}
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
    print("Columns from API:", df.columns.tolist())  # 👀 Debug
    
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



# Rule-based filter
def rule_based_filter(crop, weather):
    temp, humidity, rainfall = weather["temperature"], weather["humidity"], weather["rainfall"]
    
    rules = {
        "rice": (20 <= temp <= 35 and rainfall >= 100),
        "maize": (18 <= temp <= 27 and 50 <= rainfall <= 100),
        "jute": (24 <= temp <= 37 and rainfall >= 150),
        "cotton": (21 <= temp <= 30 and 50 <= rainfall <= 100),
        "coconut": (20 <= temp <= 32 and rainfall >= 100),
        "papaya": (22 <= temp <= 30 and 100 <= rainfall <= 150),
        "orange": (15 <= temp <= 29 and 100 <= rainfall <= 120),
        "apple": (8 <= temp <= 22 and rainfall <= 150),
        "muskmelon": (20 <= temp <= 30 and 40 <= rainfall <= 60),
        "watermelon": (20 <= temp <= 30 and 40 <= rainfall <= 60),
        "grapes": (15 <= temp <= 30 and 75 <= rainfall <= 85),
        "mango": (24 <= temp <= 30 and rainfall <= 100),
        "banana": (26 <= temp <= 30 and rainfall >= 100),
        "pomegranate": (18 <= temp <= 35 and 50 <= rainfall <= 100),
        "lentil": (18 <= temp <= 30 and 40 <= rainfall <= 60),
        "blackgram": (25 <= temp <= 35 and 60 <= rainfall <= 80),
        "mungbean": (25 <= temp <= 35 and 60 <= rainfall <= 80),
        "mothbeans": (24 <= temp <= 30 and 50 <= rainfall <= 75),
        "pigeonpeas": (26 <= temp <= 30 and 60 <= rainfall <= 100),
        "kidneybeans": (18 <= temp <= 27 and 60 <= rainfall <= 120),
        "chickpea": (10 <= temp <= 30 and 40 <= rainfall <= 60),
        "coffee": (15 <= temp <= 28 and rainfall >= 150)
    }
    return rules.get(crop, True)


# Advisory Function
def give_advice(user_input, city, state, district):
    """
    Generate unified advice considering soil, weather, and market prices.
    user_input: dict, list, or np.array with soil + weather data
    """

    # Ensure correct feature order (same as model training)
    features_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    # 🔹 Convert user_input to DataFrame
    if isinstance(user_input, dict):
        user_df = pd.DataFrame([user_input], columns=features_order)
    elif isinstance(user_input, (list, np.ndarray)):
        user_df = pd.DataFrame([user_input], columns=features_order)
    elif isinstance(user_input, pd.DataFrame):
        user_df = user_input[features_order]
    else:
        raise ValueError("❌ user_input must be dict, list, np.array, or DataFrame")

    # Scale
    soil_scaled = scaler.transform(user_df)

    # ML prediction
    ml_pred_idx = model.predict(soil_scaled)[0]
    ml_crop = encoder.inverse_transform([ml_pred_idx])[0]

    # Get weather data
    weather_data = get_weather(city)
    temp = weather_data["temp"]
    humidity = weather_data["humidity"]
    rainfall = weather_data["rainfall"]

    # Rule-based filtering
    suitable_crops = rule_based_filter(temp, humidity, rainfall)
    if ml_crop not in suitable_crops:
        print(f"⚠️ Recommended crop {ml_crop} is not suitable under current weather conditions.")
        if suitable_crops:
            ml_crop = suitable_crops[0]

    # Market price integration
    market_prices = fetch_market_prices(state, district, suitable_crops or [ml_crop])
    if not market_prices.empty:
        best_crop_row = market_prices.loc[market_prices["Modal_price"].idxmax()]
        best_crop = best_crop_row["Commodity"]
        best_price = best_crop_row["Modal_price"]
    else:
        best_crop, best_price = ml_crop, "N/A"

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
    advice = give_advice(user_input, city="Bengaluru", state="Karnataka", district="Bengaluru")
    print(advice)
