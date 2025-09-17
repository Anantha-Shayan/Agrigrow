
# Agrigrow 
 Farmer's friend

---

# 🌾 Smart Crop Advisory

## 📌 Overview

This project is a **Smart Crop Advisory System (MVP)** designed to help farmers make informed decisions about which crops to grow.
The system integrates **Machine Learning, weather data, location-specific suitability, and mandi (market) price data** to recommend the most profitable and feasible crop for a farmer.

Instead of relying only on **yield prediction**, the MVP combines multiple real-world constraints like climate, soil, and market economics into a single workflow.

---

## ⚙️ Workflow

1. **ML Crop Recommendation**

   * A trained ML model suggests suitable crops based on inputs like soil, rainfall, and historical patterns.

2. **Weather Check**

   * The suggested crops are cross-validated against **current weather conditions** (e.g., temperature, rainfall, humidity).
   * Ensures that crops unsuitable for immediate weather are filtered out.

3. **Location Suitability (ICRISAT Data)**

   * Uses **district-wise crop production data (ICRISAT)** to ensure the recommended crop is historically grown in the farmer’s district/state.
   * Prevents impractical suggestions like recommending apples in Bengaluru.

4. **Market Price Check (Mandi Data)**

   * Among the filtered crops, checks availability of **market prices from Agmarknet (mandi data)**.
   * Selects the crop with the **highest market price** in the farmer’s nearest mandi.

5. **Final Recommendation**

   * Outputs **one or a few top crops** that balance feasibility, suitability, and profitability.

---

## 📊 Data Sources

* **ICRISAT District-Level Crop Data**
  Used for mapping which crops are historically suitable in a given district/state.
  (Processed into `district_crop_map.json` and `crop_district_map.json` for fast lookups).

* **Weather Data**
  Used to filter out crops not suitable under current climate conditions.

* **Agmarknet (Mandi Prices)**
  Used to determine real-time economic value of crops.

---

## 🚀 Features in MVP

* ✅ ML-based crop recommendation.
* ✅ Weather validation to avoid impractical crops.
* ✅ Location-specific suitability via ICRISAT data.
* ✅ Market-driven selection for profitability.
* ✅ JSON-based lookup (`district_crop_map.json`, `crop_district_map.json`).

---

## 🔮 Future Improvements

* Support for crop disease/pest detection using CNN.
* Multi-lingual chatbot.
* Multi-lingual farmer-friendly app interface.
* Weather forecast and alerts.
* Access for mandi price in every market for every crop
* Market demand prediction (not just current price).
* Soil data integration (pH, fertility).
