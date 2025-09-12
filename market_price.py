import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import traceback
import warnings
warnings.filterwarnings("ignore")

# load data
df = pd.DataFrame(pd.read_csv('D:\\python\\SIH\\govdata_mandi.csv'))


"""Pre-Processing"""

df.drop(columns=['Arrival_Date','Min_x0020_Price','Max_x0020_Price'],inplace=True)
# print(df.shape)
# print(df.info())
# print(df.head())
df = df.rename(columns={'Modal_x0020_Price': 'Mode_Price'})
onehot_cols = ["State", "Grade"]
targetenc_cols = ["District", "Commodity", "Variety"]
hash_cols = ["Market"]


# Define transformers
preprocessor = ColumnTransformer(
transformers=[
("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_cols),
("target", ce.TargetEncoder(cols=targetenc_cols), targetenc_cols),
("hash", ce.HashingEncoder(cols=hash_cols, n_components=16), hash_cols),
],
remainder="drop"
)
X , y = df.drop("Mode_Price", axis=1), df["Mode_Price"]


"""Model Selection"""
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

models = {
"LinearRegression": LinearRegression(),
"RandomForest": RandomForestRegressor(),
"XGBRegressor": XGBRegressor(),
"LGBMRegressor": lgb.LGBMRegressor()
}


results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    try:
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline, X, y, cv=5,
            scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
            return_train_score=False
        )

        results[name] = {
            "R2": scores["test_r2"].mean(),
            "MAE": -scores["test_neg_mean_absolute_error"].mean(),
            "RMSE": -scores["test_neg_root_mean_squared_error"].mean()
        }
        print(f"{name} done ✅")

    except Exception as e:
        print(f"❌ {name} failed!")
        traceback.print_exc()


# Convert results to DataFrame
if results:
    results_df = pd.DataFrame(results).T
    print("\nModel Performance (5-fold CV):")
    print(results_df)


# Select best model (based on lowest RMSE)
best_model = results_df.sort_values(by="RMSE").iloc[0]
print("\nBest Model:")
print(best_model)

# -------------------------
# Fit the best model on full data
# -------------------------
best_model_name = results_df.sort_values(by="RMSE").index[0]
best_model_instance = models[best_model_name]

# Create pipeline with preprocessor + best model
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", best_model_instance)
])

# Fit on full dataset
pipeline.fit(X, y)

# -------------------------
# Example: Predict mode price for a sample input
# -------------------------
# Sample input format should match columns of X
sample_input = pd.DataFrame({
    "State": ["Karnataka"],
    "Grade": ["FAQ"],
    "District": ["Bangalore"],
    "Commodity": ["Tomato"],
    "Variety": ["Local"],
    "Market": ["KR Market"]
})

predicted_price = pipeline.predict(sample_input)[0]
print(f"\nPredicted Mode Price: ₹{predicted_price:.2f}")

