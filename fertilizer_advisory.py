from data_loader import load_kaggle_csv
import pandas as pd

# Load fertilizer data
fertilizer_data_train , fertilizer_data_test = load_kaggle_csv(
    "competitions/playground-series-s5e6", "train.csv","test.csv")

df = pd.concat([fertilizer_data_train, fertilizer_data_test], ignore_index=True)
print(df.head())
