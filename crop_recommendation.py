from data_loader import load_kaggle_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load dataset
df = load_kaggle_csv(
    "atharvaingle/crop-recommendation-dataset",
    "Crop_recommendation.csv"
)
# print(df.head())
print(df['label'].value_counts())
print(df['label'].nunique())
# print(df.shape)
# print(df.isnull().sum())
# print(df.info())
# print(df.describe())
"""Preprocessing"""
X_train, X_test, y_train, y_test = train_test_split(df.drop("label", axis=1), df["label"], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

joblib.dump(encoder, "cr_encoder.pkl")
joblib.dump(scaler, "cr_scaler.pkl")


"""MODEL SELECTION"""
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

b=BernoulliNB()
s=LinearSVC()
k=KNeighborsClassifier()
D=DecisionTreeClassifier()
R=RandomForestClassifier()
Log=LogisticRegression()
XGB=XGBClassifier()
G=GradientBoostingClassifier()

models=[b,s,k,D,R,Log,XGB,G]

accuracy=[]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__}: {accuracy_score(y_test, y_pred):.4f}")
    accuracy.append(accuracy_score(y_test, y_pred))

model = models[accuracy.index(max(accuracy))]
joblib.dump(model, "cr_model.pkl")
print('-'*50)
print('Model used: ', model.__class__.__name__)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"{model.__class__.__name__}: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print('-'*50)

# pred_df = pd.DataFrame({'N': [60],'P': [47],'K': [22],'T': [19.8652524],'H': [60.1923008],'PH': [5.953933276],'R': [100.55501690000003]})
# pred_df['Predicted'] = model.predict(pred_df)

# classes = {0: 'Rice', 1: 'Wheat', 2: 'Maize', 3: 'Barley', 4: 'Soybean', 5: 'RiceCotton', 6: 'Cotton', 7: 'Pumpkin',
#            8: 'Tomato', 9: 'Mango', 10: 'Banana', 11: 'Orange', 12: 'Sugarcane', 13: 'Coconut', 14: 'Papaya', 15: 'Grapes', 16: 'Apple', 17: 'Orange', 18: 'Pomegranate', 19: 'Litchi', 20: 'Cherry', 21: 'Plum', 22: 'Pear', 23: 'Peach', 24: 'Watermelon', 25: 'Muskmelon', 26: 'Kiwi', 27: 'Blueberry', 28: 'Raspberry', 29: 'Strawberry', 30: 'Blackberry', 31: 'Cherry'}
# print('The crop is ', classes[pred_df['Predicted'][0]])