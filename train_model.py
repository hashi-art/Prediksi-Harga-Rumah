import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

data = {
    'sqft': [1500, 2000, 1200, 2500, 1800, 1350, 3000, 2200, 1600, 2800],
    'bedrooms': [3, 4, 2, 4, 3, 2, 5, 3, 3, 4],
    'bathrooms': [2, 3, 1, 3, 2, 1, 4, 2, 2, 3],
    'price': [300000, 400000, 250000, 500000, 350000, 280000, 600000, 420000, 320000, 550000]
}
df = pd.DataFrame(data)

X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'house_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model berhasil dilatih dan disimpan!")