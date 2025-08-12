# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("iris.csv")  # Make sure iris.csv is in the same folder

# Prepare data
X = df.drop("species", axis=1)
y = df["species"]

# Convert labels to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'iris_model.pkl')

# Save label encoder (optional)
joblib.dump(le, 'label_encoder.pkl')
