import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data from Excel file
file_path = "Battery data all temperatures.xlsx"
ocv_data = pd.read_excel(file_path, sheet_name="OCV")

# Preprocess OCV data
def preprocess_ocv_data(df):
    X = df[['OCV (battery voltage)', 'temp (deg)', 'mode']].values
    y = df['SOC'].values
    
    # Encode the 'mode' column
    label_encoder = LabelEncoder()
    X[:, 2] = label_encoder.fit_transform(X[:, 2])
    
    # Convert all values to float
    X = X.astype(float)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Split OCV data into features and target
ocv_X, ocv_y = preprocess_ocv_data(ocv_data)

# Normalize features
scaler = StandardScaler()
ocv_X_scaled = scaler.fit_transform(ocv_X)

# Split OCV data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(ocv_X_scaled, ocv_y, test_size=0.2, random_state=42)

# Reshape the input data to include a time step dimension
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

# Define LSTM model
model = Sequential([
    LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Predict SOC for OCV data
ocv_soc_predictions = model.predict(X_val)

# Save the trained model
model_path = "battery_model.h5"
model.save(model_path)
print("Model saved successfully at:", model_path)
