import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data from Excel file
file_path = "Battery data all temperatures.xlsx"
ocv_data = pd.read_excel(file_path, sheet_name="OCV")
ddpt_15_data = pd.read_excel(file_path, sheet_name="DDPT_15deg")
ddpt_25_data = pd.read_excel(file_path, sheet_name="DDPT_25deg")
ddpt_45_data = pd.read_excel(file_path, sheet_name="DDPT_45deg")

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

# Preprocess DDPT data
def preprocess_ddpt_data(df):
    # Assuming the column containing time values is named differently
    X = df[['Time (Seconds)', 'Voltage (mV)', 'Current (mA)', 'mode']].values
    
    # Encode the 'mode' column
    label_encoder = LabelEncoder()
    X[:, 3] = label_encoder.fit_transform(X[:, 3])
    
    # Convert all values to float
    X = X.astype(float)
    
    return X

# Split OCV data into features and target
ocv_X, ocv_y = preprocess_ocv_data(ocv_data)

# Split DDPT data into features and target
ddpt_15_X = preprocess_ddpt_data(ddpt_15_data)
ddpt_25_X = preprocess_ddpt_data(ddpt_25_data)
ddpt_45_X = preprocess_ddpt_data(ddpt_45_data)

# Normalize features
scaler = StandardScaler()
ocv_X_scaled = scaler.fit_transform(ocv_X)
ddpt_15_X_scaled = scaler.fit_transform(ddpt_15_X)
ddpt_25_X_scaled = scaler.fit_transform(ddpt_25_X)
ddpt_45_X_scaled = scaler.fit_transform(ddpt_45_X)

# Define LSTM model
model = Sequential([
    LSTM(units=64, input_shape=(ocv_X_scaled.shape[1], ocv_X_scaled.shape[2])),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on OCV data
X_train, X_val, y_train, y_val = train_test_split(ocv_X_scaled, ocv_y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model
model_path = "ocv_model.h5"
model.save(model_path)
print("OCV Model saved successfully at:", model_path)

# Now you can repeat the process for DDPT data if needed.
# Define LSTM model for DDPT data
ddpt_model = Sequential([
    LSTM(units=64, input_shape=(ddpt_15_X_scaled.shape[1], ddpt_15_X_scaled.shape[2])),
    Dense(units=1)
])

# Compile the DDPT model
ddpt_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the DDPT model
# Assuming ddpt_15_X_scaled, ddpt_25_X_scaled, and ddpt_45_X_scaled are used for training
# and their corresponding targets are ddpt_15_y, ddpt_25_y, ddpt_45_y
# You should replace these with your actual target variables
ddpt_15_X_train, ddpt_15_X_val, ddpt_15_y_train, ddpt_15_y_val = train_test_split(ddpt_15_X_scaled, ddpt_15_y, test_size=0.2, random_state=42)
ddpt_25_X_train, ddpt_25_X_val, ddpt_25_y_train, ddpt_25_y_val = train_test_split(ddpt_25_X_scaled, ddpt_25_y, test_size=0.2, random_state=42)
ddpt_45_X_train, ddpt_45_X_val, ddpt_45_y_train, ddpt_45_y_val = train_test_split(ddpt_45_X_scaled, ddpt_45_y, test_size=0.2, random_state=42)

# Fit the model on DDPT 15deg data
ddpt_model.fit(ddpt_15_X_train, ddpt_15_y_train, validation_data=(ddpt_15_X_val, ddpt_15_y_val), epochs=10, batch_size=32)

# Fit the model on DDPT 25deg data
ddpt_model.fit(ddpt_25_X_train, ddpt_25_y_train, validation_data=(ddpt_25_X_val, ddpt_25_y_val), epochs=10, batch_size=32)

# Fit the model on DDPT 45deg data
ddpt_model.fit(ddpt_45_X_train, ddpt_45_y_train, validation_data=(ddpt_45_X_val, ddpt_45_y_val), epochs=10, batch_size=32)

# Save the trained DDPT model
ddpt_model_path = "ddpt_model.h5"
ddpt_model.save(ddpt_model_path)
print("DDPT Model saved successfully at:", ddpt_model_path)
