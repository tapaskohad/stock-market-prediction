import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



# Load the historical stock price dataset
stock_data = pd.read_csv('stock_data.csv')

# Data Preprocessing
data = stock_data['Close'].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences for LSTM model
def create_sequences(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)

timesteps = 10  # Number of timesteps to use for each prediction
X, y = create_sequences(scaled_data, timesteps)

# Split the dataset into training and testing sets
split_index = int(len(X) * 0.8)  # 80% for training, 20% for testing
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Model Development
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)


# Model Training
epochs = 100
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Model Evaluation
loss = model.evaluate(X_test, y_test)

# Stock Market Forecasting
# Assuming you have future data in the variable 'future_data'
# You may need to preprocess the future data similarly to the training data
# Here, we create a placeholder array for demonstration purposes
future_data = np.array([[0.8], [0.9], [0.95], [1.0], [0.85]])  # Replace with your actual future data

# Ensure that the future data is scaled in the same way as the training data
scaled_future_data = scaler.transform(future_data)

# Create sequences for the future data
X_future, y_future = create_sequences(scaled_future_data, timesteps)

# Forecast future stock prices
future_predictions = model.predict(X_future)
future_predictions = scaler.inverse_transform(future_predictions)

# Display future predictions
print(future_predictions)
