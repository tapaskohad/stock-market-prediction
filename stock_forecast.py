import numpy as np
from tensorflow import keras

X_train = np.random.rand(100, 10, 1)
y_train = np.random.rand(100, 1)

model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=100, batch_size=1)

future_sequence = []
last_sequence = X_train[-1]
num_future_time_steps = 4

for _ in range(num_future_time_steps):
    next_time_step = model.predict(np.array([last_sequence]))[0][0]
    
    future_sequence.append(next_time_step)
    
    last_sequence = np.concatenate([last_sequence[1:], [[next_time_step]]])

X_future = np.array(future_sequence)

print("Predicted future values:")
print(X_future)
