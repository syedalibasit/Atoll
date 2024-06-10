# %% Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import joblib
import time
from tensorflow.keras.initializers import Ones
import lightgbm as lgb
import tensorflow as tf
from keras.initializers import Constant
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, BatchNormalization
import random
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load Training Dataset
data = pd.read_csv("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/subset_2percent.csv")

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

c1 = 28
c2 = 31.69
Tx_Pwr = 43
f = 2100 *10**6
bv = 4.6
bh = 65
Ah = 25
Av = 20
zeta = 0.5
hob = 20
hue = 1.5
W = 20
# np.random.seed(42)

data['bv'] = bv
data['bh'] = bh
data['Av'] = Av
data['Ah'] = Ah
data['f'] = f
data['zeta'] = zeta
data['pioverbvh'] = (4*np.pi)/(data['bv']*data['bh'])
data['c1'] = c1
data['c2'] = c2
data['Tx_Pwr'] = Tx_Pwr
data['hob'] = hob
data['hue'] = hue
data['W'] = W
X = data[['UE_Tilt','BS_Tilt','bv','Av','UE_Azimuth','BS_Azimuth','bh','Ah','zeta','pioverbvh','Distance','f','W','hob','hue','BS_Height','c1','c2','Tx_Pwr']]
y = data['RSRP']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed_value)

# %%% Scaling
mm_scal_train = MaxAbsScaler()
scaled = mm_scal_train.fit_transform(X_train)
scaled_val = mm_scal_train.fit_transform(X_val)

# %%% Build the NN model
regularizer=regularizers.l1(1.0)
model = Sequential()
model.add(Dense(32, input_dim=19, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))


# %%% Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# Create the Adam optimizer with the specified learning rate
optimizer = Adam(learning_rate=0.001)

# Compile the model with the Adam optimizer and mean squared error loss
model.compile(optimizer=optimizer, loss='mean_squared_error')

#%%% summary
model.summary()

# %%% Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# %% Train the model
# Start the timer
start_time = time.time()
history = model.fit(scaled, y_train, epochs=1000, batch_size=32, validation_data=(scaled_val, y_val), callbacks=[early_stopping])
# End the timer
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))

# %%% Plot the loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(10, 6))
plt.plot(loss, label='Training Loss')

# If you have validation loss: 
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


# %% Load Testing Dataset
df = pd.read_csv("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/subset_5000.csv")

c1 = 28
c2 = 31.69
Tx_Pwr = 43
f = 2100 *10**6
bv = 4.6
bh = 65
Ah = 25
Av = 20
zeta = 0.5
hob = 20
hue = 1.5
W = 20
np.random.seed(42)

df['bv'] = bv
df['bh'] = bh
df['Av'] = Av
df['Ah'] = Ah
df['f'] = f
df['zeta'] = zeta
df['pioverbvh'] = (4*np.pi)/(df['bv']*df['bh'])
df['c1'] = c1
df['c2'] = c2
df['Tx_Pwr'] = Tx_Pwr
df['hob'] = hob
df['hue'] = hue
df['W'] = W
test_data_X = df[['UE_Tilt','BS_Tilt','bv','Av','UE_Azimuth','BS_Azimuth','bh','Ah','zeta','pioverbvh','Distance','f','W','hob','hue','BS_Height','c1','c2','Tx_Pwr']]
y_test = df['RSRP']

# %%% Load the scaler and Model
X_test_scaled = mm_scal_train.transform(test_data_X)

# %%% Predict on the test set
# Start the timer for prediction
start_time_pred = time.time()
y_test_pred = model.predict(X_test_scaled)
# End the timer for prediction
end_time_pred = time.time()

# Calculate the prediction time
prediction_time = end_time_pred - start_time_pred
print("Prediction time: {:.2f} seconds".format(prediction_time))
# %%% Calculate metrics for test set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

print(f"\nTest Metrics:")
print(f"MSE: {mse_test}")
print(f"RMSE: {rmse_test}")
print(f"R^2: {r2_test}")
print(f"MAE: {mae_test}")
print(f"MAPE: {mape_test}")

 
