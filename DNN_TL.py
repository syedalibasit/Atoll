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

data['bv'] = bv#np.random.uniform(1, 10, size=len(data))#bv
data['bh'] = bh#np.random.uniform(40, 80, size=len(data))#bh
data['Av'] = Av#np.random.uniform(1, 50, size=len(data))#Av
data['Ah'] = Ah#np.random.uniform(1, 50, size=len(data))#Ah
data['f'] = f#np.random.uniform(2.1, 4.2, size=len(data))#f
data['zeta'] = zeta#np.random.uniform(0.5, 2.5, size=len(data))#zeta
data['pioverbvh'] = (4*np.pi)/(data['bv']*data['bh'])#np.random.uniform(10, 25, size=len(data))
data['c1'] = c1#np.random.uniform(5, 50, size=len(data))#c1
data['c2'] = c2#np.random.uniform(5, 50, size=len(data))#c2
data['Tx_Pwr'] = Tx_Pwr#np.random.uniform(5, 50, size=len(data))#Tx_Pwr
data['hob'] = hob#np.random.uniform(5, 50, size=len(data))#hob
data['hue'] = hue#np.random.uniform(1.5, 22.5, size=len(data))#hue
data['W'] = W#np.random.uniform(5, 50, size=len(data))#W
X = data[['UE_Tilt','BS_Tilt','bv','Av','UE_Azimuth','BS_Azimuth','bh','Ah','zeta','pioverbvh','Distance','f','W','hob','hue','BS_Height','c1','c2','Tx_Pwr']]
y = data['RSRP']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed_value)

# %%% Scaling
# mm_scal_train = MinMaxScaler(feature_range=(-1, 1));
mm_scal_train = MaxAbsScaler()
scaled = mm_scal_train.fit_transform(X_train)
scaled_val = mm_scal_train.fit_transform(X_val)

#%%%
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting
plt.figure(figsize=(15, 10))
sns.histplot(scaled, bins=50, kde=True)
plt.title('Distribution of Complete Dataset')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
#%%%
# joblib.dump(scaler, "C:/Users/2687492Z/Desktop/proposed_model/NN/PMIRC/saved_models/DNN_scaler_model.joblib")
# joblib.dump(scaler, "C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/downsampling_experiment/DNN.joblib")
# %%% Build the NN model
# initializer = tf.keras.initializers.Constant(1.)
# np.random.seed(42)
# tf.random.set_seed(42)
# model = Sequential()
# model.add(Dense(19, input_dim=19, activation='relu'))  
# model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(19, activation='relu'))
# model.add(Dense(1, activation='linear'))
# regularizer=regularizers.l1(1.0)
# model = Sequential()
# model.add(Dense(19, input_dim=17, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(19, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='linear'))
# np.random.seed(42)
# tf.random.set_seed(42)
# model = Sequential()
# model.add(Dense(32, input_dim=19, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='linear'))
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
# initializer = tf.keras.initializers.Constant(1.)
# model = Sequential()
# model.add(Dense(32, input_dim=19, activation='relu', kernel_initializer=initializer))  
# model.add(Dense(32, activation='relu', kernel_initializer=initializer))
# model.add(Dense(32, activation='relu', kernel_initializer=initializer))
# model.add(Dense(32, activation='relu', kernel_initializer=initializer))
# model.add(Dense(1, activation='linear', kernel_initializer=initializer))
# model = Sequential()
# model.add(Dense(16, input_dim=19, activation='relu'))
# model.add(Dense(16, activation='relu')) 
# model.add(Dense(1, activation='linear'))

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

# history = model.fit(scaled, y, epochs=10000, batch_size=32, callbacks=[early_stopping])
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

# %%% Save the model
# model.save("C:/Users/2687492Z/Desktop/proposed_model/NN/PMIRC/saved_models/DNN_model_3000")
# model.save("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/downsampling_experiment/DNN")

initial_weights = {}
for layer in model.layers:
    if layer.trainable:
        layer_weights = layer.get_weights()
        initial_weights[layer.name] = [np.array(weight) for weight in layer_weights]
print(initial_weights)
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

df['bv'] = bv#np.random.uniform(1, 10, size=len(df))#bv
df['bh'] = bh#np.random.uniform(40, 80, size=len(df))#bh
df['Av'] = Av#np.random.uniform(1, 50, size=len(df))#Av
df['Ah'] = Ah#np.random.uniform(1, 50, size=len(df))#Ah
df['f'] = f#np.random.uniform(2.1, 4.2, size=len(df))#f
df['zeta'] = zeta#np.random.uniform(0.5, 2.5, size=len(df))#zeta
df['pioverbvh'] = (4*np.pi)/(df['bv']*df['bh'])#np.random.uniform(10, 25, size=len(df))#
df['c1'] = c1#np.random.uniform(5, 50, size=len(df))#c1
df['c2'] = c2#np.random.uniform(5, 50, size=len(df))#c2
df['Tx_Pwr'] = Tx_Pwr#np.random.uniform(5, 50, size=len(df))#Tx_Pwr
df['hob'] = hob#np.random.uniform(5, 50, size=len(df))
df['hue'] = hue#np.random.uniform(1.5, 22.5, size=len(df))
df['W'] = W#np.random.uniform(5, 50, size=len(df))
test_data_X = df[['UE_Tilt','BS_Tilt','bv','Av','UE_Azimuth','BS_Azimuth','bh','Ah','zeta','pioverbvh','Distance','f','W','hob','hue','BS_Height','c1','c2','Tx_Pwr']]
y_test = df['RSRP']

# %%% Load the scaler and Model
# scaler = joblib.load("C:/Users/2687492Z/Desktop/proposed_model/NN/PMIRC/saved_models/DNN_scaler_model.joblib")
# loaded_model = keras.models.load_model("C:/Users/2687492Z/Desktop/proposed_model/NN/PMIRC/saved_models/DNN_model_3000")
# scaler_test = joblib.load("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/downsampling_experiment/DNN.joblib")

# loaded_model = keras.models.load_model("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/downsampling_experiment/DNN")
X_test_scaled = mm_scal_train.transform(test_data_X)
# %%%
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting
plt.figure(figsize=(15, 10))
sns.histplot(X_test_scaled, bins=50, kde=True)
plt.title('Distribution of Complete Dataset')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
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
# %%% 
# Plot the probability density functions
import seaborn as sns

# Plot the probability density functions
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual RSRP Values', color='blue', fill=True)
sns.kdeplot(y_test_pred, label='Predicted RSRP Values', color='red', fill=True)
plt.xlabel('RSRP Values')
plt.ylabel('Density')
plt.title('Probability Density of Actual vs Predicted RSRP Values')
plt.legend()
plt.grid(True)
# Save the plot as a high-resolution PDF
# plt.savefig("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/results/probability_density_plot.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.show()
# %%%
y_test = np.ravel(y_test)  # or y_test.flatten()
y_test_pred = np.ravel(y_test_pred)
differences = np.abs(y_test - y_test_pred)
mean_difference = np.mean(differences)
print(mean_difference)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(differences)), differences, label='Absolute Differences', marker='o')
plt.xlabel('Sample Index')
plt.ylabel('Absolute Difference')
plt.title('Absolute Differences between Actual and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# %%% Transfer Learning
# target_model = clone_model(model)

target_model = model

target_model.summary()
#%%%
updated_weights = {}
for layer in target_model.layers:
    if layer.trainable:
        layer_weights = layer.get_weights()
        updated_weights[layer.name] = [np.array(weight) for weight in layer_weights]
print(updated_weights)
#%%%
new_df = pd.read_csv("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/tgt_train_data_BS_Height_14.csv")

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

new_df['bv'] = bv
new_df['bh'] = bh
new_df['Av'] = Av
new_df['Ah'] = Ah
new_df['f'] = f
new_df['zeta'] = zeta
new_df['pioverbvh'] = (4*np.pi)/(new_df['bv']*new_df['bh'])
new_df['c1'] = c1
new_df['c2'] = c2
new_df['Tx_Pwr'] = Tx_Pwr
new_df['hob'] = hob
new_df['hue'] = hue
new_df['W'] = W
# new_df['BS_Tilt'] = np.random.uniform(0, 10, size=len(new_df))
# new_df['Tx_Pwr'] = np.random.uniform(34, 43, size=len(new_df))
new_data_X = new_df[['UE_Tilt','BS_Tilt','bv','Av','UE_Azimuth','BS_Azimuth','bh','Ah','zeta','pioverbvh','Distance','f','W','hob','hue','BS_Height','c1','c2','Tx_Pwr']]
new_y_test = new_df['RSRP']
# %%%
# mm_scal_target = MinMaxScaler(feature_range=(-1, 1));
mm_scal_target = MaxAbsScaler()
scaled_target = mm_scal_target.fit_transform(new_data_X);

# %%%
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting
plt.figure(figsize=(15, 10))
sns.histplot(scaled_target, bins=50, kde=True)
plt.title('Distribution of Complete Dataset')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
# %%%
# List of layer names to set as trainable
trainable_layers = [6,9]

# Set specified layers to be trainable
for i, layer in enumerate(target_model.layers):
    if i in trainable_layers:
        layer.trainable = True
    else:
        layer.trainable = False
#%%%
target_model.compile(optimizer='adam', loss='mean_squared_error')
#%%%
target_model.summary()
#%%%
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
#%%% # Train the model on the new dataset
start_time_TL_train = time.time()
target_model.fit(scaled_target, new_y_test, epochs=10000, batch_size=32, callbacks=[early_stopping])
end_time_TL_train = time.time()
# Calculate the prediction time
TL_train_time = end_time_TL_train - start_time_TL_train
print("Train time Target model: {:.2f} seconds".format(TL_train_time))
#%%%
X_test_scaled = mm_scal_target.transform(test_data_X)

#%%%
start_time_pred = time.time()
TL_loss_pred = target_model.predict(X_test_scaled)
# End the timer for prediction
end_time_pred = time.time()

# Calculate the prediction time
prediction_time = end_time_pred - start_time_pred
print("Prediction time Target model: {:.2f} seconds".format(prediction_time))
#%%%
mse_test = mean_squared_error(y_test, TL_loss_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, TL_loss_pred)
mae_test = mean_absolute_error(y_test, TL_loss_pred)
mape_test = mean_absolute_percentage_error(y_test, TL_loss_pred)

print(f"\nTest Metrics:")
print(f"MSE: {mse_test}")
print(f"RMSE: {rmse_test}")
print(f"R^2: {r2_test}")
print(f"MAE: {mae_test}")
print(f"MAPE: {mape_test}")
#%%%
# Plot the probability density functions
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the probability density functions
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual RSRP Values', color='blue', fill=True)
sns.kdeplot(y_test_pred, label='Source Domain Predicted RSRP Values', color='red', fill=True)
sns.kdeplot(TL_loss_pred, label='Target Domain Predicted RSRP Values', color='gold', fill=False)
plt.xlabel('RSRP Values')
plt.ylabel('Density')
plt.title('Baseline DNN Model Probability Density of Actual vs Predicted RSRP Values in Source and Target')
plt.legend()
plt.grid(True)
plt.savefig("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/results/DNN_pdp.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.show()
#%%%
updated_weights = {}
for layer in target_model.layers:
    if layer.trainable:
        layer_weights = layer.get_weights()
        updated_weights[layer.name] = [np.array(weight) for weight in layer_weights]
print(updated_weights)
#%%%
# Define the number of layers to freeze in each iteration
# num_layers_to_freeze = [0, 1, 2, 3, 4, 5]  # Freeze 0 to 6 layers

# # Load your new dataset (X_train, y_train, X_val, y_val)
# # Replace X_train, y_train, X_val, y_val with your actual dataset

# for num_layers in num_layers_to_freeze:

#     # Freeze layers
#     for layer in target_model.layers[:num_layers]:
#         layer.trainable = False

#     # Compile the model for regression
#     target_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

#     # Train the model on the new dataset
#     target_model.fit(scaled_target, new_y_test, epochs=10000, batch_size=32, callbacks=[early_stopping])

#     start_time_pred = time.time()
#     TL_loss_pred = target_model.predict(test_data_X)
#     # End the timer for prediction
#     end_time_pred = time.time()

#     # Calculate the prediction time
#     prediction_time = end_time_pred - start_time_pred
#     print("Prediction time: {:.2f} seconds".format(prediction_time))

#     mse_test = mean_squared_error(y_test, TL_loss_pred)
#     rmse_test = np.sqrt(mse_test)
#     r2_test = r2_score(y_test, TL_loss_pred)
#     mae_test = mean_absolute_error(y_test, TL_loss_pred)
#     mape_test = mean_absolute_percentage_error(y_test, TL_loss_pred)

#     print(f"\nTest Metrics:")
#     print(f"MSE: {mse_test}")
#     print(f"RMSE: {rmse_test}")
#     print(f"R^2: {r2_test}")
#     print(f"MAE: {mae_test}")
#     print(f"MAPE: {mape_test}")
    
    
    
    
    
    
    
    
    
    
# %%% Dataset Distribution

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting
plt.figure(figsize=(15, 10))
sns.histplot(X_scaled, bins=50, kde=True)
plt.title('Distribution of Complete Dataset')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# %%
print("Model")

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

lgb_model = lgb.LGBMRegressor(**params)

lgb_model.fit(X_scaled, y)


print("\n Performance on Test Data")
predictions = lgb_model.predict(X_test_scaled)
print("MAPE: %.3f" % np.mean(np.abs((y_test - predictions) / y_test) * 100))
print("MAE: ",mean_absolute_error(y_test, predictions))
print("MSE: ",mean_squared_error(y_test, predictions))
print("RMSE: ",np.sqrt(mean_squared_error(y_test, predictions)))
# print("Explained Variance Score (R-squared Value): ",model.score(X_test_scaled, y_test))


#%%%
from sklearn.preprocessing import MinMaxScaler

# Sample train and test data
train_data = [[1], [2], [3], [4], [5]]
test_data = [[1], [2], [3]]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data
scaler.fit(train_data)

# Transform the training data
scaled_train_data = scaler.transform(train_data)

# Transform the test data
scaled_test_data = scaler.transform(test_data)

print("Scaled train data:", scaled_train_data)
print("Scaled test data:", scaled_test_data)

 
 