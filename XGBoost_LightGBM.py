import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import time
# from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import os
from openpyxl import load_workbook

# %% Load Training Dataset
data = pd.read_csv("C:/Users/2687492Z/Desktop/proposed_model/NN/Transfer_learning/TL-Model/dataset/subset_2percent.csv")

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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#%%%
# # XGBoost regression
# xgboost_model = xgb.XGBRegressor(
#     objective='reg:squarederror',  # Define the objective function
#     n_estimators=100,  # Number of boosting rounds
#     learning_rate=0.001,  # Learning rate
#     max_depth=6,  # Maximum depth of a tree
#     subsample=0.8,  # Subsample ratio of the training instance
#     colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
#     random_state=42  # Seed for reproducibility
# )
# #%%%
# start_time = time.time()
# xgboost_model.fit(X_train, y_train)
# # End the timer
# end_time = time.time()
# # Calculate the training time
# training_time = end_time - start_time
# print("Training time: {:.2f} seconds".format(training_time))
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

#%%%
# start_time_pred = time.time()
# y_pred_xgb = xgboost_model.predict(test_data_X)
# # End the timer for prediction
# end_time_pred = time.time()
# # Calculate the prediction time
# prediction_time = end_time_pred - start_time_pred
# print("Prediction time: {:.2f} seconds".format(prediction_time))
# #%%%
# mse_test = mean_squared_error(y_test, y_pred_xgb)
# rmse_test = np.sqrt(mse_test)
# r2_test = r2_score(y_test, y_pred_xgb)
# mae_test = mean_absolute_error(y_test, y_pred_xgb)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_xgb)

# print(f"\nTest Metrics:")
# print(f"MSE: {mse_test}")
# print(f"RMSE: {rmse_test}")
# print(f"R^2: {r2_test}")
# print(f"MAE: {mae_test}")
# print(f"MAPE: {mape_test}")
# rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
# print(f'XGBoost RMSE: {rmse_xgb}')
# #%%%
# # LightGBM regression
# lightgbm_model = lgb.LGBMRegressor(
#     objective='regression',  # Define the objective function
#     n_estimators=100,  # Number of boosting rounds
#     learning_rate=0.001,  # Learning rate
#     max_depth=6,  # Maximum depth of a tree
#     subsample=0.8,  # Subsample ratio of the training instance
#     colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
#     random_state=42  # Seed for reproducibility
# )
# #%%%
# start_time = time.time()
# lightgbm_model.fit(X_train, y_train)
# # End the timer
# end_time = time.time()
# # Calculate the training time
# training_time = end_time - start_time
# print("Training time: {:.2f} seconds".format(training_time))
# #%%%
# start_time_pred = time.time()
# y_pred_lgb = lightgbm_model.predict(test_data_X)
# # End the timer for prediction
# end_time_pred = time.time()
# # Calculate the prediction time
# prediction_time = end_time_pred - start_time_pred
# print("Prediction time: {:.2f} seconds".format(prediction_time))
# #%%%
# mse_test = mean_squared_error(y_test, y_pred_lgb)
# rmse_test = np.sqrt(mse_test)
# r2_test = r2_score(y_test, y_pred_lgb)
# mae_test = mean_absolute_error(y_test, y_pred_lgb)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_lgb)

# print(f"\nTest Metrics:")
# print(f"MSE: {mse_test}")
# print(f"RMSE: {rmse_test}")
# print(f"R^2: {r2_test}")
# print(f"MAE: {mae_test}")
# print(f"MAPE: {mape_test}")
# rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
# print(f'LightGBM RMSE: {rmse_lgb}')
#%%% Initialize and train the models
# Standardize the dataset for traditional models
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data_X)

models = {
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Linear Regression": LinearRegression(),
    "LightGBM": lgb.LGBMRegressor(
        objective='regression',
        n_estimators=100,
        learning_rate=0.001,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    "XGBoost": xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.001,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}
#%%% Dictionary to store the results
results = {}

for name, model in models.items():
    # Measure training time
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    end_train = time.time()
    training_time = end_train - start_train
    
    # Measure prediction time
    start_predict = time.time()
    y_test_pred = model.predict(X_test_scaled)
    end_predict = time.time()
    prediction_time = end_predict - start_predict
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    results[name] = {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAE":mae,
        "MAPE":mape,
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time
    }
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}, Training Time: {training_time:.4f}s, Prediction Time: {prediction_time:.4f}s")
#%%%
# X_train_dnn, X_val_dnn, y_train_dnn, y_val_dnn = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Apply MaxAbsScaler for the DNN
mm_scaler = MaxAbsScaler()
X_train_dnn_scaled = mm_scaler.fit_transform(X_train)
X_val_dnn_scaled = mm_scaler.transform(X_val)
X_test_dnn_scaled = mm_scaler.transform(test_data_X)
# Add the DNN model
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
# model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(19, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizer))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Measure training time for DNN
start_train = time.time()
history = model.fit(X_train_dnn_scaled, y_train, epochs=1000, batch_size=32, validation_data=(X_val_dnn_scaled, y_val), callbacks=[early_stopping])
end_train = time.time()
training_time = end_train - start_train

# Measure prediction time for DNN
start_predict = time.time()
y_pred_dnn = model.predict(X_test_dnn_scaled)
end_predict = time.time()
prediction_time = end_predict - start_predict

    # Calculate metrics
mse = mean_squared_error(y_test, y_pred_dnn)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_dnn)
mae = mean_absolute_error(y_test, y_pred_dnn)
mape = mean_absolute_percentage_error(y_test, y_pred_dnn)
    
results["DNN"] = {
"MSE": mse,
"RMSE": rmse,
"R2": r2,
"MAE":mae,
"MAPE":mape,
"Training Time (s)": training_time,
"Prediction Time (s)": prediction_time
}
print(f"DNN - MSE: {mse:.2f}, R2: {r2:.2f}, Training Time: {training_time:.4f}s, Prediction Time: {prediction_time:.4f}s")
#%%%
# Display the results
results_df = pd.DataFrame(results).T
print("\nSummary of Results:")
print(results_df)

#%%% Function to save results_df in an Excel file

# Define the list of model names
model_names = [
    "Gradient Boosting",
    "AdaBoost",
    "Extra Trees",
    "Random Forest",
    "Decision Tree",
    "Linear Regression",
    "LightGBM",
    "XGBoost",
    # "DNN"
]

# Check if 'Model' column already exists
if 'Model' not in results_df.columns:
    # Add the model names as the first column
    results_df.insert(0, 'Model', model_names)
else:
    print("The 'Model' column already exists in the DataFrame.")
    
# Define the path
file_path = r"C:\Users\2687492Z\Desktop\proposed_model\NN\Transfer_learning\TL-Model\dataset\results\Results.xlsx"

# Prompt the user for the new sheet name
new_sheet_name = input("Please enter the name for the new sheet: ")

# Use ExcelWriter to append the new sheet to the existing Excel file
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name=new_sheet_name, index=False)

print(f"Results have been saved to {file_path} in the sheet named '{new_sheet_name}'")