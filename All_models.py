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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Standardize the dataset for traditional models
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data_X)

#%%% Initialize and train the models

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
