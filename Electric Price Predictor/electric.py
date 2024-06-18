# Importing Required Libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog


# Loading the dataset
df = pd.read_csv("dataset.csv")

# Data Preprocessing
# Remove Null Values
df.dropna(inplace=True)

# Feature Selection
# Selecting relevant features
X = df[['Year', 'Month', 'People_No', 'Room_No',
        'Fuel_cost', 'Coal_Cost', 'Holiday_No']]
y = df['Consumption']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Model Training
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train.values, y_train.values)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train.values, y_train.values)

# Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(
    100, 50, 10), activation='relu', solver='adam', random_state=42)
nn_model.fit(X_train, y_train)

# Model Evaluation
# Linear Regression
y_pred_lr = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, y_pred_lr)

# Random Forest Regression
y_pred_rf = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

# Neural Network
y_pred_nn = nn_model.predict(X_test)
nn_mae = mean_absolute_error(y_test, y_pred_nn)
nn_mse = mean_squared_error(y_test, y_pred_nn)
nn_rmse = np.sqrt(nn_mse)
nn_r2 = r2_score(y_test, y_pred_nn)

# Printing Model Evaluation Results
print("Linear Regression Results:")
print("MAE: ", lr_mae)
print("MSE: ", lr_mse)
print("RMSE: ", lr_rmse)
print("R2 Score: ", lr_r2)
print()
print("Random Forest Regression Results:")
print("MAE: ", rf_mae)
print("MSE: ", rf_mse)
print("RMSE: ", rf_rmse)
print("R2 Score: ", rf_r2)
print()
print("Neural Network Results:")
print("MAE: ", nn_mae)
print("MSE: ", nn_mse)
print("RMSE: ", nn_rmse)
print("R2 Score: ", nn_r2)

# Model Deployment
# Deploying the best-performing model for predicting electricity prices
if lr_rmse < rf_rmse and lr_rmse < nn_rmse:
    model = lr_model
elif rf_rmse < lr_rmse and rf_rmse < nn_rmse:
    model = rf_model
else:
    model = nn_model
print(model)
# Saving the model for future use
with open('electricity_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Loading the test dataset for predicting electricity prices
test_df = pd.read_csv("dataset.csv")

# Data Preprocessing for test data
# Remove Null Values
test_df.dropna(inplace=True)

# Feature Selection for test data
X_test = test_df[['Year', 'Month', 'People_No',
                  'Room_No', 'Fuel_cost', 'Coal_Cost', 'Holiday_No']]
y_test = test_df['Consumption']


# Model Prediction
y_pred = model.predict(X_test)

# Printing Predictions
print("Predicted Electricity Prices: ", y_pred)

# Model Evaluation on Test Data
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# regr = linear_model.LinearRegression()
# model.fit(X_test, y_test)

# predict the Consumption:
pred = model.predict([[2016, 12, 8, 7, 48, 18, 3]])

# X_test = test_df['Consumption']
# y_test = test_df['Bill_Paid']

# coef_times = model.coef_

predi_bill = pred*2.7
print(pred)


# Printing Model Evaluation Results on Test Data
print()
print("Model Evaluation Results on Test Data:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R2 Score: ", r2)

# Initialize the Tkinter UI
root = tk.Tk()
root.title("Electricity Price Prediction")

# Function to select and load the dataset


def load_dataset():
    file_path = filedialog.askopenfilename()
    if file_path:
        global df
        df = pd.read_csv(file_path)
        status_label.config(text="Dataset loaded successfully.")

# Function to train and test the machine learning models


def train_and_test_model():
    # Data Preprocessing
    df.dropna(inplace=True)
    X = df[['Year', 'Month', 'People_No', 'Room_No',
            'Fuel_cost', 'Coal_Cost', 'Holiday_No']]
    y = df['Consumption']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Model Training
    models = {'Linear Regression': LinearRegression(),
              'Random Forest Regression': RandomForestRegressor(),
              'Neural Network Regression': MLPRegressor()}
    model_scores = {}
    for model_name, model in models.items():
        model.fit(X_train.values, y_train.values)
        y_pred = model.predict(X_test.values)
        mae = mean_absolute_error(y_test, y_pred)
        model_scores[model_name] = mae

    # Model Selection
    best_model = min(model_scores, key=model_scores.get)
    model = models[best_model]
    model.fit(X, y)
    status_label.config(text=f"{best_model} model trained and selected.")

    # Model Evaluation on Test Data
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Printing Model Evaluation Results on Test Data
    results_text = f"Model Evaluation Results on Test Data:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR2 Score: {r2:.2f}"
    results_label.config(text=results_text)
    results_label2.config(text=predi_bill)


# Add UI widgets
load_button = tk.Button(root, text="Load Dataset", command=load_dataset)
load_button.pack(pady=10)

train_button = tk.Button(
    root, text="Train and Test Model", command=train_and_test_model)
train_button.pack(pady=10)

status_label = tk.Label(root, text="")
status_label.pack(pady=10)

results_label = tk.Label(root, text="")
results_label.pack(pady=10)

results_label2 = tk.Label(root, text="")
results_label2.pack(pady=10)
# predi_bill
root.mainloop()
