# Bharat-Intern-House-price-Machine-learning-Internship
This project focuses on predicting house prices using a dataset of housing features. The goal is to build and evaluate machine learning models to forecast house prices based on various attributes of the properties. The project leverages both traditional machine learning techniques and neural network models.
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df=pd.read_csv('/content/drive/MyDrive/HousePricePrediction (2).csv')
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Step 1: Load and Explore the Dataset
print("Loading dataset...")
df=pd.read_csv('/content/drive/MyDrive/HousePricePrediction (2).csv')

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(df.head())
# Getting the basic information and check for missing values
print("\nDataset Info:")
print(df.info())

# Summary statistics to understand data distribution
print("\nSummary Statistics:")
print(df.describe())
# Step 2: Data Preprocessing
print("\nHandling missing values...")
# Fill missing values with median for numerical columns
df.fillna(df.median(numeric_only=True), inplace=True)

# Encoding categorical variables using one-hot encoding
print("\nEncoding categorical variables...")
df = pd.get_dummies(df, drop_first=True)

# Checking the data after preprocessing
print("\nData after preprocessing:")
print(df.head())
# Split the data into features and target variable
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Split into training and testing sets
print("\nSplitting the data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Building with scikit-learn
print("\nTraining RandomForestRegressor model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
# Step 4: Evaluate the Random Forest Model
print("\nEvaluating RandomForestRegressor model...")
rf_y_pred = rf_model.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test, rf_y_pred)
print(f"RandomForestRegressor Mean Squared Error: {rf_mse:.2f}")

# Step 5: Building and Training Neural Network Model with TensorFlow
print("\nBuilding and training Neural Network model...")
nn_model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Training the neural network
history = nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 6: Evaluate the Neural Network Model
print("\nEvaluating Neural Network model...")
nn_y_pred = nn_model.predict(X_test_scaled)
nn_mse = mean_squared_error(y_test, nn_y_pred)
print(f"Neural Network Mean Squared Error: {nn_mse:.2f}")

# Step 7: Visualizations
print("\nCreating visualizations...")

# Feature Importance from Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Residual Plot for Random Forest
residuals = y_test - rf_y_pred
plt.figure(figsize=(10, 6))
plt.scatter(rf_y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot (Random Forest)')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()

# Loss Curve for Neural Network
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\nAnalysis complete!")

