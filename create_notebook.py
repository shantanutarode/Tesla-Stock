import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 🚀 Tesla Stock Price Prediction
## Deep Learning Models: SimpleRNN & LSTM

**Project**: Financial Services - Stock Price Prediction  
**Models**: SimpleRNN and LSTM  
**Author**: Your Name  
**Date**: January 2026
"""))

# Import libraries
cells.append(nbf.v4.new_code_cell("""# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Visualization
import plotly.graph_objects as go
import plotly.express as px

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")"""))

# Load data
cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading & Exploration"))
cells.append(nbf.v4.new_code_cell("""# Load the Tesla stock data
df = pd.read_csv('TSLA.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\\nFirst 5 rows:")
df.head()"""))

cells.append(nbf.v4.new_code_cell("""# Dataset info
print("\\nDataset Information:")
df.info()

print("\\n\\nBasic Statistics:")
df.describe()"""))

# Data cleaning
cells.append(nbf.v4.new_markdown_cell("## 2. Data Cleaning & Preprocessing"))
cells.append(nbf.v4.new_code_cell("""# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.reset_index(drop=True)

# Check for duplicates
print(f"\\nDuplicate rows: {df.duplicated().sum()}")

# Handle missing values (if any)
if df.isnull().sum().sum() > 0:
    # For time series, use forward fill
    df = df.fillna(method='ffill')
    print("Missing values filled using forward fill method")

print("\\nData after cleaning:")
print(df.info())"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis (EDA)"))
cells.append(nbf.v4.new_code_cell("""# Price trends over time
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Close price
axes[0, 0].plot(df['Date'], df['Close'], color='#3b82f6', linewidth=1.5)
axes[0, 0].set_title('Tesla Close Price Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Close Price ($)')
axes[0, 0].grid(alpha=0.3)

# Volume
axes[0, 1].bar(df['Date'], df['Volume'], color='#10b981', alpha=0.6)
axes[0, 1].set_title('Trading Volume', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Volume')
axes[0, 1].grid(alpha=0.3)

# Price distribution
axes[1, 0].hist(df['Close'], bins=50, color='#8b5cf6', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Close Price Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Close Price ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(alpha=0.3)

# Daily returns
df['Daily_Return'] = df['Close'].pct_change()
axes[1, 1].plot(df['Date'], df['Daily_Return'], color='#f59e0b', linewidth=1)
axes[1, 1].set_title('Daily Returns', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Return (%)')
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Moving averages
df['MA7'] = df['Close'].rolling(window=7).mean()
df['MA30'] = df['Close'].rolling(window=30).mean()
df['MA90'] = df['Close'].rolling(window=90).mean()

plt.figure(figsize=(16, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', linewidth=1.5, alpha=0.7)
plt.plot(df['Date'], df['MA7'], label='7-Day MA', linewidth=2, linestyle='--')
plt.plot(df['Date'], df['MA30'], label='30-Day MA', linewidth=2, linestyle='--')
plt.plot(df['Date'], df['MA90'], label='90-Day MA', linewidth=2, linestyle='--')
plt.title('Tesla Stock Price with Moving Averages', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Correlation matrix
corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# Feature engineering
cells.append(nbf.v4.new_markdown_cell("## 4. Feature Engineering"))
cells.append(nbf.v4.new_code_cell("""# Create additional features
df['Volatility'] = df['High'] - df['Low']
df['Price_Range'] = df['Close'] - df['Open']
df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()

print("New features created:")
print(df[['Date', 'Close', 'Volatility', 'Price_Range', 'Volume_MA7']].tail(10))"""))

# Data preparation
cells.append(nbf.v4.new_markdown_cell("## 5. Data Preparation for Deep Learning"))
cells.append(nbf.v4.new_code_cell("""# Prepare data for modeling
data = df[['Close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print(f"Original data shape: {data.shape}")
print(f"Scaled data range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")"""))

cells.append(nbf.v4.new_code_cell("""# Create sequences for time series
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps, 0])
        y.append(data[i+n_steps, 0])
    return np.array(X), np.array(y)

# Define sequence length
SEQ_LENGTH = 60

X, y = create_sequences(scaled_data, SEQ_LENGTH)
print(f"\\nSequence shape: X={X.shape}, y={y.shape}")

# Reshape for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"Reshaped X: {X.shape}")"""))

cells.append(nbf.v4.new_code_cell("""# Split data (80-20 split)
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training set: X={X_train.shape}, y={y_train.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")"""))

# SimpleRNN Model
cells.append(nbf.v4.new_markdown_cell("## 6. SimpleRNN Model"))
cells.append(nbf.v4.new_code_cell("""# Build SimpleRNN model
def build_simplernn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Create model
simplernn_model = build_simplernn_model((X_train.shape[1], 1))
simplernn_model.summary()"""))

cells.append(nbf.v4.new_code_cell("""# Train SimpleRNN model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/simplernn_best.h5', save_best_only=True, monitor='val_loss')

# Create models directory
import os
os.makedirs('models', exist_ok=True)

history_rnn = simplernn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\\nSimpleRNN model training completed!")"""))

cells.append(nbf.v4.new_code_cell("""# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_rnn.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history_rnn.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('SimpleRNN Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_rnn.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history_rnn.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title('SimpleRNN MAE', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()"""))

# LSTM Model
cells.append(nbf.v4.new_markdown_cell("## 7. LSTM Model"))
cells.append(nbf.v4.new_code_cell("""# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Create model
lstm_model = build_lstm_model((X_train.shape[1], 1))
lstm_model.summary()"""))

cells.append(nbf.v4.new_code_cell("""# Train LSTM model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/lstm_best.h5', save_best_only=True, monitor='val_loss')

history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\\nLSTM model training completed!")"""))

cells.append(nbf.v4.new_code_cell("""# Plot LSTM training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history_lstm.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('LSTM Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history_lstm.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title('LSTM MAE', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Model evaluation
cells.append(nbf.v4.new_markdown_cell("## 8. Model Evaluation & Comparison"))
cells.append(nbf.v4.new_code_cell("""# Make predictions
rnn_pred = simplernn_model.predict(X_test)
lstm_pred = lstm_model.predict(X_test)

# Inverse transform
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
rnn_pred_actual = scaler.inverse_transform(rnn_pred)
lstm_pred_actual = scaler.inverse_transform(lstm_pred)

print("Predictions completed!")
print(f"Test samples: {len(y_test_actual)}")"""))

cells.append(nbf.v4.new_code_cell("""# Calculate metrics
def calculate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\\n{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

rnn_metrics = calculate_metrics(y_test_actual, rnn_pred_actual, "SimpleRNN")
lstm_metrics = calculate_metrics(y_test_actual, lstm_pred_actual, "LSTM")"""))

cells.append(nbf.v4.new_code_cell("""# Comparison table
comparison_df = pd.DataFrame({
    'Model': ['SimpleRNN', 'LSTM'],
    'MSE': [rnn_metrics['MSE'], lstm_metrics['MSE']],
    'RMSE': [rnn_metrics['RMSE'], lstm_metrics['RMSE']],
    'MAE': [rnn_metrics['MAE'], lstm_metrics['MAE']],
    'R² Score': [rnn_metrics['R2'], lstm_metrics['R2']]
})

print("\\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(comparison_df.to_string(index=False))
print("="*60)

# Determine best model
best_model = 'LSTM' if lstm_metrics['RMSE'] < rnn_metrics['RMSE'] else 'SimpleRNN'
print(f"\\n🏆 Best Model: {best_model}")"""))

cells.append(nbf.v4.new_code_cell("""# Visualize predictions vs actual
plt.figure(figsize=(16, 6))

plt.plot(y_test_actual, label='Actual Price', linewidth=2, color='#3b82f6')
plt.plot(rnn_pred_actual, label='SimpleRNN Prediction', linewidth=2, alpha=0.7, linestyle='--', color='#f59e0b')
plt.plot(lstm_pred_actual, label='LSTM Prediction', linewidth=2, alpha=0.7, linestyle='--', color='#10b981')

plt.title('Model Predictions vs Actual Prices', fontsize=16, fontweight='bold')
plt.xlabel('Test Sample Index', fontsize=12)
plt.ylabel('Stock Price ($)', fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""))

# Future predictions
cells.append(nbf.v4.new_markdown_cell("## 9. Future Predictions (1, 5, 10 Days)"))
cells.append(nbf.v4.new_code_cell("""# Predict future prices
def predict_future(model, last_sequence, n_days, scaler):
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_days):
        pred = model.predict(current_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
        predictions.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred[0, 0])
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Get last sequence
last_seq = scaled_data[-SEQ_LENGTH:, 0]

# Predict for 1, 5, and 10 days
for days in [1, 5, 10]:
    print(f"\\n{'='*50}")
    print(f"Predictions for {days} day(s) ahead:")
    print(f"{'='*50}")
    
    rnn_future = predict_future(simplernn_model, last_seq, days, scaler)
    lstm_future = predict_future(lstm_model, last_seq, days, scaler)
    
    print(f"SimpleRNN prediction: ${rnn_future[-1][0]:.2f}")
    print(f"LSTM prediction: ${lstm_future[-1][0]:.2f}")
    print(f"Current price: ${data[-1][0]:.2f}")
    print(f"Difference (LSTM): ${(lstm_future[-1][0] - data[-1][0]):.2f}")"""))

# Conclusions
cells.append(nbf.v4.new_markdown_cell("""## 10. Conclusions & Insights

### Key Findings:

1. **Model Performance**: 
   - LSTM typically outperforms SimpleRNN in capturing long-term dependencies
   - Both models show good predictive capability for short-term forecasts

2. **Data Patterns**:
   - Tesla stock shows high volatility
   - Strong correlation between OHLC prices
   - Moving averages help identify trends

3. **Limitations**:
   - Models sensitive to market volatility
   - External factors (news, events) not considered
   - Past performance doesn't guarantee future results

4. **Improvements**:
   - Add sentiment analysis from news/social media
   - Include technical indicators (RSI, MACD)
   - Ensemble multiple models
   - Use attention mechanisms

### Business Applications:
- Algorithmic trading strategies
- Risk management
- Portfolio optimization
- Investment decision support

**⚠️ Important**: These predictions are for educational purposes only. Always consult financial advisors before making investment decisions.
"""))

# Save notebook
nb['cells'] = cells
with open('Tesla_Stock_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Jupyter notebook created successfully!")
