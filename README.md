# 🚀 Tesla Stock Price Prediction

## 📊 Project Overview

A comprehensive Deep Learning project that predicts Tesla (TSLA) stock prices using **SimpleRNN** and **LSTM** models. The project includes detailed exploratory data analysis, data preprocessing, feature engineering, and model comparison with hyperparameter tuning.

## 🎯 Problem Statement

Create a predictive Deep Learning model to predict Tesla stock prices with the following objectives:
- Implement and compare **SimpleRNN** and **LSTM** models
- Predict stock closing prices for **1 day, 5 days, and 10 days**
- Handle missing values appropriately for time-series data
- Optimize models using **GridSearchCV** for hyperparameter tuning

## 📈 Skills & Technologies

- **Programming**: Python 3.x
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deep Learning**: TensorFlow, Keras (SimpleRNN, LSTM)
- **Model Optimization**: GridSearchCV
- **Deployment**: Streamlit

## 🏗️ Project Structure

```
Tesla Stock Prediction/
├── TSLA.csv                          # Dataset
├── requirements.txt                   # Python dependencies
├── Tesla_Stock_Analysis.ipynb         # Complete analysis notebook
├── streamlit_app.py                   # Streamlit web application
├── models/                            # Saved models
│   ├── simplernn_1day.h5
│   ├── simplernn_5day.h5
│   ├── simplernn_10day.h5
│   ├── lstm_1day.h5
│   ├── lstm_5day.h5
│   └── lstm_10day.h5
└── README.md                          # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd "Tesla Stock Prediction"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook**
   ```bash
   jupyter notebook Tesla_Stock_Analysis.ipynb
   ```

4. **Run Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Dataset

The dataset contains Tesla stock price data with the following features:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Adj Close**: Adjusted closing price
- **Volume**: Trading volume

## 🧠 Models Implemented

### 1. SimpleRNN
- Basic recurrent neural network
- Captures short-term dependencies
- Baseline model for comparison

### 2. LSTM (Long Short-Term Memory)
- Advanced RNN architecture
- Handles long-term dependencies
- Typically better performance on sequential data

### Prediction Horizons
- **1-Day Prediction**: Next day closing price
- **5-Day Prediction**: 5 days ahead closing price
- **10-Day Prediction**: 10 days ahead closing price

## 📈 Project Workflow

1. **Data Cleaning** (20%)
   - Handle missing values
   - Remove duplicates
   - Data type conversions

2. **Data Preprocessing** (20%)
   - Feature scaling (MinMaxScaler)
   - Time-series sequence creation
   - Train-test split (temporal)

3. **Data Visualization** (10%)
   - Stock price trends
   - Moving averages
   - Volume analysis
   - Correlation matrices

4. **Feature Engineering** (10%)
   - Moving averages (7-day, 30-day)
   - Volatility indicators
   - Technical indicators

5. **Deep Learning Modeling** (30%)
   - SimpleRNN implementation
   - LSTM implementation
   - Model training with early stopping

6. **Model Evaluation & Optimization** (10%)
   - MSE, RMSE, MAE metrics
   - GridSearchCV hyperparameter tuning
   - Model comparison

## 🎨 Streamlit App Features

- **Interactive Dashboard**: Beautiful UI with Tailwind CSS styling
- **Real-time Predictions**: Select model and prediction horizon
- **Visualizations**: Interactive charts with Plotly
- **Model Comparison**: Compare SimpleRNN vs LSTM performance
- **Historical Analysis**: Explore past trends

## 📊 Business Use Cases

1. **Automated Trading Strategies**
2. **Risk Management & Portfolio Optimization**
3. **Long-Term Investment Planning**
4. **Competitor Analysis**
5. **Financial Forecasting**

## 📝 Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

## 🎓 Learning Outcomes

- Time-series data preprocessing
- Deep learning for stock prediction
- RNN and LSTM architecture understanding
- Hyperparameter tuning techniques
- Model evaluation and comparison
- Web application deployment with Streamlit

## 👨‍💻 Author

**Your Name**
- Domain: Financial Services
- Project Type: Deep Learning - Time Series Prediction

## 📅 Project Timeline

- **Submission Deadline**: January 12, 2026
- **Status**: In Progress

## 📄 License

This project is created for educational purposes.

## 🙏 Acknowledgments

- Dataset: Yahoo Finance (Tesla Stock Data)
- Frameworks: TensorFlow, Keras, Streamlit
- Inspiration: Financial market prediction research

---

**Note**: Stock market predictions are for educational purposes only. Always consult financial advisors before making investment decisions.
