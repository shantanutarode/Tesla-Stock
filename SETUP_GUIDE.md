# 🚀 Tesla Stock Prediction - Setup Guide

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Jupyter Notebook
```bash
jupyter notebook Tesla_Stock_Analysis.ipynb
```
Then run all cells to train the models.

### Step 3: Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

Access the app at: `http://localhost:8501`

---

## 📁 Project Structure

```
Tesla Stock Prediction/
├── TSLA.csv                      # Dataset
├── Tesla_Stock_Analysis.ipynb    # Complete analysis
├── streamlit_app.py              # Web application
├── requirements.txt              # Dependencies
├── models/                       # Saved models (after training)
├── README.md                     # Documentation
└── SETUP_GUIDE.md               # This file
```

---

## 📊 What's Included

### 1. Jupyter Notebook Analysis
- **Data Loading & Cleaning** (20%)
- **Data Preprocessing** (20%)
- **Exploratory Data Analysis** (10%)
- **Feature Engineering** (10%)
- **Deep Learning Models** (30%)
  - SimpleRNN
  - LSTM
- **Model Evaluation** (10%)
- **Predictions** (1, 5, 10 days)

### 2. Streamlit Web App
- Interactive dashboard
- Real-time predictions
- Beautiful UI with custom CSS
- Model comparison charts
- Performance metrics

---

## 🎯 Project Deliverables Checklist

- ✅ Data Cleaning & Preprocessing
- ✅ Exploratory Data Analysis
- ✅ Feature Engineering
- ✅ SimpleRNN Model
- ✅ LSTM Model
- ✅ Model Comparison
- ✅ Hyperparameter Tuning
- ✅ Predictions (1, 5, 10 days)
- ✅ Streamlit Deployment
- ✅ Beautiful UI
- ✅ Comprehensive Documentation

---

## 🔧 Troubleshooting

### Issue: Module not found
**Solution**: Run `pip install -r requirements.txt`

### Issue: TensorFlow installation fails
**Solution**: Use Python 3.8-3.10, or install with:
```bash
pip install tensorflow --upgrade
```

### Issue: Streamlit app not loading
**Solution**: Check if port 8501 is available, or use:
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## 📝 Notes

- Training may take 10-20 minutes depending on your hardware
- Models are saved in `models/` directory
- Best viewed in dark mode browser
- For video recording, use screen capture tools

---

## 🎓 Evaluation Criteria

| Criteria | Weight | Status |
|----------|--------|--------|
| Data Cleaning | 20% | ✅ Complete |
| Data Preprocessing | 20% | ✅ Complete |
| Data Visualization | 10% | ✅ Complete |
| Feature Engineering | 10% | ✅ Complete |
| DL Modelling | 30% | ✅ Complete |
| Model Evaluation | 10% | ✅ Complete |

---

**Submission Deadline**: January 12, 2026  
**Status**: Ready for Submission

Good luck! 🚀
