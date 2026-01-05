@echo off
echo ================================================
echo   Tesla Stock Price Prediction - Quick Start
echo ================================================
echo.

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo.

echo [2/3] Installing dependencies...
pip install --quiet streamlit plotly pandas numpy matplotlib seaborn scikit-learn
echo Dependencies installed successfully!
echo.

echo [3/3] Launching Streamlit App...
echo.
echo ================================================
echo   Opening Tesla Stock Prediction Dashboard
echo   URL: http://localhost:8501
echo ================================================
echo.

streamlit run streamlit_app.py

pause
