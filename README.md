India AQI Analysis Dashboard
# Real-Time Air Quality Index (AQI) Analysis & Forecasting Web Application

🚀 Project Description
A full-stack data science web app for analyzing, visualizing, and forecasting Air Quality Index (AQI) across Indian cities and states (2010–2025, including real-time updates).
Built with Python (FastAPI) backend and React frontend, it allows users to:

Select cities and year(s) to analyze

View summary statistics, variability metrics, box plots, violin plots, histograms, and more

See pollutant correlations (correlation matrices and heatmaps)

Forecast future AQI values using AI models (ARIMA time series)

Get health impact insights based on AQI categories

💡 Key Features
CSV Import + API Integration: Ready to analyze datasets from Dataful, Govt OGD portal, or live government APIs

Summary Analytics: Mean, median, std dev, quartiles, IQR, correlation

Rich Visualizations: Box plot, violin plot, histogram, density plot, scatter, correlation matrix, geographic/hexbin plots

Prediction Engine: ARIMA model for multi-month/year AQI forecasting with confidence intervals

Health Insights: Dashboard summarizes good/bad AQI with actionable advice

Modern UI: Clean and interactive React frontend

📦 Tech Stack
Backend: Python, FastAPI, pandas, statsmodels, seaborn, matplotlib

Frontend: React.js, axios, recharts, react-select

Data: CSV files (Dataful, CPCB, OGD), or direct API fetch (Data.gov.in API key)

Deployment: Docker (optional), Render.com / Railway / Google Cloud Run

⚙️ Local Installation & Setup
Clone the project

bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
Setup Python environment

bash
conda create -n aqi_project python=3.10
conda activate aqi_project
cd backend
pip install -r requirements.txt
Configure .env

Create a .env file in /backend/:

text
DATA_GOV_API_KEY=your_api_key_here
DEBUG=True
Prepare AQI dataset

Option 1: Download from Dataful and save as /backend/data/aqi_data.csv

Option 2: Use data_collector.py to fetch live data

Option 3: Use public government CSV (see README links)

Run FastAPI backend

bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Setup React frontend

bash
cd ../frontend
npm install
npm start
The app should be live at http://localhost:3000

🎯 Key Usage Steps
Open the app in your browser.

Select city/cities and year.

(Optional) Choose a prediction period (“Next N months” or “Next year”).

Click “Run Analysis”.

View visualizations, summary stats, AQI health assessment, and predictions.
