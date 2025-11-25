🌏 India AQI Analysis Dashboard
Real-Time Air Quality Monitoring, Analytics & Forecasting (2010–2025)

A Full-Stack Data Science Web Application

<p align="center"> <img src="https://img.shields.io/badge/FastAPI-API%20Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/> <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=white"/> <img src="https://img.shields.io/badge/Python-Data%20Science-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/ARIMA-Forecasting-5C2D91?style=for-the-badge"/> </p>
🌟 Overview

The India AQI Analysis Dashboard is a powerful analytics platform built to study, visualize, and forecast Air Quality Index (AQI) levels across Indian cities from 2010–2025, including optional real-time API updates.

Built with a Python FastAPI backend and a React.js frontend, the dashboard combines statistical analysis, data visualization, and machine learning forecasting into one clean, interactive interface.


README

🚀 Key Features
📥 Data Import Options

Upload CSV from Dataful, CPCB, OGD

Real-time data from Data.gov.in API
(requires API key)

📊 Comprehensive Statistics

Mean, median, quartiles

Standard deviation, IQR

Trend & variability analysis

📈 Beautiful Visualizations

Box plots / Violin plots

Correlation matrix & Heatmap

Histograms / Density plots

Scatter plots

Geographic plots / Hexbin maps

🤖 AI Forecasting Engine

ARIMA time-series forecasting

Predict up to N months / full year

Confidence interval support

🩺 Health Insights Panel

AQI category classification

Health risk level interpretation

Safety recommendations

🧰 Tech Stack

Backend: FastAPI, Pandas, Statsmodels, Seaborn, Matplotlib

Frontend: React.js, Axios, Recharts, React-Select

Deployment: Render / Railway / GCR

Optional: Docker support


README

🛠️ Installation & Setup
📦 1. Clone the Repository

git clone https://github.com/Anand-DN/Real-Time-Air-Quality-Index-AQI-Analysis-Forecasting-Web-Application.git
cd YOUR_REPO_NAME

🐍 Backend Setup (FastAPI)
🔧 2. Create Python Environment
conda create -n aqi_project python=3.10
conda activate aqi_project

📁 3. Install Dependencies
cd backend
pip install -r requirements.txt

🔑 4. Add Environment Variables

Create /backend/.env:

DATA_GOV_API_KEY=your_api_key_here
DEBUG=True

📊 5. Add Dataset

Choose one:

Download CSV → save to /backend/data/aqi_data.csv

OR run data_collector.py


README

▶️ 6. Run Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

💻 Frontend Setup (React)
📦 7. Install Dependencies
cd ../frontend
npm install

▶️ 8. Run Frontend
npm start


App runs at: http://localhost:3000

Backend runs at: http://localhost:8000



README

🎯 How to Use

Select city/cities

Choose analysis year range

(Optional) Select prediction horizon

Click Run Analysis

View:

Summary statistics

Visualizations

Correlation matrix

AQI health assessment

Forecast graphs


README

📸 Screenshots

(Add your dashboard screenshots here)

![Dashboard Screenshot](./images/dashboard.png)

🧠 Future Enhancements

LSTM/Prophet forecasting

Mobile-friendly UI overhaul

Automated daily AQI ingestion

Interactive AQI heatmap

Pollutant breakdown analytics

🤝 Contributing

Pull requests and suggestions are welcome!

📄 License

MIT License © 2025
