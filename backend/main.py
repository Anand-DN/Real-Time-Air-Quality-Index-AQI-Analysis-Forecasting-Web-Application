
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="AQI Analysis API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_CACHE = {}

@app.on_event("startup")
async def load_data_on_startup():
    global DATA_CACHE
    try:
        if os.path.exists('data/aqi_data_waqi.csv'):
            df = pd.read_csv('data/aqi_data_waqi.csv')
            DATA_CACHE['df'] = df
            print(f"✓ Preloaded {len(df)} records from aqi_data_waqi.csv")
        elif os.path.exists('data/aqi_data_openaq.csv'):
            df = pd.read_csv('data/aqi_data_openaq.csv')
            DATA_CACHE['df'] = df
            print(f"✓ Preloaded {len(df)} records from aqi_data_openaq.csv")
        else:
            print("⚠️ No AQI data CSV found at startup.")
    except Exception as e:
        print(f"Startup data load error: {e}")

class AnalysisRequest(BaseModel):
    cities: List[str]
    year: int
    predict_months: Optional[int] = None
    predict_year: Optional[int] = None

@app.get("/")
async def root():
    return {
        "message": "AQI Analysis API",
        "status": "running",
        "debug_mode": os.getenv('DEBUG', 'False')
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/cities")
async def get_cities():
    default_cities = [
        "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
        "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
    ]
    try:
        df = DATA_CACHE.get('df')
        if df is not None and 'city' in df.columns:
            cities = sorted(df['city'].unique().tolist())
            return {"cities": cities}
        else:
            return {"cities": default_cities}
    except Exception as e:
        print(f"Error loading cities: {e}")
        return {"cities": default_cities}

@app.post("/api/analyze")
async def run_analysis(request: AnalysisRequest):
    try:
        data = load_aqi_data(request.cities, request.year)
        print(f"✓ Loaded {len(data)} records for analysis ({request.cities}, {request.year})")
        predictions = None
        if request.predict_months or request.predict_year:
            print(f"📊 Prediction requested: months={request.predict_months}, year={request.predict_year}")
            predictions = generate_predictions(
                data, 
                predict_months=request.predict_months,
                predict_year=request.predict_year
            )
            print(f"Predictions result: {type(predictions)} | Values: {predictions.get('forecasted_values') if predictions else 'None'}")
        else:
            print("No prediction requested.")

        if data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for cities {request.cities} in year {request.year}. Please run data_collector.py first or create sample data."
            )

        results = {
            "summary_stats": calculate_summary_statistics(data),
            "variability_metrics": calculate_variability_metrics(data),
            "correlation_matrix": calculate_correlation_matrix(data),
            "visualizations": generate_visualizations(data),
            "predictions": predictions,
            "ai_summary": generate_ai_summary(data, predictions)
        }
        results = convert_np_types(results)
        return results
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def load_aqi_data(cities, year):
    # Use cached data if available (recommended for Render)
    if 'df' in DATA_CACHE:
        df = DATA_CACHE['df'].copy()
        if 'city' not in df.columns:
            print("⚠️ No 'city' column in cached data.")
            return pd.DataFrame()
        df = df[df['city'].isin(cities)]
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['year'] = df['timestamp'].dt.year
            df = df[df['year'] == year]
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df = df[df['year'] == year]
        else:
            print("⚠️ No 'timestamp' or 'date' in data.")
        print(f"[DATA] After filters: {len(df)} rows")
        return df

    # Fallback if cache missing: try direct load
    if os.path.exists('data/aqi_data_waqi.csv'):
        df = pd.read_csv('data/aqi_data_waqi.csv')
        df = df[df['city'].isin(cities)]
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['year'] = df['timestamp'].dt.year
        df = df[df['year'] == year]
        print(f"[DATA-Fallback] After filters: {len(df)} rows")
        return df
    if os.path.exists('data/aqi_data_openaq.csv'):
        df = pd.read_csv('data/aqi_data_openaq.csv')
        df = df[df['city'].isin(cities)]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df = df[df['year'] == year]
        print(f"[DATA-Fallback] After filters: {len(df)} rows")
        return df

    # Generate sample data if nothing found (for testing/demo)
    print("⚠️ No data file found, generating sample data")
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    sample_data = []
    for city in cities:
        for date in dates:
            sample_data.append({
                'city': city,
                'timestamp': date,
                'aqi': np.random.randint(50, 200),
                'pm25': np.random.randint(20, 100),
                'pm10': np.random.randint(30, 150),
            })
    return pd.DataFrame(sample_data)


def convert_np_types(o):
    """Recursively convert numpy types to Python types."""
    import numpy as np
    if isinstance(o, dict):
        return {k: convert_np_types(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [convert_np_types(i) for i in o]
    elif isinstance(o, np.generic):
        return o.item()
    return o


def calculate_summary_statistics(df):
    """Calculate mean, median, and variations"""
    
    # Get numeric column (AQI or value)
    if 'aqi' in df.columns:
        values = df['aqi'].replace([np.inf, -np.inf], np.nan).dropna()
    elif 'value' in df.columns:
        values = df['value'].replace([np.inf, -np.inf], np.nan).dropna()
    else:
        return {}
    
    if len(values) == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "count": 0
        }
    
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "count": int(len(values)),
        "quartiles": {
            "Q1": float(values.quantile(0.25)),
            "Q2": float(values.quantile(0.50)),
            "Q3": float(values.quantile(0.75))
        }
    }


def calculate_variability_metrics(df):
    """Calculate Std Dev, MAD, IQR"""
    
    if 'aqi' in df.columns:
        values = df['aqi'].replace([np.inf, -np.inf], np.nan).dropna()
    elif 'value' in df.columns:
        values = df['value'].replace([np.inf, -np.inf], np.nan).dropna()
    else:
        return {}
    
    if len(values) < 2:
        return {
            "std_dev": None,
            "mad": None,
            "iqr": None,
            "variance": None,
            "cv": None
        }
    
    mean_val = values.mean()
    
    return {
        "std_dev": float(values.std()),
        "mad": float(np.mean(np.abs(values - mean_val))),
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "variance": float(values.var()),
        "cv": float((values.std() / mean_val) * 100) if mean_val != 0 else None
    }

def calculate_correlation_matrix(df):
    """Calculate correlation matrix for pollutants"""
    
    # Select numeric pollutant columns
    pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    available_cols = [col for col in pollutant_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return {}
    
    # Create correlation matrix
    corr_df = df[available_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(corr_df) < 2:
        return {}
    
    corr_matrix = corr_df.corr()
    
    # Convert to dictionary format
    return {
        "columns": corr_matrix.columns.tolist(),
        "data": corr_matrix.values.tolist()
    }

def generate_predictions(df, predict_months=None, predict_year=None):
    """Generate AQI predictions using ARIMA time series forecasting"""
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        
        # Determine value and date columns
        value_col = 'aqi' if 'aqi' in df.columns else 'value'
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        
        if date_col not in df.columns:
            return None
        
        # Prepare time series data
        ts_df = df[[date_col, value_col]].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        ts_df = ts_df.sort_values(date_col)
        
        # Aggregate by date (average if multiple records per day)
        ts_df = ts_df.groupby(date_col)[value_col].mean().reset_index()
        ts_df.set_index(date_col, inplace=True)
        
        if len(ts_df) < 10:
            return {
                "error": "Insufficient data for forecasting (minimum 10 data points required)",
                "forecasted_values": [],
                "dates": []
            }
        Fdef
        # Determine forecast periods
        if predict_year:
            periods = 12  # 12 months for yearly prediction
        elif predict_months:
            periods = predict_months
        else:
            periods = 6  # Default to 6 months
        
        # Check stationarity
        adf_test = adfuller(ts_df[value_col].dropna())
        is_stationary = adf_test[1] < 0.05
        
        # Fit ARIMA model
        # Using auto-selected parameters (1,1,1) as a reasonable default
        try:
            model = ARIMA(ts_df[value_col], order=(1, 1 if not is_stationary else 0, 1))
            fitted = model.fit()
            
            # Generate forecast
            forecast = fitted.forecast(steps=periods)
            
            # Generate future dates
            last_date = ts_df.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq='M'  # Monthly frequency
            )
            
            # Get confidence intervals
            forecast_df = fitted.get_forecast(steps=periods)
            conf_int = forecast_df.conf_int()
            
            return {
                "forecasted_values": [float(v) for v in forecast.tolist()],
                "dates": [str(d.strftime('%Y-%m-%d')) for d in future_dates],
                "confidence_intervals": {
                    "lower": [float(x) for x in conf_int.iloc[:, 0].tolist()],
                    "upper": [float(x) for x in conf_int.iloc[:, 1].tolist()]
                },
                "model_info": {
                    "aic": float(fitted.aic),
                    "bic": float(fitted.bic),
                    "is_stationary": bool(is_stationary)
                },
                "current_aqi": float(ts_df[value_col].iloc[-1]),
                "forecast_period": "year" if predict_year else f"{periods} months"
            }
            
        except Exception as e:
            print(f"ARIMA fitting error: {e}")
            
            # Fallback: Simple moving average prediction
            window = min(7, len(ts_df) // 2)
            ma = ts_df[value_col].rolling(window=window).mean()
            last_ma = ma.iloc[-1]
            
            future_dates = pd.date_range(
                start=ts_df.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq='M'
            )
            
            # Simple forecast with slight trend
            trend = (ts_df[value_col].iloc[-1] - ts_df[value_col].iloc[-window]) / window
            forecasted = [last_ma + (i * trend) for i in range(1, periods + 1)]
            
            return {
                "forecasted_values": forecasted,
                "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
                "model_info": {
                    "method": "Moving Average",
                    "window": window
                },
                "current_aqi": float(ts_df[value_col].iloc[-1]),
                "forecast_period": "year" if predict_year else f"{periods} months"
            }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None



def generate_visualizations(df):
    """Generate plots and return as base64"""
    
    plots = {}
    
    # Determine value column
    value_col = 'aqi' if 'aqi' in df.columns else 'value'
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col, 'city'])
    
    if len(df_clean) == 0:
        print("⚠ No valid data for visualizations")
        return {}
    
    try:
        # 1. Box Plot (works with single or multiple cities)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if len(df_clean['city'].unique()) > 1:
            # Multiple cities - boxplot by city
            df_clean.boxplot(column=value_col, by='city', ax=ax, grid=False, patch_artist=True)
            plt.suptitle('')
            ax.set_title('AQI Distribution by City', fontsize=14, fontweight='bold')
            ax.set_xlabel('City', fontsize=12)
            ax.set_ylabel('AQI Value', fontsize=12)
            plt.xticks(rotation=45, ha='right')
        else:
            # Single city - simple boxplot
            bp = ax.boxplot([df_clean[value_col]], patch_artist=True, labels=[df_clean['city'].iloc[0]])
            bp['boxes'][0].set_facecolor('skyblue')
            ax.set_title('AQI Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('AQI Value', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plots['boxplot'] = fig_to_base64(fig)
        
        # 2. Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_clean[value_col], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
        ax.set_xlabel('AQI Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('AQI Distribution Histogram', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plots['histogram'] = fig_to_base64(fig)
        
        # 3. Density Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        df_clean[value_col].plot(kind='density', ax=ax, color='green', linewidth=2)
        ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], alpha=0.2, color='green')
        ax.set_xlabel('AQI Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('AQI Density Plot', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plots['density'] = fig_to_base64(fig)
        
        # 4. Violin Plot (works with single or multiple cities)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if len(df_clean['city'].unique()) > 1:
            # Multiple cities
            sns.violinplot(data=df_clean, x='city', y=value_col, ax=ax, palette='Set2')
            plt.xticks(rotation=45, ha='right')
        else:
            # Single city
            sns.violinplot(data=df_clean, y=value_col, ax=ax, color='skyblue')
            ax.set_xlabel(df_clean['city'].iloc[0], fontsize=12)
        
        ax.set_title('AQI Violin Plot by City', fontsize=14, fontweight='bold')
        ax.set_ylabel('AQI Value', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plots['violin'] = fig_to_base64(fig)
        
        # 5. Scatter Plot (Time series if timestamp available)
        if 'timestamp' in df_clean.columns or 'date' in df_clean.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            date_col = 'timestamp' if 'timestamp' in df_clean.columns else 'date'
            df_plot = df_clean.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            df_plot = df_plot.sort_values(date_col)
            
            if len(df_clean['city'].unique()) > 1:
                # Multiple cities - different colors
                for city in df_plot['city'].unique():
                    city_data = df_plot[df_plot['city'] == city]
                    ax.scatter(city_data[date_col], city_data[value_col], label=city, alpha=0.6, s=50)
                ax.legend()
            else:
                # Single city
                ax.scatter(df_plot[date_col], df_plot[value_col], alpha=0.6, s=50, color='skyblue')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('AQI Value', fontsize=12)
            ax.set_title('AQI Over Time', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plots['scatter'] = fig_to_base64(fig)
        
        # 6. Contour Plot (if multiple cities with lat/lon)
        if len(df_clean['city'].unique()) > 1 and 'lat' in df_clean.columns and 'lon' in df_clean.columns:
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Group by city and get average AQI
                city_avg = df_clean.groupby('city').agg({
                    'lat': 'first',
                    'lon': 'first',
                    value_col: 'mean'
                }).reset_index()
                
                scatter = ax.scatter(
                    city_avg['lon'], 
                    city_avg['lat'], 
                    c=city_avg[value_col], 
                    s=500, 
                    cmap='RdYlGn_r', 
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=2
                )
                
                # Add city labels
                for idx, row in city_avg.iterrows():
                    ax.annotate(
                        row['city'], 
                        (row['lon'], row['lat']),
                        fontsize=9,
                        ha='center',
                        va='center',
                        fontweight='bold'
                    )
                
                plt.colorbar(scatter, ax=ax, label='Average AQI')
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.set_title('Geographic AQI Distribution', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plots['contour'] = fig_to_base64(fig)
            except Exception as e:
                print(f"Contour plot error: {e}")
        
        # 7. Correlation Heatmap
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        available_cols = [col for col in pollutant_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            corr_df = df[available_cols].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(corr_df) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = corr_df.corr()
                
                sns.heatmap(
                    corr_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": 0.8},
                    ax=ax,
                    vmin=-1,
                    vmax=1
                )
                
                ax.set_title('Pollutant Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plots['correlation_heatmap'] = fig_to_base64(fig)
        
        # 8. Hexagonal Binning (for large datasets)
        if len(df_clean) > 50 and 'timestamp' in df_clean.columns:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                df_hex = df_clean.copy()
                df_hex['timestamp'] = pd.to_datetime(df_hex['timestamp'])
                df_hex['days'] = (df_hex['timestamp'] - df_hex['timestamp'].min()).dt.days
                
                hexbin = ax.hexbin(
                    df_hex['days'], 
                    df_hex[value_col], 
                    gridsize=20, 
                    cmap='YlOrRd',
                    mincnt=1
                )
                
                plt.colorbar(hexbin, ax=ax, label='Count')
                ax.set_xlabel('Days from Start', fontsize=12)
                ax.set_ylabel('AQI Value', fontsize=12)
                ax.set_title('AQI Hexagonal Binning Plot', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plots['hexbin'] = fig_to_base64(fig)
            except Exception as e:
                print(f"Hexbin plot error: {e}")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"✓ Generated {len(plots)} visualizations: {list(plots.keys())}")
    return plots


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_base64

def generate_ai_summary(df, predictions=None):
    """Generate detailed summary with AQI interpretation and predictions"""
    stats = calculate_summary_statistics(df)
    variability = calculate_variability_metrics(df)
    
    if stats.get('count', 0) == 0:
        return "No data available for analysis."
    
    # Helper function to safely format values
    def fmt(value):
        return f"{value:.2f}" if value is not None else "N/A"
    
    # AQI Level Classification
    def get_aqi_category(aqi_value):
        if aqi_value is None:
            return "Unknown", "⚪"
        elif aqi_value <= 50:
            return "Good", "🟢"
        elif aqi_value <= 100:
            return "Satisfactory", "🟡"
        elif aqi_value <= 200:
            return "Moderate", "🟠"
        elif aqi_value <= 300:
            return "Poor", "🔴"
        elif aqi_value <= 400:
            return "Very Poor", "🟣"
        else:
            return "Severe", "🟤"
    
    # Get categories for mean and median
    mean_category, mean_emoji = get_aqi_category(stats.get('mean'))
    median_category, median_emoji = get_aqi_category(stats.get('median'))
    
    # Health implications
    health_advice = {
        "Good": "Air quality is satisfactory, and air pollution poses little or no risk.",
        "Satisfactory": "Air quality is acceptable. However, sensitive individuals should consider limiting prolonged outdoor exertion.",
        "Moderate": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        "Poor": "Everyone may begin to experience health effects. Members of sensitive groups may experience more serious effects.",
        "Very Poor": "Health alert: everyone may experience more serious health effects.",
        "Severe": "Health warnings of emergency conditions. The entire population is more likely to be affected."
    }
    
    # Determine dominant pollutant if available
    dominant_pollutant = "Not available"
    if 'dominant_pollutant' in df.columns:
        pollutant_counts = df['dominant_pollutant'].value_counts()
        if len(pollutant_counts) > 0:
            dominant_pollutant = pollutant_counts.index[0].upper()
    
    summary = f"""Analysis Summary for {stats['count']} records:

📊 Central Tendency:
   • Mean AQI: {fmt(stats.get('mean'))} {mean_emoji} ({mean_category})
   • Median AQI: {fmt(stats.get('median'))} {median_emoji} ({median_category})
   • Range: {fmt(stats.get('min'))} - {fmt(stats.get('max'))}

📈 Variability:
   • Standard Deviation: {fmt(variability.get('std_dev'))}
   • IQR (Interquartile Range): {fmt(variability.get('iqr'))}
   • Coefficient of Variation: {fmt(variability.get('cv'))}%

🔍 Data Characteristics:
   • The data shows {'HIGH' if variability.get('cv') and variability['cv'] > 30 else 'MODERATE'} variability in AQI levels
   • Distribution is {'RIGHT-SKEWED (mean > median)' if stats.get('mean') and stats.get('median') and stats['mean'] > stats['median'] else 'RELATIVELY SYMMETRIC'}
   • Dominant Pollutant: {dominant_pollutant}

🏥 Health Impact Assessment:
   Overall Air Quality Status: {mean_emoji} {mean_category.upper()}
   
   {health_advice.get(mean_category, 'No specific advice available.')}

💡 Key Insights:
   • {'🚨 ALERT: Air quality frequently exceeds safe levels!' if stats.get('mean', 0) > 200 else '✓ Air quality is generally within acceptable ranges.' if stats.get('mean', 0) <= 100 else '⚠️ Air quality is moderate and requires monitoring.'}
   • {'High day-to-day variation indicates inconsistent air quality.' if variability.get('cv', 0) > 40 else 'Relatively consistent air quality patterns observed.'}
"""
    
    # Add prediction analysis if available
    if predictions and not predictions.get('error'):
        forecast_values = predictions.get('forecasted_values', [])
        
        if forecast_values:
            avg_forecast = sum(forecast_values) / len(forecast_values)
            forecast_category, forecast_emoji = get_aqi_category(avg_forecast)
            current_aqi = predictions.get('current_aqi', stats.get('mean'))
            
            # Trend analysis
            if avg_forecast > current_aqi * 1.1:
                trend = "📈 WORSENING"
                trend_desc = "Air quality is predicted to deteriorate"
            elif avg_forecast < current_aqi * 0.9:
                trend = "📉 IMPROVING"
                trend_desc = "Air quality is predicted to improve"
            else:
                trend = "➡️ STABLE"
                trend_desc = "Air quality is expected to remain relatively stable"
            
            summary += f"""

🔮 Forecast Analysis ({predictions.get('forecast_period', 'N/A')}):
   Current AQI: {fmt(current_aqi)}
   Predicted Average AQI: {fmt(avg_forecast)} {forecast_emoji} ({forecast_category})
   
   Trend: {trend}
   {trend_desc}
   
   📅 Detailed Forecast:
"""
            
            # Add first 6 predictions with dates
            for i, (date, value) in enumerate(zip(predictions.get('dates', []), forecast_values)):
                if i < 6:  # Show first 6 months
                    cat, emoji = get_aqi_category(value)
                    summary += f"   • {date}: {fmt(value)} {emoji}\n"
            
            if len(forecast_values) > 6:
                summary += f"   ... and {len(forecast_values) - 6} more periods\n"
            
            # Add confidence intervals if available
            if 'confidence_intervals' in predictions:
                conf_lower = predictions['confidence_intervals']['lower']
                conf_upper = predictions['confidence_intervals']['upper']
                avg_lower = sum(conf_lower) / len(conf_lower)
                avg_upper = sum(conf_upper) / len(conf_upper)
                
                summary += f"""
   
   📊 Forecast Confidence (95%):
   • Lower Bound: {fmt(avg_lower)}
   • Upper Bound: {fmt(avg_upper)}
   • Uncertainty Range: ±{fmt((avg_upper - avg_lower) / 2)}
"""
            
            # Recommendations based on forecast
            summary += f"""
   
   💡 Recommendations:
"""
            if avg_forecast > 200:
                summary += """   • ⚠️ HIGH ALERT: Prepare for poor air quality
   • Consider reducing outdoor activities
   • Use air purifiers and masks
   • Monitor daily AQI updates closely
"""
            elif avg_forecast > 100:
                summary += """   • ⚠️ CAUTION: Moderate air quality expected
   • Sensitive groups should limit prolonged outdoor exposure
   • Keep windows closed during peak pollution hours
"""
            else:
                summary += """   • ✓ Air quality expected to remain acceptable
   • Continue regular outdoor activities
   • Maintain awareness of daily variations
"""
    
    elif predictions and predictions.get('error'):
        summary += f"""

🔮 Prediction Status:
   ⚠️ {predictions.get('error', 'Unable to generate forecast')}
"""
    
    return summary.strip()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
