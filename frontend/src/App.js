import axios from 'axios';
import React, { useEffect, useState } from 'react';
import Select from 'react-select';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

function App() {
  const [cities, setCities] = useState([]);
  const [selectedCities, setSelectedCities] = useState([]);
  const [year, setYear] = useState(2024);
  const [predictOption, setPredictOption] = useState('none');
  const [predictValue, setPredictValue] = useState(6);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Generate years 2010-2025
  const years = Array.from({ length: 11 }, (_, i) => 2015 + i);

  useEffect(() => {
    // Fetch available cities from backend
    axios.get('${API_BASE_URL}/api/cities')
      .then(res => {
        const cityOptions = res.data.cities.map(city => ({
          value: city,
          label: city
        }));
        setCities(cityOptions);
      })
      .catch(err => {
        console.error('Failed to fetch cities:', err);
        // Fallback cities
        const fallbackCities = [
          'Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata',
          'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'
        ].map(city => ({ value: city, label: city }));
        setCities(fallbackCities);
      });
  }, []);

  const handleAnalyze = async () => {
    if (selectedCities.length === 0) {
      setError('Please select at least one city');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const payload = {
        cities: selectedCities.map(c => c.value),
        year: year,
        predict_months: predictOption === 'months' ? (predictValue || 6) : null,
        predict_year: predictOption === 'year' ? 1 : null
      };

      console.log('Sending request:', payload); // Debug log

      const response = await axios.post('${API_BASE_URL}/api/analyze', payload, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30000, // 30 second timeout
      });

      console.log('Response received:', response.data); // Debug log
      setResults(response.data);

    } catch (error) {
      console.error('Full error:', error); // Debug log

      if (error.code === 'ERR_NETWORK') {
        setError('Cannot connect to backend. Make sure it is running on http://localhost:8000');
      } else if (error.response) {
        setError('Analysis failed: ' + (error.response?.data?.detail || error.message));
      } else {
        setError('Network error: ' + error.message);
      }
    }

    setLoading(false);
  };

  return (
    <div className="App">
      <header>
        <h1>🌍 India AQI Analysis Dashboard</h1>
        <p>Real-time Air Quality Index Analysis & Forecasting</p>
      </header>

      <div className="controls-panel">
        <div className="control-group">
          <label>1. Select Cities:</label>
          <Select
            isMulti
            options={cities}
            value={selectedCities}
            onChange={setSelectedCities}
            placeholder="Select one or more cities..."
            className="city-select"
          />
        </div>

        <div className="control-group">
          <label>2. Select Year:</label>
          <select value={year} onChange={(e) => setYear(parseInt(e.target.value))}>
            {years.map(y => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>3. Prediction Option:</label>
          <select value={predictOption} onChange={(e) => {
            setPredictOption(e.target.value);
            if (e.target.value === 'none') {
              setPredictValue(null); // Set to null instead of keeping old value
            } else if (e.target.value === 'months') {
              setPredictValue(6); // Default 6 months
            } else if (e.target.value === 'year') {
              setPredictValue(12); // 12 months for a year
            }
          }}>
            <option value="none">No Prediction</option>
            <option value="months">Next N Months</option>
            <option value="year">Next Year</option>
          </select>

          {predictOption === 'months' && (
            <input
              type="number"
              min="1"
              max="24"
              value={predictValue || 6}
              onChange={(e) => setPredictValue(parseInt(e.target.value) || 6)}
              placeholder="Number of months"
              style={{ marginTop: '10px' }}
            />
          )}
        </div>

        <button
          className="analyze-btn"
          onClick={handleAnalyze}
          disabled={loading || selectedCities.length === 0}
        >
          {loading ? '⏳ Analyzing...' : '🚀 Run Analysis'}
        </button>

        {error && <div className="error-message">{error}</div>}
      </div>

      {results && (
        <div className="results-container">
          <h2>📊 Analysis Results</h2>

          {/* Summary Statistics */}
          <section className="stats-section">
            <h3>Summary Statistics</h3>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-label">Mean</span>
                <span className="stat-value">{results.summary_stats.mean?.toFixed(2)}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Median</span>
                <span className="stat-value">{results.summary_stats.median?.toFixed(2)}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Std Dev</span>
                <span className="stat-value">{results.variability_metrics.std_dev?.toFixed(2)}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">IQR</span>
                <span className="stat-value">{results.variability_metrics.iqr?.toFixed(2)}</span>
              </div>
            </div>
          </section>

          {/* Correlation Matrix */}
          {results.correlation_matrix && results.correlation_matrix.columns && (
            <section className="stats-section">
              <h3>🔗 Correlation Matrix</h3>
              <div className="correlation-info">
                <p>Shows relationships between different pollutants (PM2.5, PM10, O3, NO2, SO2, CO)</p>
                <ul>
                  <li><strong>1.0:</strong> Perfect positive correlation</li>
                  <li><strong>0.0:</strong> No correlation</li>
                  <li><strong>-1.0:</strong> Perfect negative correlation</li>
                </ul>
              </div>
              <div className="correlation-table-container">
                <table className="correlation-table">
                  <thead>
                    <tr>
                      <th></th>
                      {results.correlation_matrix.columns.map((col, idx) => (
                        <th key={idx}>{col.toUpperCase()}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.correlation_matrix.data.map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        <th>{results.correlation_matrix.columns[rowIdx].toUpperCase()}</th>
                        {row.map((value, colIdx) => (
                          <td
                            key={colIdx}
                            className={`corr-cell ${value >= 0.7 ? 'high-positive' :
                              value >= 0.4 ? 'medium-positive' :
                                value >= 0 ? 'low-positive' :
                                  value >= -0.4 ? 'low-negative' :
                                    value >= -0.7 ? 'medium-negative' : 'high-negative'
                              }`}
                          >
                            {value?.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Visualizations */}
          <section className="viz-section">
            <h3>📈 Visualizations</h3>
            <div className="viz-grid">
              {Object.entries(results.visualizations || {}).map(([name, img]) => (
                <div key={name} className="viz-card">
                  <h4>{name.replace(/_/g, ' ').toUpperCase()}</h4>
                  <img src={`data:image/png;base64,${img}`} alt={name} />
                </div>
              ))}
            </div>
          </section>

          {/* AI Summary */}
          <section className="summary-section">
            <h3>🤖 AI-Generated Summary</h3>
            <div className="summary-text">
              <pre>{results.ai_summary}</pre>
            </div>
          </section>

          {/* Predictions */}
          {results.predictions && !results.predictions.error && (
            <section className="predictions-section">
              <h3>🔮 Forecasted Values - {results.predictions.forecast_period}</h3>

              {results.predictions.current_aqi && (
                <div className="current-aqi">
                  <p><strong>Current AQI:</strong> {results.predictions.current_aqi.toFixed(2)}</p>
                </div>
              )}

              <div className="forecast-grid">
                {results.predictions.forecasted_values.map((val, idx) => {
                  const date = results.predictions.dates[idx];
                  const getAqiColor = (aqi) => {
                    if (aqi <= 50) return '#4caf50';
                    if (aqi <= 100) return '#ffeb3b';
                    if (aqi <= 200) return '#ff9800';
                    if (aqi <= 300) return '#f44336';
                    if (aqi <= 400) return '#9c27b0';
                    return '#795548';
                  };

                  return (
                    <div key={idx} className="forecast-card" style={{ borderLeftColor: getAqiColor(val) }}>
                      <div className="forecast-date">{date}</div>
                      <div className="forecast-value" style={{ color: getAqiColor(val) }}>
                        {val.toFixed(2)}
                      </div>
                      {results.predictions.confidence_intervals && (
                        <div className="forecast-range">
                          ±{((results.predictions.confidence_intervals.upper[idx] - results.predictions.confidence_intervals.lower[idx]) / 2).toFixed(1)}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {results.predictions.model_info && (
                <div className="model-info">
                  <p><strong>Model:</strong> {results.predictions.model_info.method || 'ARIMA Time Series'}</p>
                  {results.predictions.model_info.aic && (
                    <p><strong>Model Accuracy (AIC):</strong> {results.predictions.model_info.aic.toFixed(2)}</p>
                  )}
                </div>
              )}
            </section>
          )}

          {results.predictions && results.predictions.error && (
            <section className="predictions-section">
              <h3>🔮 Forecast Status</h3>
              <div className="error-message">{results.predictions.error}</div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
