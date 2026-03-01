# Dashboard Setup & Usage Guide

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard**:
   ```bash
   streamlit run src/dashboard.py
   ```
   
   Or use the batch script on Windows:
   ```bash
   run_dashboard.bat
   ```

   The dashboard will open at `http://localhost:8501`

### Features

The Malaysia Blood Donations Dashboard provides:

- **📈 Time Series Tab**: Visualize daily donation trends with interactive date range filters
- **🗺️ State Comparison Tab**: Bar chart comparing total donations across all states
- **🔴 Blood Type Analysis Tab**: Pie chart and table showing blood type distribution
- **📅 Patterns & Trends Tab**: 
  - Weekly patterns showing donation trends by day of week
  - State × Blood Type heatmap for detailed cross-analysis

### Filters (Sidebar)

- **States**: Multi-select filter for choosing one or more states
- **Blood Types**: Filter by blood type (A, B, O, AB, all)
- **Date Range**: Select custom date range for analysis
- **Summary Statistics**: Real-time stats including total, average, and peak donations

### Data Management

- **Automatic Download**: On first run, the app automatically downloads the latest blood donation data from data.gov.my
- **Caching**: Downloaded data is cached in `tmp/blood_donations.pkl` for fast subsequent loads
- **Smart Refresh**: Data is loaded from cache for speed; modify the pickle file to refresh with latest data

---

## Docker Deployment

### Single Dashboard Service

Build and run just the dashboard:

```bash
docker build -f Dockerfile.dashboard -t blood-donations-dashboard .
docker run -p 8501:8501 blood-donations-dashboard
```

### Full Stack (API + Dashboard)

Run both API and Dashboard services together:

```bash
docker-compose up -d
```

Access:
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## Architecture

### Component Overview

```
src/dashboard.py
├── Data Loading: Downloads from data.gov.my or loads cached pickle
├── Filtering: State, Blood Type, Date Range
├── Visualization: 4 tabs with matplotlib charts
└── Caching: Streamlit @st.cache_resource for performance
```

### Key Dependencies

- **Streamlit**: Web app framework (lightweight, fast)
- **Polars**: Fast data processing
- **Matplotlib**: Visualization
- **Pickle**: Data serialization for caching

---

## File Structure

```
.
├── src/
│   ├── dashboard.py                  # Main Streamlit app
│   └── donations/
│       └── setup_and_validation.py   # Data download & validation
├── tests/
│   └── test_dashboard_setup.py       # Dashboard setup test
├── Dockerfile                        # API container
├── Dockerfile.dashboard              # Dashboard container
├── docker-compose.yml                # Multi-service orchestration
├── run_dashboard.bat                 # Windows batch script
├── .streamlit/
│   └── config.toml                   # Streamlit configuration
└── tmp/
    └── blood_donations.pkl           # Cached data (auto-created)
```

---

## Configuration

Streamlit settings are in `.streamlit/config.toml`:
- Theme colors (red for primary, white background)
- Max upload size: 200MB
- CSRF protection: enabled

---

## Troubleshooting

**Dashboard won't start?**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check port 8501 is not in use: `netstat -ano | findstr 8501` (Windows)

**Data not loading?**
- Check internet connection (for first-time download)
- Verify `tmp/` directory exists
- Clear cache and restart: delete `tmp/blood_donations.pkl`

**Slow dashboard?**
- Data is cached in pickle; subsequent loads should be fast
- If slow, consider filtering to smaller date range or fewer states

---

## Future Enhancements

Possible additions:
- Real-time predictions using the ML model from the API
- Data drift detection indicators
- State-level trends with forecasting
- Export filtered data to CSV/Excel
- Donation alerts/notifications
