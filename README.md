# ecoaware
Delhi Climate and AQI Monitoring Dashboard

## Run the project

1. Install Python dependencies:
   `pip install -r requirements.txt`
2. Start the Flask server:
   `python backend/api/server.py`
3. Open:
   `http://127.0.0.1:5000`

This setup uses:
- Python Flask backend
- Plain HTML, CSS, and JavaScript frontend
- multi-page routes for Home, AQI, Temperature, Policy Insights, and Contact
- `/api/aqi` API endpoint for AQI data and prediction
- `datasets/Data_training/AQI_merged_all.csv` as the AQI source
- Random Forest prediction by default, with support for model files later

## Optional model files

If you later export trained models, place them here:

- `backend/models/random_forest_model.joblib`
- `backend/models/decision_tree_model.joblib`

If those files are not present, the app trains a Random Forest model directly from your AQI dataset.
