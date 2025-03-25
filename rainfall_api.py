import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Prophet and RandomForest imports
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# FASTAPI APP CONFIGURATION
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def load_data_multi_city(csv_path: str) -> pd.DataFrame:
    """
    Loads a multi-city CSV file.
    For rainfall: expects 'city_name', 'date', 'rain_sum (mm)'.
    For temperature: expects at least 'city_name' and 'date'.
    Converts 'date' to datetime and renames it to 'ds'. 
    For rainfall, renames 'rain_sum (mm)' to 'y'.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if "city_name" not in df.columns or "date" not in df.columns:
        raise ValueError("CSV must contain 'city_name' and 'date' columns.")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.rename(columns={"date": "ds"}, inplace=True)
    df.sort_values("ds", inplace=True)
    
    # For rainfall, rename the rainfall column to 'y'
    if "rain_sum (mm)" in df.columns:
        df.rename(columns={"rain_sum (mm)": "y"}, inplace=True)
        df.dropna(subset=["ds", "y"], inplace=True)
    
    return df

def build_residual_dataset(df_city: pd.DataFrame, prophet_model: Prophet) -> pd.DataFrame:
    """
    Creates an in-sample Prophet forecast and computes the residual (y - yhat).
    Expects df_city to have columns 'ds' and 'y'.
    """
    forecast = prophet_model.predict(df_city[["ds"]])
    df_merged = df_city.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    df_merged["residual"] = df_merged["y"] - df_merged["yhat"]
    return df_merged

def create_lag_features(df_res: pd.DataFrame, target_col="residual", max_lag=3) -> pd.DataFrame:
    """
    Adds time-based features (day_of_year, day_of_week) and lagged versions
    of the target column.
    """
    df_lag = df_res.copy()
    df_lag["day_of_year"] = df_lag["ds"].dt.dayofyear
    df_lag["day_of_week"] = df_lag["ds"].dt.dayofweek

    for lag in range(1, max_lag + 1):
        df_lag[f"{target_col}_lag_{lag}"] = df_lag[target_col].shift(lag)

    df_lag.dropna(inplace=True)
    return df_lag

def iterative_residual_prediction(
    rf_model: RandomForestRegressor,
    df_history: pd.DataFrame,
    future_dates: pd.Series,
    feature_cols: list,
    max_lag: int = 3
) -> pd.DataFrame:
    """
    Iteratively predicts the residual for each future date using the RF model.
    """
    df_temp = df_history.copy()
    predictions = []

    for dt in future_dates:
        last_row = df_temp.iloc[-1].copy()
        # Update lag features: shift the residual lags forward
        for lag in range(1, max_lag + 1):
            if lag == 1:
                new_lag_val = last_row["residual"]
            else:
                new_lag_val = last_row[f"residual_lag_{lag-1}"]
            last_row[f"residual_lag_{lag}"] = new_lag_val

        last_row["ds"] = dt
        last_row["day_of_year"] = dt.day_of_year
        last_row["day_of_week"] = dt.day_of_week

        X_new = pd.DataFrame([last_row[feature_cols]], columns=feature_cols)
        resid_pred = rf_model.predict(X_new)[0]

        new_row = last_row.copy()
        new_row["ds"] = dt
        new_row["residual"] = resid_pred
        df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)

        predictions.append({"ds": dt, "residual_pred": resid_pred})

    return pd.DataFrame(predictions)

def create_hybrid_forecast(
    prophet_model: Prophet,
    rf_model: RandomForestRegressor,
    df_supervised: pd.DataFrame,
    feature_cols: list,
    target_year: int,
    max_lag: int
) -> pd.DataFrame:
    """
    Combines a Prophet forecast with a residual correction from an RF model.
    """
    last_date = df_supervised["ds"].max()
    end_of_year = pd.to_datetime(f"{target_year}-12-31")
    days_needed = (end_of_year - last_date).days

    # If your data extends beyond the requested year, 
    # you can either allow partial overlap or raise a ValueError.
    # Here we raise a ValueError if the year is not strictly in the future:
    if days_needed < 1:
        raise ValueError(
            f"Requested year {target_year} ends on {end_of_year.strftime('%Y-%m-%d')}, "
            f"but last training date is {last_date.strftime('%Y-%m-%d')}."
        )

    future_df = prophet_model.make_future_dataframe(periods=days_needed, freq="D")
    full_fc = prophet_model.predict(future_df)

    start_of_year = pd.to_datetime(f"{target_year}-01-01")
    fc_year = full_fc[(full_fc["ds"] >= start_of_year) & (full_fc["ds"] <= end_of_year)].copy()

    future_dates = fc_year["ds"].sort_values().unique()

    # Take the last row of your supervised set as the "history" seed
    df_history_for_resid = df_supervised.iloc[[-1]].copy()
    df_pred = iterative_residual_prediction(
        rf_model=rf_model,
        df_history=df_history_for_resid,
        future_dates=future_dates,
        feature_cols=feature_cols,
        max_lag=max_lag
    )

    fc_year = fc_year.merge(df_pred, on="ds", how="left")
    fc_year["residual_pred"] = fc_year["residual_pred"].fillna(0)
    fc_year["yhat_hybrid"] = fc_year["yhat"] + fc_year["residual_pred"]

    return fc_year

# -----------------------------------------------------------------------------
# RAINFALL FORECAST ENDPOINT
# -----------------------------------------------------------------------------
CSV_PATH_RAIN = "dataset/city_base_weather.csv"  # <-- UPDATE to your actual path
df_all_cities = None  # Will be loaded at startup
MAX_LAG = 3

class HybridForecastResponse(BaseModel):
    forecast: list  # list of forecast records

@app.on_event("startup")
def startup_event():
    """
    Load the rainfall CSV once at startup.
    """
    global df_all_cities
    df_all_cities = load_data_multi_city(CSV_PATH_RAIN)
    if df_all_cities.empty:
        raise RuntimeError("Loaded rainfall DataFrame is empty. Check your CSV file and path.")

@app.get("/forecast", response_model=HybridForecastResponse)
def get_city_forecast(city: str, year: int):
    """
    Returns a daily rainfall forecast for the given city and year.
    Example: GET /forecast?city=Kurunegala&year=2027
    """
    global df_all_cities

    df_city = df_all_cities[df_all_cities["city_name"] == city].copy()
    if df_city.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"No rainfall data found for city '{city}'"
        )

    model_dir = "models"  # <-- UPDATE to your actual path for rainfall models
    prophet_model_path = os.path.join(model_dir, f"{city}_prophet_model.pkl")
    rf_model_path = os.path.join(model_dir, f"{city}_rf_residual_model.pkl")

    if not os.path.isfile(prophet_model_path) or not os.path.isfile(rf_model_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Could not find saved models for city '{city}'. "
                f"Expected files: {prophet_model_path} and {rf_model_path}"
            )
        )

    try:
        prophet_model = joblib.load(prophet_model_path)
        rf_model = joblib.load(rf_model_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error loading models: {str(e)}"
        )

    # Prepare a residual dataset (y - yhat) from historical
    try:
        df_residual = build_residual_dataset(df_city, prophet_model)
        df_supervised = create_lag_features(df_residual, target_col="residual", max_lag=MAX_LAG)
        feature_cols = [c for c in df_supervised.columns if "lag_" in c or "day_of_" in c]
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error preparing supervised data: {str(e)}"
        )

    if df_supervised.empty:
        raise HTTPException(
            status_code=500,
            detail="Residual dataset is empty after creating lag features. Possibly too few rows."
        )

    # Generate the hybrid forecast
    try:
        hybrid_fc = create_hybrid_forecast(
            prophet_model=prophet_model,
            rf_model=rf_model,
            df_supervised=df_supervised,
            feature_cols=feature_cols,
            target_year=year,
            max_lag=MAX_LAG
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating rainfall forecast: {str(e)}"
        )

    forecast_json = hybrid_fc.to_dict(orient="records")
    return {"forecast": forecast_json}

# -----------------------------------------------------------------------------
# TEMPERATURE FORECAST ENDPOINT
# -----------------------------------------------------------------------------
CSV_PATH_TEM = "dataset/city_base_Tem.csv"  # <-- UPDATE to your actual path
df_all_cities_tem = None
MAX_LAG_TEM = 3

def load_temperature_data() -> pd.DataFrame:
    """
    Loads the temperature CSV and returns a DataFrame.
    Expects columns: 'city_name', 'date' (converted to 'ds'), 
    and temperature columns: 'temperature_2m_max (°C)', 'temperature_2m_mean (°C)', 
    'temperature_2m_min (°C)'.
    """
    global df_all_cities_tem
    if df_all_cities_tem is None:
        df_all_cities_tem = load_data_multi_city(CSV_PATH_TEM)
        if df_all_cities_tem.empty:
            raise RuntimeError("Loaded temperature DataFrame is empty. Check your CSV file and path.")
    return df_all_cities_tem

@app.get("/tem_forecast")
def get_temperature_forecast(city: str, year: int):
    """
    Returns a 1-year forecast for max, mean, and min temperature for the given city and year.
    Example: GET /tem_forecast?city=Anuradhapura&year=2026
    """
    df_tem = load_temperature_data()

    df_city = df_tem[df_tem["city_name"] == city].copy()
    if df_city.empty:
        raise HTTPException(status_code=404, detail=f"No temperature data found for city '{city}'")

    # Use separate mappings for CSV column names and model file name fragments.
    csv_map = {
        "max": "temperature_2m_max (°C)",
        "mean": "temperature_2m_mean (°C)",
        "min": "temperature_2m_min (°C)",
    }
    model_map = {
        "max": "temperature_2m_max_C",
        "mean": "temperature_2m_mean_C",
        "min": "temperature_2m_min_C",
    }
    model_dir_tem = "models_tem"  # <-- UPDATE to your actual path for temp models
    merged_df = None

    for temp_type in ["max", "mean", "min"]:
        csv_col = csv_map[temp_type]
        model_col = model_map[temp_type]
        
        # Prepare data for this temperature type: select CSV column, rename to 'y'
        df_var = df_city[["ds", csv_col]].copy()
        df_var.rename(columns={csv_col: "y"}, inplace=True)

        # Construct expected model file paths
        prophet_path = os.path.join(model_dir_tem, f"{city}_prophet_{model_col}.pkl")
        rf_path = os.path.join(model_dir_tem, f"{city}_rf_residual_{model_col}.pkl")

        if not (os.path.isfile(prophet_path) and os.path.isfile(rf_path)):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Could not find temperature models for '{temp_type}' in city '{city}'. "
                    f"Expected: {prophet_path} and {rf_path}"
                )
            )

        try:
            prophet_model = joblib.load(prophet_path)
            rf_model = joblib.load(rf_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading {temp_type} models: {str(e)}"
            )

        # Build the residual dataset and create lags
        try:
            df_residual = build_residual_dataset(df_var, prophet_model)
            df_supervised = create_lag_features(df_residual, target_col="residual", max_lag=MAX_LAG_TEM)
            feature_cols = [c for c in df_supervised.columns if "lag_" in c or "day_of_" in c]
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error preparing supervised data for {temp_type} temperature: {str(e)}"
            )

        if df_supervised.empty:
            raise HTTPException(
                status_code=500,
                detail=f"Residual dataset is empty for {temp_type} temperature. Possibly too few rows?"
            )

        # Generate the hybrid forecast for this temperature type
        try:
            hybrid_fc = create_hybrid_forecast(
                prophet_model=prophet_model,
                rf_model=rf_model,
                df_supervised=df_supervised,
                feature_cols=feature_cols,
                target_year=year,
                max_lag=MAX_LAG_TEM
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating {temp_type} temperature forecast: {str(e)}"
            )

        # Rename forecast columns to include temperature type as a prefix
        subset = hybrid_fc[["ds", "yhat", "yhat_lower", "yhat_upper", "yhat_hybrid"]].copy()
        subset.rename(columns={
            "yhat": f"{temp_type}_yhat",
            "yhat_lower": f"{temp_type}_yhat_lower",
            "yhat_upper": f"{temp_type}_yhat_upper",
            "yhat_hybrid": f"{temp_type}_yhat_hybrid",
        }, inplace=True)

        # Merge each temperature type's forecast columns into a single DataFrame
        if merged_df is None:
            merged_df = subset
        else:
            merged_df = pd.merge(merged_df, subset, on="ds", how="inner")

    forecast_json = merged_df.to_dict(orient="records")
    return {"forecast": forecast_json}

# -----------------------------------------------------------------------------
# RUN THE APP (if you run this file directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
