import os
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback capability if Prophet is not installed or has compilation issues
try:
    from prophet import Prophet
    HAS_PROPHET = True
    logger.info("Prophet successfully imported.")
except ImportError:
    HAS_PROPHET = False
    logger.warning("Prophet not found. Falling back to Statsmodels Holt-Winters Exponential Smoothing.")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_sales(
    product: str = None,
    store: str = None,
    horizon_weeks: int = 12,
    data_path: str = "/workspace/shared/sales-forecast-agent/data/mock_sales.csv"
) -> dict:
    """
    Fits a forecasting model on historical weekly sales data for a specific store
    and/or product, and forecasts sales for a specified weekly horizon (12, 26, or 52 weeks).
    
    Returns a dictionary containing:
        - "forecast_df": pd.DataFrame with columns: [date, actual_sales, forecasted_sales, lower_bound, upper_bound]
        - "growth_pct": float representing the growth/decline rate compared to recent history
        - "forecast_summary": str narrative summary of the model results.
    """
    if not os.path.exists(data_path):
        from src.utils.helpers import get_data_path
        resolved_path = get_data_path("mock_sales.csv")
        if os.path.exists(resolved_path):
            data_path = resolved_path
        else:
            raise FileNotFoundError(f"Data file not found at {data_path} or {resolved_path}. Please generate mock data first.")
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data based on parameters
    filtered_df = df.copy()
    if store is not None and store != "":
        filtered_df = filtered_df[filtered_df['store'] == store]
    if product is not None and product != "":
        filtered_df = filtered_df[filtered_df['product'] == product]
        
    if filtered_df.empty:
        raise ValueError(f"No data found matching store={store} and product={product}.")
        
    # Group by date to aggregate sales across filtered segments (e.g. if store or product is None)
    agg_df = filtered_df.groupby('date')['sales'].sum().reset_index()
    agg_df = agg_df.sort_values('date')
    
    # Define time-series data for modeling
    hist_df = agg_df.copy()
    
    if HAS_PROPHET:
        # Prepare for Prophet: columns ds and y
        prophet_df = hist_df.rename(columns={'date': 'ds', 'sales': 'y'})
        
        # Initialize and fit Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=horizon_weeks, freq='W')
        forecast = model.predict(future)
        
        # Merge back with actuals
        merged = pd.merge(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            prophet_df[['ds', 'y']],
            on='ds',
            how='left'
        )
        
        # Format output dataframe
        forecast_df = merged.rename(columns={
            'ds': 'date',
            'y': 'actual_sales',
            'yhat': 'forecasted_sales',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        })
        
    else:
        # Fallback to Statsmodels Holt-Winters Exponential Smoothing
        hist_series = hist_df.set_index('date')['sales']
        # Weekly data, seasonal period = 52 weeks
        model = ExponentialSmoothing(
            hist_series,
            seasonal_periods=52,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        ).fit()
        
        # Generate out-of-sample forecast
        future_dates = pd.date_range(
            start=hist_series.index[-1] + pd.Timedelta(weeks=1),
            periods=horizon_weeks,
            freq='W'
        )
        
        forecast_values = model.forecast(horizon_weeks)
        
        # Construct standard error approximations (simulate bounds)
        residuals = model.resid
        std_err = np.std(residuals)
        
        # Build forecasted dataframe
        forecast_part = pd.DataFrame({
            'date': future_dates,
            'actual_sales': np.nan,
            'forecasted_sales': forecast_values.values,
            'lower_bound': np.maximum(0, forecast_values.values - 1.96 * std_err),
            'upper_bound': forecast_values.values + 1.96 * std_err
        })
        
        hist_part = pd.DataFrame({
            'date': hist_series.index,
            'actual_sales': hist_series.values,
            'forecasted_sales': model.fittedvalues.values,
            'lower_bound': np.maximum(0, model.fittedvalues.values - 1.96 * std_err),
            'upper_bound': model.fittedvalues.values + 1.96 * std_err
        })
        
        forecast_df = pd.concat([hist_part, forecast_part], ignore_index=True)
    
    # Calculate Growth Percentage
    # We compare the average forecasted weekly sales over the horizon
    # to the average weekly sales over the last matching historical horizon (e.g. last N weeks)
    hist_actuals = forecast_df[forecast_df['actual_sales'].notna()]
    forecast_only = forecast_df[forecast_df['actual_sales'].isna()]
    
    recent_hist_avg = hist_actuals['actual_sales'].tail(horizon_weeks).mean()
    forecast_avg = forecast_only['forecasted_sales'].mean()
    
    if recent_hist_avg > 0:
        growth_pct = float((forecast_avg - recent_hist_avg) / recent_hist_avg)
    else:
        growth_pct = 0.0
        
    # Generate Narrative Summary
    store_str = store if store else "All Stores"
    product_str = product if product else "All Products"
    
    trend_direction = "an increase" if growth_pct > 0 else "a decrease"
    growth_sign = "+" if growth_pct > 0 else ""
    
    summary = (
        f"Forecasting results for {store_str} - {product_str} over the next {horizon_weeks} weeks:\n"
        f"- Estimated Average Weekly Demand: {forecast_avg:,.2f} units.\n"
        f"- Projected sales trend shows {trend_direction} of {growth_sign}{growth_pct:.2%} "
        f"compared to the historical average of the last {horizon_weeks} weeks ({recent_hist_avg:,.2f} units).\n"
        f"- 95% Confidence Interval boundaries: [{forecast_only['lower_bound'].mean():,.2f}, {forecast_only['upper_bound'].mean():,.2f}] units."
    )
    
    return {
        "forecast_df": forecast_df,
        "growth_pct": growth_pct,
        "forecast_summary": summary
    }
