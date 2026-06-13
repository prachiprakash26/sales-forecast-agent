from src.forecasting.prophet_model import forecast_sales as model_forecast_sales

def forecast_sales(
    product: str = None,
    store: str = None,
    horizon_weeks: int = 12
) -> dict:
    """
    Generates a forecast for the specified product and store over a weekly horizon (12, 26, or 52 weeks).
    Uses Meta Prophet under the hood (with ExponentialSmoothing fallback).
    
    Returns a dictionary summarizing forecast results and predictions.
    """
    # Call core forecasting model logic
    results = model_forecast_sales(
        product=product,
        store=store,
        horizon_weeks=horizon_weeks
    )
    
    # Format the forecast output for LLM consumption
    forecast_df = results["forecast_df"]
    forecast_only = forecast_df[forecast_df['actual_sales'].isna()]
    
    # Select key columns and round them for readability
    weekly_predictions = []
    for _, row in forecast_only.iterrows():
        weekly_predictions.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "forecasted_sales": round(row["forecasted_sales"], 2),
            "lower_bound": round(row["lower_bound"], 2),
            "upper_bound": round(row["upper_bound"], 2)
        })
        
    return {
        "store": store,
        "product": product,
        "horizon_weeks": horizon_weeks,
        "growth_pct": round(results["growth_pct"], 4),
        "forecast_summary": results["forecast_summary"],
        "forecast_predictions": weekly_predictions  # concise weekly forecast table
    }
