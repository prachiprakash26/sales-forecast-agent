import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils.helpers import get_data_path

def explain_forecast_drivers(product: str = None, store: str = None) -> dict:
    """
    Fits a linear model to historical sales to estimate the relative contribution 
    of: trend, promotions, holidays, price, and seasonality.
    
    Returns a dictionary of driver effects and percentage contributions.
    """
    data_path = get_data_path("mock_sales.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter dataset
    filtered_df = df.copy()
    if store is not None:
        filtered_df = filtered_df[filtered_df['store'] == store]
    if product is not None:
        filtered_df = filtered_df[filtered_df['product'] == product]
        
    if filtered_df.empty:
        return {
            "error": f"No data found matching store={store} and product={product}."
        }
        
    # Group by date to aggregate if multiple stores or products are included
    agg_df = filtered_df.groupby('date').agg({
        'sales': 'sum',
        'price': 'mean',
        'promotion_flag': 'max',  # 1 if any store had promo
        'holiday_flag': 'max',
    }).reset_index()
    agg_df = agg_df.sort_values('date')
    
    n_rows = len(agg_df)
    if n_rows < 10:
        return {
            "error": "Not enough data points to run driver decomposition."
        }
        
    # Feature Engineering
    # 1. Trend feature (linear sequence)
    agg_df['trend'] = np.arange(1, n_rows + 1)
    
    # 2. Seasonality features (annual sine and cosine of week number)
    weeks = agg_df['date'].dt.isocalendar().week
    agg_df['sin_season'] = np.sin(2 * np.pi * weeks / 52.0)
    agg_df['cos_season'] = np.cos(2 * np.pi * weeks / 52.0)
    
    # Target and predictor matrix
    X_cols = ['trend', 'promotion_flag', 'holiday_flag', 'price', 'sin_season', 'cos_season']
    X = agg_df[X_cols]
    y = agg_df['sales']
    
    # Fit OLS Linear Regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate effects: coefficient * mean(feature_value) or coefficient * standard deviation
    # Using absolute contribution of each feature to the variance/magnitude of prediction
    coefs = model.coef_
    intercept = model.intercept_
    
    # Calculate absolute contributions of each feature
    # For seasonality, combine sine and cosine
    trend_vals = coefs[0] * X['trend']
    promo_vals = coefs[1] * X['promotion_flag']
    holiday_vals = coefs[2] * X['holiday_flag']
    price_vals = coefs[3] * X['price']
    season_vals = coefs[4] * X['sin_season'] + coefs[5] * X['cos_season']
    
    # Compute the average absolute impact of each driver
    trend_impact = float(np.mean(np.abs(trend_vals)))
    promo_impact = float(np.mean(np.abs(promo_vals)))
    holiday_impact = float(np.mean(np.abs(holiday_vals)))
    price_impact = float(np.mean(np.abs(price_vals)))
    season_impact = float(np.mean(np.abs(season_vals)))
    
    total_impact = trend_impact + promo_impact + holiday_impact + price_impact + season_impact
    if total_impact == 0:
        total_impact = 1.0
        
    # Relative percentages
    trend_pct = trend_impact / total_impact
    promo_pct = promo_impact / total_impact
    holiday_pct = holiday_impact / total_impact
    price_pct = price_impact / total_impact
    season_pct = season_impact / total_impact
    
    # Model coefficients summary (unit changes)
    coef_summary = {
        "trend_weekly_change": round(float(coefs[0]), 4),
        "promotion_uplift_units": round(float(coefs[1]), 2),
        "holiday_uplift_units": round(float(coefs[2]), 2),
        "price_coefficient": round(float(coefs[3]), 2), # unit change in sales per dollar increase
    }
    
    # Construct business explanations
    price_direction = "decreases" if coefs[3] < 0 else "increases"
    explanation = (
        f"Forecast Driver Analysis (R-squared: {model.score(X, y):.2f}):\n"
        f"- **Trend**: Explains {trend_pct:.1%} of sales variance. Baseline weekly growth is {coefs[0]:+.2f} units.\n"
        f"- **Price**: Explains {price_pct:.1%} of sales variance. A $1 price increase {price_direction} weekly sales by {abs(coefs[3]):.2f} units.\n"
        f"- **Promotions**: Explains {promo_pct:.1%} of sales variance. Running promotions adds an average of {coefs[1]:.2f} units per week.\n"
        f"- **Seasonality**: Explains {season_pct:.1%} of sales variance, capturing annual peak-and-trough cycles.\n"
        f"- **Holidays**: Explains {holiday_pct:.1%} of sales variance. Holiday weeks boost sales by an average of {coefs[2]:.2f} units."
    )
    
    return {
        "store": store,
        "product": product,
        "model_fit_r2": round(float(model.score(X, y)), 4),
        "driver_percentages": {
            "trend": round(trend_pct, 4),
            "pricing": round(price_pct, 4),
            "promotions": round(promo_pct, 4),
            "seasonality": round(season_pct, 4),
            "holidays": round(holiday_pct, 4)
        },
        "coefficients": coef_summary,
        "business_explanation": explanation
    }
