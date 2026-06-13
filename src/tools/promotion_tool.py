import os
import pandas as pd
import numpy as np
from src.utils.helpers import get_data_path

def analyze_promotion_impact(product: str = None, store: str = None) -> dict:
    """
    Analyzes the sales impact of promotions for a specific product and/or store.
    Calculates average sales during promotional and non-promotional periods,
    uplift percentages, and estimates incremental revenue.
    
    Returns a dictionary of promotional KPIs.
    """
    data_path = get_data_path("mock_sales.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply filtering
    filtered_df = df.copy()
    if store is not None and store != "":
        filtered_df = filtered_df[filtered_df['store'] == store]
    if product is not None and product != "":
        filtered_df = filtered_df[filtered_df['product'] == product]
        
    if filtered_df.empty:
        return {
            "error": f"No data found matching store={store} and product={product}."
        }
        
    # Split into promo and non-promo groups
    promo_df = filtered_df[filtered_df['promotion_flag'] == 1]
    non_promo_df = filtered_df[filtered_df['promotion_flag'] == 0]
    
    if promo_df.empty:
        return {
            "message": "No promotional weeks found in the filtered dataset.",
            "average_promoted_sales": 0.0,
            "average_non_promoted_sales": float(non_promo_df['sales'].mean()) if not non_promo_df.empty else 0.0,
            "uplift_pct": 0.0,
            "incremental_revenue": 0.0
        }
        
    avg_promo_sales = float(promo_df['sales'].mean())
    avg_non_promo_sales = float(non_promo_df['sales'].mean())
    
    # Uplift percentage
    if avg_non_promo_sales > 0:
        uplift_pct = (avg_promo_sales - avg_non_promo_sales) / avg_non_promo_sales
    else:
        uplift_pct = 0.0
        
    # Incremental revenue calculation
    # For each promo week: Incremental Sales Volume = max(0, actual_sales - avg_non_promo_sales)
    # Incremental Revenue = Incremental Sales Volume * actual_price
    incremental_revenue = 0.0
    total_promo_revenue = 0.0
    
    for _, row in promo_df.iterrows():
        promo_sales = row['sales']
        price = row['price']
        inc_sales = max(0.0, promo_sales - avg_non_promo_sales)
        incremental_revenue += inc_sales * price
        total_promo_revenue += promo_sales * price
        
    total_non_promo_revenue = float((non_promo_df['sales'] * non_promo_df['price']).sum())
    
    return {
        "store": store,
        "product": product,
        "promotion_weeks_count": len(promo_df),
        "non_promotion_weeks_count": len(non_promo_df),
        "average_promoted_sales": round(avg_promo_sales, 2),
        "average_non_promoted_sales": round(avg_non_promo_sales, 2),
        "uplift_pct": round(uplift_pct, 4),
        "total_promotional_revenue": round(total_promo_revenue, 2),
        "estimated_incremental_revenue": round(incremental_revenue, 2)
    }
