import os
import pandas as pd
import numpy as np
from src.utils.helpers import get_data_path
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def inventory_recommendation(product: str = None, store: str = None) -> dict:
    """
    Compares current inventory levels against the forecasted weekly demand 
    to provide stock recommendations (increase, maintain, or reduce stock).
    
    Uses recent actuals and Exponential Smoothing to quickly project demand.
    """
    data_path = get_data_path("mock_sales.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply filtering if specified
    filtered_df = df.copy()
    if store is not None:
        filtered_df = filtered_df[filtered_df['store'] == store]
    if product is not None:
        filtered_df = filtered_df[filtered_df['product'] == product]
        
    if filtered_df.empty:
        return {
            "error": f"No data found matching store={store} and product={product}."
        }
        
    # Get all combinations of store & product present in the filtered data
    combos = filtered_df.groupby(['store', 'product'])
    
    increase_stock = []
    maintain_stock = []
    reduce_stock = []
    
    for (s_id, p_id), combo_df in combos:
        combo_df = combo_df.sort_values('date')
        
        # Current inventory is the latest recorded value
        current_inv = int(combo_df['inventory'].iloc[-1])
        
        # Forecast weekly demand for the next 12 weeks
        # We use a fast Simple Exponential Smoothing model for performance across multiple items
        sales_series = combo_df['sales'].values
        try:
            model = SimpleExpSmoothing(sales_series).fit(smoothing_level=0.3, optimized=False)
            # Forecast next 12 weeks and get average weekly demand
            forecast_demand = model.forecast(12)
            avg_weekly_demand = float(np.mean(forecast_demand))
        except Exception:
            # Simple moving average fallback if fitting fails
            avg_weekly_demand = float(np.mean(sales_series[-4:]))
            
        avg_weekly_demand = max(0.1, avg_weekly_demand)
        
        # Weeks of Supply (WOS) = Current Inventory / Average Weekly Forecasted Demand
        weeks_of_supply = current_inv / avg_weekly_demand
        
        item_summary = {
            "store": s_id,
            "product": p_id,
            "current_inventory": current_inv,
            "estimated_weekly_demand": round(avg_weekly_demand, 2),
            "weeks_of_supply": round(weeks_of_supply, 2)
        }
        
        # Recommendation logic:
        # - WOS < 2.0 weeks: High stockout risk -> Increase Stock
        # - WOS > 5.0 weeks: Excess capital tied up -> Reduce Stock
        # - Otherwise: Healthy level -> Maintain Stock
        if weeks_of_supply < 2.0:
            increase_stock.append(item_summary)
        elif weeks_of_supply > 5.0:
            reduce_stock.append(item_summary)
        else:
            maintain_stock.append(item_summary)
            
    # Sort recommendations by weeks of supply
    increase_stock = sorted(increase_stock, key=lambda x: x["weeks_of_supply"])
    reduce_stock = sorted(reduce_stock, key=lambda x: x["weeks_of_supply"], reverse=True)
    
    return {
        "store_filter": store,
        "product_filter": product,
        "increase_stock": increase_stock[:15], # cap length for prompt injection limit
        "maintain_stock": maintain_stock[:15],
        "reduce_stock": reduce_stock[:15]
    }
