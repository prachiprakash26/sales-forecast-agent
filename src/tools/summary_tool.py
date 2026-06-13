import os
import pandas as pd
import numpy as np
from src.utils.helpers import get_data_path

def summarize_sales(product: str = None, store: str = None) -> dict:
    """
    Summarizes historical sales data. Can be filtered by store and/or product.
    Calculates overall metrics including total sales, average sales, sales growth,
    and identifies the top products and stores.
    
    Returns a dictionary of metrics.
    """
    data_path = get_data_path("mock_sales.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply filtering
    filtered_df = df.copy()
    if store is not None:
        filtered_df = filtered_df[filtered_df['store'] == store]
    if product is not None:
        filtered_df = filtered_df[filtered_df['product'] == product]
        
    if filtered_df.empty:
        return {
            "error": f"No historical records found matching store={store} and product={product}."
        }
        
    total_sales = float(filtered_df['sales'].sum())
    average_sales = float(filtered_df['sales'].mean())
    
    # Growth metric (comparing second half of the time-series to the first half)
    unique_dates = sorted(filtered_df['date'].unique())
    midpoint = len(unique_dates) // 2
    first_half_dates = unique_dates[:midpoint]
    second_half_dates = unique_dates[midpoint:]
    
    first_half_sales = filtered_df[filtered_df['date'].isin(first_half_dates)]['sales'].sum()
    second_half_sales = filtered_df[filtered_df['date'].isin(second_half_dates)]['sales'].sum()
    
    if first_half_sales > 0:
        growth_pct = float((second_half_sales - first_half_sales) / first_half_sales)
    else:
        growth_pct = 0.0
        
    # Top Products ranking (sum of sales)
    top_products_df = filtered_df.groupby('product')['sales'].sum().reset_index()
    top_products_df = top_products_df.sort_values('sales', ascending=False)
    top_products = top_products_df.to_dict(orient='records')
    
    # Top Stores ranking (sum of sales)
    top_stores_df = filtered_df.groupby('store')['sales'].sum().reset_index()
    top_stores_df = top_stores_df.sort_values('sales', ascending=False)
    top_stores = top_stores_df.to_dict(orient='records')
    
    return {
        "store_filter": store,
        "product_filter": product,
        "total_sales": round(total_sales, 2),
        "average_weekly_sales": round(average_sales, 2),
        "growth_pct": round(growth_pct, 4),
        "top_products": top_products[:5], # top 5
        "top_stores": top_stores[:5]      # top 5
    }
