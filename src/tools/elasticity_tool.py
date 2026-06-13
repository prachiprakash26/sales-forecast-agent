import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils.helpers import get_data_path

def estimate_price_elasticity(product: str = None, store: str = None) -> dict:
    """
    Estimates the price elasticity of demand using a log-log regression model:
      log(sales) ~ log(price)
    
    Filters data by product and/or store.
    Returns the elasticity coefficient and its business interpretation.
    """
    data_path = get_data_path("mock_sales.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter dataset
    filtered_df = df.copy()
    if store is not None and store != "":
        filtered_df = filtered_df[filtered_df['store'] == store]
    if product is not None and product != "":
        filtered_df = filtered_df[filtered_df['product'] == product]
        
    if filtered_df.empty:
        return {
            "error": f"No data found matching store={store} and product={product}."
        }
        
    # Group by date to aggregate if filtering isn't specific
    # In log-log regression we need individual price-demand points. 
    # Grouping by store and product is best, but if we aggregate, we sum sales and average prices.
    agg_df = filtered_df.groupby(['date', 'store', 'product']).agg({
        'sales': 'sum',
        'price': 'mean'
    }).reset_index()
    
    # Ensure sales and prices are positive and non-zero to avoid infinite values in log transform
    clean_df = agg_df[(agg_df['sales'] > 0) & (agg_df['price'] > 0)].copy()
    
    if len(clean_df) < 5:
        return {
            "error": "Not enough positive sales records to compute price elasticity."
        }
        
    # Log transformation
    clean_df['log_sales'] = np.log(clean_df['sales'])
    clean_df['log_price'] = np.log(clean_df['price'])
    
    X = clean_df[['log_price']]
    y = clean_df['log_sales']
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    elasticity = float(model.coef_[0])
    r2 = float(model.score(X, y))
    
    # Interpretation logic
    if elasticity < -1.0:
        category = "Elastic"
        interpretation = (
            f"Demand is highly ELASTIC (elasticity = {elasticity:.2f}). "
            f"A 1% increase in price leads to an estimated {abs(elasticity):.2f}% decrease in sales volume. "
            f"Conversely, lowering prices during promotions is likely to drive significant volume growth and increase total revenue."
        )
    elif -1.0 <= elasticity < 0:
        category = "Inelastic"
        interpretation = (
            f"Demand is INELASTIC (elasticity = {elasticity:.2f}). "
            f"A 1% increase in price leads to an estimated {abs(elasticity):.2f}% decrease in sales volume. "
            f"Since consumers are relatively price-insensitive, raising prices could increase total revenue, "
            f"whereas aggressive discounting might decrease total profitability."
        )
    elif elasticity == 0:
        category = "Perfectly Inelastic"
        interpretation = "Demand is completely unaffected by price changes (elasticity = 0.0)."
    else:
        category = "Positive Elasticity (Giffen/Veblen behavior)"
        interpretation = (
            f"Demand shows a POSITIVE relationship with price (elasticity = {elasticity:.2f}). "
            f"This suggests that higher prices correlate with higher sales, which in practice is often due to "
            f"confounding variables such as promotions run during high-seasonality periods, holiday effects, "
            f"or stockout-driven price rises during peak demand."
        )
        
    return {
        "store": store,
        "product": product,
        "elasticity": round(elasticity, 4),
        "category": category,
        "model_fit_r2": round(r2, 4),
        "interpretation": interpretation,
        "data_points": len(clean_df)
    }
