import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.utils.helpers import get_data_path

def detect_anomalies(
    product: str = None,
    store: str = None,
    contamination: float = 0.02
) -> dict:
    """
    Detects sales anomalies in historical data using the Isolation Forest algorithm.
    Can be filtered by product and/or store.
    
    Returns a dictionary containing a list of detected anomalies.
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
        
    # Group by date, store, product to check individual row anomalies
    # (Or if we aggregate, we run on aggregated sales. Running on individual rows is better 
    # to pinpoint specific anomalies)
    X = filtered_df[['sales']].values
    
    # Initialize and fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    
    # Fit and predict
    # 1 for normal, -1 for anomaly
    preds = iso_forest.fit_predict(X)
    scores = iso_forest.decision_function(X) # lower = more anomalous
    
    filtered_df['is_anomaly'] = preds
    filtered_df['anomaly_score'] = scores
    
    # Filter anomalies
    anomalies_df = filtered_df[filtered_df['is_anomaly'] == -1]
    
    # Sort anomalies so the most severe (lowest score) are first
    anomalies_df = anomalies_df.sort_values('anomaly_score')
    
    anomalies_list = []
    for _, row in anomalies_df.iterrows():
        anomalies_list.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "store": row["store"],
            "product": row["product"],
            "sales": float(row["sales"]),
            "price": float(row["price"]),
            "promotion_flag": int(row["promotion_flag"]),
            "holiday_flag": int(row["holiday_flag"]),
            "inventory": int(row["inventory"]),
            "anomaly_score": round(float(row["anomaly_score"]), 4)
        })
        
    return {
        "store": store,
        "product": product,
        "total_records_analyzed": len(filtered_df),
        "anomalies_count": len(anomalies_list),
        "anomalies": anomalies_list[:20]  # Return top 20 most severe anomalies to keep token count reasonable
    }
