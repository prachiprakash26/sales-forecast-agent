import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_mock_sales_data(output_path: str = "data/mock_sales.csv") -> pd.DataFrame:
    """
    Generates 3 years of realistic weekly sales data for 5 stores and 10 products
    incorporating trend, seasonality, promotions, holidays, price elasticity,
    competitor pricing, random noise, and inventory constraints.
    
    Saves the generated data to output_path.
    """
    np.random.seed(42)
    
    # 3 years of weekly dates (starting 2023-01-01 to 2025-12-28, approx 156 weeks)
    start_date = datetime(2023, 1, 1)
    weeks = 156
    dates = [start_date + timedelta(weeks=i) for i in range(weeks)]
    
    stores = [f"Store_{i}" for i in range(1, 6)]
    products = [f"Product_{i}" for i in range(1, 11)]
    
    # Base configuration for products
    # Product base prices between $10 and $100
    base_prices = {prod: np.round(10.0 + (i * 10.0) + np.random.uniform(-2, 2), 2) for i, prod in enumerate(products)}
    # Elasticity coefficients (negative relationship between price and demand)
    elasticities = {prod: np.round(np.random.uniform(-2.5, -1.2), 2) for prod in products}
    # Base weekly demand volume
    base_demands = {prod: np.random.randint(100, 500) for prod in products}
    
    # Store-specific scaling factor (different store sizes)
    store_scales = {store: np.round(np.random.uniform(0.6, 1.4), 2) for store in stores}
    # Store-specific trend coefficient (growth rate per year)
    store_trends = {store: np.round(np.random.uniform(0.02, 0.08), 4) for store in stores}
    
    rows = []
    
    for date in dates:
        week_num = date.isocalendar()[1]
        year = date.year
        year_idx = year - 2023  # 0, 1, 2
        
        # Holiday flag (weeks around major holidays: Easter ~w14, Memorial Day ~w21, July 4th ~w27, Thanksgiving ~w47, Christmas/NYE ~w51, w52)
        holiday_flag = 1 if week_num in [14, 21, 27, 47, 51, 52] else 0
        
        for store in stores:
            for product in products:
                # 1. Base demand with store scale
                base_demand = base_demands[product] * store_scales[store]
                
                # 2. Long-term trend (linear growth)
                trend_factor = 1.0 + (store_trends[store] * (year_idx + week_num / 52.0))
                
                # 3. Seasonality (annual sine wave + holiday peak)
                # Highs in summer (weeks 24-32) and late Q4 (weeks 48-52)
                seasonality = 1.0 + 0.12 * np.sin(2 * np.pi * week_num / 52.0)
                if holiday_flag == 1:
                    holiday_uplift = 0.25  # 25% uplift
                else:
                    holiday_uplift = 0.0
                
                # 4. Promotion flag (approx 15% probability, product-store specific)
                promotion_flag = 1 if np.random.uniform(0, 1) < 0.15 else 0
                promo_uplift = 0.30 if promotion_flag == 1 else 0.0
                
                # 5. Price variation & competitor price
                # Price drops by 10% to 20% on promotion, fluctuates slightly otherwise
                base_p = base_prices[product]
                if promotion_flag == 1:
                    price = np.round(base_p * np.random.uniform(0.80, 0.90), 2)
                else:
                    price = np.round(base_p * np.random.uniform(0.95, 1.05), 2)
                
                # Competitor price generally mirrors ours but with noise
                competitor_price = np.round(price * np.random.uniform(0.90, 1.10), 2)
                
                # 6. Price elasticity effect: demand = base_demand * (price/base_price) ^ elasticity
                price_ratio = price / base_p
                elasticity_factor = price_ratio ** elasticities[product]
                
                # Calculate deterministic demand
                demanded_sales = base_demand * trend_factor * seasonality * (1.0 + holiday_uplift) * (1.0 + promo_uplift) * elasticity_factor
                
                # Add random noise (normal distribution)
                noise_std = demanded_sales * 0.08  # 8% of current demand
                demanded_sales += np.random.normal(0, noise_std)
                demanded_sales = max(0.0, demanded_sales)
                
                # 7. Inventory constraints (mock a weekly supply chain replenishment)
                # Inventory is usually replenished but can run out during high demand weeks
                expected_weekly_sales = base_demand * trend_factor
                base_inventory = expected_weekly_sales * np.random.uniform(1.2, 2.2)
                
                # If there was a promo, sometimes inventory isn't enough (out-of-stocks)
                if promotion_flag == 1 and np.random.uniform(0, 1) < 0.25:
                    inventory = np.round(demanded_sales * np.random.uniform(0.6, 0.95))
                else:
                    inventory = np.round(base_inventory * np.random.uniform(0.8, 1.5))
                
                inventory = int(max(0, inventory))
                
                # Actual sales is bounded by inventory
                sales = np.round(min(demanded_sales, inventory), 2)
                
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store": store,
                    "product": product,
                    "sales": sales,
                    "price": price,
                    "promotion_flag": promotion_flag,
                    "holiday_flag": holiday_flag,
                    "inventory": inventory,
                    "competitor_price": competitor_price
                })
                
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Mock sales data successfully saved to {output_path}. Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    generate_mock_sales_data()
