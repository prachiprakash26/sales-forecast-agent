import os
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.data_generator import generate_mock_sales_data
from src.forecasting.prophet_model import forecast_sales
from src.tools.summary_tool import summarize_sales
from src.tools.promotion_tool import analyze_promotion_impact
from src.tools.anomaly_tool import detect_anomalies
from src.tools.inventory_tool import inventory_recommendation
from src.tools.driver_analysis_tool import explain_forecast_drivers
from src.tools.elasticity_tool import estimate_price_elasticity
from src.agent.agent import SalesForecastAgent

def run_verification():
    print("=== STARTING AGENT VERIFICATION ===")
    
    # 1. Verify Data Ingest
    print("\n[1/4] Checking Data Availability...")
    data_path = "data/mock_sales.csv"
    if not os.path.exists(data_path):
        print("Data not found. Generating mock dataset...")
        generate_mock_sales_data(data_path)
    else:
        print(f"Data file active at {data_path}")
        
    # 2. Verify Tool Executions directly
    print("\n[2/4] Testing Tool Execution Interfaces...")
    
    print("  |- Testing summary_tool...")
    summary = summarize_sales(product="Product_1", store="Store_1")
    assert "total_sales" in summary, "Summary tool failed!"
    print(f"     Success. Total Product_1 Sales at Store_1: {summary['total_sales']:,}")
    
    print("  |- Testing forecasting engine (fallback mode support)...")
    fc = forecast_sales(product="Product_3", store="Store_1", horizon_weeks=12)
    assert len(fc["forecast_df"]) > 0, "Forecast engine failed!"
    print(f"     Success. Forecast generated. Summary: {fc['forecast_summary']}")
    
    print("  |- Testing promotion_tool...")
    promo = analyze_promotion_impact(product="Product_3", store="Store_1")
    assert "uplift_pct" in promo, "Promotion tool failed!"
    print(f"     Success. Promo uplift: {promo['uplift_pct']:.2%}")
    
    print("  |- Testing anomaly_tool...")
    anom = detect_anomalies(product="Product_2", store="Store_1", contamination=0.02)
    assert "anomalies_count" in anom, "Anomaly tool failed!"
    print(f"     Success. Anomalies detected: {anom['anomalies_count']}")
    
    print("  |- Testing inventory_tool...")
    inv = inventory_recommendation(product="Product_4", store="Store_1")
    assert "increase_stock" in inv, "Inventory tool failed!"
    print(f"     Success. Increase stock list count: {len(inv['increase_stock'])}")
    
    print("  |- Testing driver_analysis_tool...")
    drivers = explain_forecast_drivers(product="Product_1", store="Store_1")
    assert "driver_percentages" in drivers, "Driver tool failed!"
    print(f"     Success. Drivers pricing R2: {drivers['model_fit_r2']}")
    
    print("  |- Testing elasticity_tool...")
    elasticity = estimate_price_elasticity(product="Product_5", store="Store_1")
    assert "elasticity" in elasticity, "Elasticity tool failed!"
    print(f"     Success. Price elasticity: {elasticity['elasticity']} ({elasticity['category']})")
    
    # 3. Verify Agent Offline Fallback Routing
    print("\n[3/4] Testing Agent Fallback Loop...")
    agent = SalesForecastAgent(base_url="http://localhost:9999/v1") # Port 9999 triggers immediate offline fallback
    response = agent.ask("Forecast sales of Product_3 for Store_1 next 12 weeks")
    assert "Demand Forecast Summary" in response, "Agent forecasting request failed!"
    
    response_promo = agent.ask("What is promotion uplift for Product_3 in Store_1?")
    assert "Promotional Uplift Analysis" in response_promo, "Agent promotion query failed!"
    
    print("\n[4/4] Verification Succeeded!")
    print("All python modules parsed, imported, and executed correctly.")

if __name__ == "__main__":
    run_verification()
