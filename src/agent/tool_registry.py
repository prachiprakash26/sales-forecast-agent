import json
import logging
from typing import Dict, Any, List

# Configure logger
logger = logging.getLogger(__name__)

# Import tools
from src.tools.summary_tool import summarize_sales
from src.tools.forecast_tool import forecast_sales
from src.tools.promotion_tool import analyze_promotion_impact
from src.tools.anomaly_tool import detect_anomalies
from src.tools.inventory_tool import inventory_recommendation
from src.tools.driver_analysis_tool import explain_forecast_drivers
from src.tools.elasticity_tool import estimate_price_elasticity

# Define OpenAI function schemas
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "summarize_sales",
            "description": "Aggregates and summarizes historical sales data (total revenue, weekly average, growth trends, and rankings of top products and stores).",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Specific product filter (e.g., 'Product_1' to 'Product_10'). If omitted, aggregates all products.",
                    },
                    "store": {
                        "type": "string",
                        "description": "Specific store filter (e.g., 'Store_1' to 'Store_5'). If omitted, aggregates all stores.",
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_sales",
            "description": "Generates a weekly sales forecast for a given store/product over a horizon (12, 26, or 52 weeks) using Prophet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Specific product to forecast. If omitted, forecasts total sales.",
                    },
                    "store": {
                        "type": "string",
                        "description": "Specific store to forecast. If omitted, forecasts total sales.",
                    },
                    "horizon_weeks": {
                        "type": "integer",
                        "description": "Forecasting horizon in weeks. Supported values: 12, 26, 52.",
                        "default": 12
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_promotion_impact",
            "description": "Analyzes the effectiveness of promotions by comparing sales during promotional vs non-promotional periods and calculating incremental revenue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Specific product to analyze. If omitted, aggregates all products.",
                    },
                    "store": {
                        "type": "string",
                        "description": "Specific store to analyze. If omitted, aggregates all stores.",
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Scans historical sales data using an Isolation Forest model to detect anomalous weekly spikes or drops.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Filter by specific product.",
                    },
                    "store": {
                        "type": "string",
                        "description": "Filter by specific store.",
                    },
                    "contamination": {
                        "type": "number",
                        "description": "Expected ratio of anomalies in the data (e.g. 0.02 for 2%).",
                        "default": 0.02
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inventory_recommendation",
            "description": "Compares current stock against future demand forecasts to recommend inventory additions, reductions, or maintenance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Filter recommendations to a specific product.",
                    },
                    "store": {
                        "type": "string",
                        "description": "Filter recommendations to a specific store.",
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_forecast_drivers",
            "description": "Runs OLS regression to decompose sales into the primary drivers: trend, promotions, holidays, unit pricing, and annual seasonality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Filter driver analysis to a specific product.",
                    },
                    "store": {
                        "type": "string",
                        "description": "Filter driver analysis to a specific store.",
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_price_elasticity",
            "description": "Performs log-log regression (log(sales) ~ log(price)) to calculate price elasticity of demand and suggest pricing strategies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Filter elasticity estimation to a specific product.",
                    },
                    "store": {
                        "type": "string",
                        "description": "Filter elasticity estimation to a specific store.",
                    }
                },
                "required": []
            }
        }
    }
]

# Tool lookup map
TOOL_FUNCTIONS: Dict[str, Any] = {
    "summarize_sales": summarize_sales,
    "forecast_sales": forecast_sales,
    "analyze_promotion_impact": analyze_promotion_impact,
    "detect_anomalies": detect_anomalies,
    "inventory_recommendation": inventory_recommendation,
    "explain_forecast_drivers": explain_forecast_drivers,
    "estimate_price_elasticity": estimate_price_elasticity
}

def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    Looks up a tool by name and executes it with the parsed arguments.
    Returns the result of the function execution as a JSON string.
    """
    logger.info(f"Executing tool '{name}' with arguments: {arguments}")
    
    if name not in TOOL_FUNCTIONS:
        error_msg = {"error": f"Tool '{name}' is not registered in the system."}
        return json.dumps(error_msg)
        
    try:
        # Resolve python function
        func = TOOL_FUNCTIONS[name]
        
        # Call function
        result = func(**arguments)
        
        # Serialize back to JSON string
        return json.dumps(result, default=str)
        
    except Exception as e:
        logger.error(f"Error executing tool '{name}': {str(e)}", exc_info=True)
        return json.dumps({"error": f"Execution of tool '{name}' failed: {str(e)}"})
