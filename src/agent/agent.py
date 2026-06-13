import os
import re
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tool_registry import TOOL_SCHEMAS, execute_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesForecastAgent:
    """
    An AI Sales Forecasting Agent that interacts with Qwen3 via a vLLM API endpoint.
    Maintains memory of conversations and executes multi-step tool calls natively.
    Includes a rule-based fallback if the vLLM server is offline.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "none",
        model: str = "Qwen/Qwen3-30B-Instruct"
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.base_url = base_url
        logger.info(f"Initialized SalesForecastAgent pointing to {base_url} with model {model}.")
        
    def ask(self, question: str) -> str:
        """
        Sends the user question to Qwen3, handles all requested tool calls, 
        and prints the final business response.
        """
        print(f"\033[1;34m[User]:\033[0m {question}\n")
        self.messages.append({"role": "user", "content": question})
        
        # Check if vLLM server is responsive
        server_online = True
        try:
            # Short timeout check
            import httpx
            httpx.get(f"{self.base_url.replace('/v1', '')}/health", timeout=1.0)
        except Exception:
            server_online = False
            
        if not server_online:
            print("\033[1;33m[System Notification]: Local vLLM server is offline. Running in Local Offline Heuristic Mode...\033[0m")
            return self._fallback_execution(question)
            
        try:
            return self._run_agentic_loop()
        except Exception as e:
            logger.error("Error in agent loop, running offline fallback", exc_info=True)
            print(f"\033[1;31m[Agent Error]:\033[0m Model call failed ({str(e)}). Running offline fallback...")
            return self._fallback_execution(question)

    def _run_agentic_loop(self) -> str:
        """
        Runs the standard OpenAI client tool-calling loop.
        """
        max_turns = 5
        for turn in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2
            )
            
            message = response.choices[0].message
            
            # If the model wants to call tools
            if message.tool_calls:
                # Add model message to history (required for OpenAI tool loop)
                self.messages.append(message)
                
                print(f"\033[1;32m[Agent thinking...]:\033[0m Requesting {len(message.tool_calls)} tool execution(s).")
                
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  |- \033[36mCalling tool:\033[0m {tool_name}({', '.join(f'{k}={v}' for k, v in tool_args.items())})")
                    
                    # Execute tool
                    tool_result_str = execute_tool(tool_name, tool_args)
                    
                    # Append result to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result_str
                    })
                
                # Continue loop to send tool responses back to model
                continue
                
            # If the model has completed and returned a text response
            final_content = message.content
            self.messages.append({"role": "assistant", "content": final_content})
            
            print(f"\033[1;35m[Agent Response]:\033[0m\n{final_content}\n")
            return final_content
            
        print("\033[1;31m[Agent Alert]:\033[0m Reached maximum execution turns without response.")
        return "Reached maximum execution turns."

    def _fallback_execution(self, question: str) -> str:
        """
        A rule-based classifier fallback when the vLLM server is unavailable.
        Uses regex to determine what the user is asking and calls the python tool directly.
        """
        # Parse store and product names from question
        store_match = re.search(r'\bStore_[1-5]\b', question, re.IGNORECASE)
        product_match = re.search(r'\bProduct_(10|[1-9])\b', question, re.IGNORECASE)
        
        store = store_match.group(0) if store_match else None
        product = product_match.group(0) if product_match else None
        
        # Classify the tool
        q_lower = question.lower()
        
        # Initialize placeholders
        tool_name = ""
        args = {"store": store, "product": product}
        
        if "elasticity" in q_lower or "price response" in q_lower:
            tool_name = "estimate_price_elasticity"
        elif "forecast" in q_lower or "predict" in q_lower or "projection" in q_lower:
            tool_name = "forecast_sales"
            # Extract horizon
            horizon_match = re.search(r'\b(12|26|52)\b', question)
            args["horizon_weeks"] = int(horizon_match.group(0)) if horizon_match else 12
        elif "promo" in q_lower or "discount" in q_lower or "uplift" in q_lower:
            tool_name = "analyze_promotion_impact"
        elif "anomaly" in q_lower or "spikes" in q_lower or "drops" in q_lower or "outlier" in q_lower:
            tool_name = "detect_anomalies"
        elif "inventory" in q_lower or "replenish" in q_lower or "stock" in q_lower or "shortage" in q_lower:
            tool_name = "inventory_recommendation"
        elif "driver" in q_lower or "why" in q_lower or "cause" in q_lower or "impact of" in q_lower:
            tool_name = "explain_forecast_drivers"
        else:
            tool_name = "summarize_sales"
            
        print(f"  |- \033[36mClassified Offline Call:\033[0m {tool_name}({', '.join(f'{k}={v}' for k, v in args.items())})")
        
        # Execute tool
        result_str = execute_tool(tool_name, args)
        result_dict = json.loads(result_str)
        
        if "error" in result_dict:
            response = f"Could not perform analysis. Tool error: {result_dict['error']}"
            print(f"\033[1;35m[Agent Response]:\033[0m\n{response}\n")
            return response
            
        # Format a business response manually based on the tool result
        response = self._format_manual_response(tool_name, result_dict, store, product)
        print(f"\033[1;35m[Agent Response]:\033[0m\n{response}\n")
        
        # Save mock history
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _format_manual_response(self, tool_name: str, data: Dict[str, Any], store: str, product: str) -> str:
        """
        Formats a natural language business response for the offline heuristic fallback.
        """
        store_lbl = store if store else "all stores"
        prod_lbl = product if product else "all products"
        
        if tool_name == "estimate_price_elasticity":
            return (
                f"### Price Elasticity Report for {prod_lbl} ({store_lbl})\n\n"
                f"- **Elasticity Coefficient**: {data['elasticity']}\n"
                f"- **Category**: {data['category']}\n"
                f"- **R-Squared Fit**: {data['model_fit_r2']}\n\n"
                f"**Business Analysis**:\n{data['interpretation']}\n\n"
                f"*Note: Analyzed {data['data_points']} positive weekly sales points.*"
            )
            
        elif tool_name == "forecast_sales":
            return (
                f"### Demand Forecast Summary for {prod_lbl} ({store_lbl})\n\n"
                f"**Executive Narrative**:\n{data['forecast_summary']}\n\n"
                f"**Upcoming 4 Weeks Forecast Detail**:\n"
                f"| Date | Forecasted Sales | Lower Bound | Upper Bound |\n"
                f"| :--- | :--- | :--- | :--- |\n"
                f"| {data['forecast_predictions'][0]['date']} | {data['forecast_predictions'][0]['forecasted_sales']:,} | {data['forecast_predictions'][0]['lower_bound']:,} | {data['forecast_predictions'][0]['upper_bound']:,} |\n"
                f"| {data['forecast_predictions'][1]['date']} | {data['forecast_predictions'][1]['forecasted_sales']:,} | {data['forecast_predictions'][1]['lower_bound']:,} | {data['forecast_predictions'][1]['upper_bound']:,} |\n"
                f"| {data['forecast_predictions'][2]['date']} | {data['forecast_predictions'][2]['forecasted_sales']:,} | {data['forecast_predictions'][2]['lower_bound']:,} | {data['forecast_predictions'][2]['upper_bound']:,} |\n"
                f"| {data['forecast_predictions'][3]['date']} | {data['forecast_predictions'][3]['forecasted_sales']:,} | {data['forecast_predictions'][3]['lower_bound']:,} | {data['forecast_predictions'][3]['upper_bound']:,} |\n\n"
                f"*Use Plotly figure rendering in the notebook for complete visual profile.*"
            )
            
        elif tool_name == "analyze_promotion_impact":
            return (
                f"### Promotional Uplift Analysis for {prod_lbl} ({store_lbl})\n\n"
                f"- **Promo Weeks Analyzed**: {data['promotion_weeks_count']}\n"
                f"- **Avg Weekly Sales (On Promo)**: {data['average_promoted_sales']:,} units\n"
                f"- **Avg Weekly Sales (No Promo)**: {data['average_non_promoted_sales']:,} units\n"
                f"- **Uplift Factor**: **+{data['uplift_pct']:.1%}**\n\n"
                f"**Financial Performance**:\n"
                f"- Total revenue generated during promotions: **${data['total_promotional_revenue']:,.2f}**\n"
                f"- Estimated **Incremental Revenue** from promos: **${data['estimated_incremental_revenue']:,.2f}**\n\n"
                f"**Strategic Takeaway**:\n"
                f"The promotions run for {prod_lbl} in {store_lbl} have shown a powerful response with an uplift of {data['uplift_pct']:.1%}. "
                f"We recommend continuing tactical campaigns, especially in high-seasonality periods."
            )
            
        elif tool_name == "detect_anomalies":
            anom_text = ""
            for item in data['anomalies'][:3]:
                anom_text += f"- **{item['date']}**: Sales of **{item['sales']}** units (Price: ${item['price']}, Score: {item['anomaly_score']})\n"
            if not anom_text:
                anom_text = "No extreme anomalies detected in this segment."
                
            return (
                f"### Anomalous Activity Report ({prod_lbl} - {store_lbl})\n\n"
                f"Total weeks analyzed: {data['total_records_analyzed']}. Anomalies found: {data['anomalies_count']}.\n\n"
                f"**Top Severe Deviations**:\n{anom_text}\n"
                f"**Recommendation**:\n"
                f"Check weather patterns, stock availability records, or billing issues for dates with negative anomalies to verify if supply constraints caused stockouts."
            )
            
        elif tool_name == "inventory_recommendation":
            inc_lines = "\n".join([f"  - Store: {x['store']}, Product: {x['product']} (Stock: {x['current_inventory']} vs Wkly Demand: {x['estimated_weekly_demand']}, WOS: {x['weeks_of_supply']})" for x in data['increase_stock'][:5]])
            dec_lines = "\n".join([f"  - Store: {x['store']}, Product: {x['product']} (Stock: {x['current_inventory']} vs Wkly Demand: {x['estimated_weekly_demand']}, WOS: {x['weeks_of_supply']})" for x in data['reduce_stock'][:5]])
            
            return (
                f"### Inventory Replenishment recommendations ({prod_lbl} - {store_lbl})\n\n"
                f"**Stock Shortage / Replenish List (WOS < 2.0)**:\n"
                f"{inc_lines if inc_lines else '  - None.'}\n\n"
                f"**Overstock / Excess Capital List (WOS > 5.0)**:\n"
                f"{dec_lines if dec_lines else '  - None.'}\n\n"
                f"**Action Plan**:\n"
                f"Immediately re-order products on the shortage list to prevent lost revenue. Run promotional discounting or relocate stock from the overstocked stores to optimize carrying costs."
            )
            
        elif tool_name == "explain_forecast_drivers":
            return (
                f"### Forecast Driver Analysis for {prod_lbl} ({store_lbl})\n\n"
                f"**Model Goodness of Fit (R²)**: {data['model_fit_r2']:.4f}\n\n"
                f"**Relative Drivers Contribution**:\n"
                f"- Pricing: {data['driver_percentages']['pricing']:.1%}\n"
                f"- Promotions: {data['driver_percentages']['promotions']:.1%}\n"
                f"- Seasonality: {data['driver_percentages']['seasonality']:.1%}\n"
                f"- Trend: {data['driver_percentages']['trend']:.1%}\n"
                f"- Holidays: {data['driver_percentages']['holidays']:.1%}\n\n"
                f"**Business Narrative Interpretation**:\n{data['business_explanation']}"
            )
            
        else: # summarize_sales
            return (
                f"### Sales Summary Report ({prod_lbl} - {store_lbl})\n\n"
                f"- **Total sales volume**: {data['total_sales']:,} units\n"
                f"- **Average weekly volume**: {data['average_weekly_sales']:,} units\n"
                f"- **Sales Growth (Second Half vs First Half)**: {data['growth_pct']:+.2%}\n\n"
                f"**Performance Rankings (Top Categories)**:\n"
                f"- Top Products: {', '.join([x['product'] for x in data['top_products'][:3]])}\n"
                f"- Top Stores: {', '.join([x['store'] for x in data['top_stores'][:3]])}"
            )
