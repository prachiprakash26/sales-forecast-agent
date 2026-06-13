# AI-Powered Sales Forecasting Agent

This repository implements a production-quality AI-Powered Sales Forecasting Agent that runs inside Jupyter notebooks. The agent interfaces with a local `vLLM` server hosting `Qwen/Qwen3-4B` using the `Pydantic AI` framework and calls local Python analytics tools (Prophet, Isolation Forest, log-log OLS regressions) to answer business queries.

## Key Features
- **Pydantic AI Agent**: Uses Pydantic AI for clean model configuration, automatic tool schemas generation from signatures/docstrings, and robust message history management.
- **Prophet Forecasting Engine**: Automated time-series forecasting with trend, seasonality, and confidence boundaries.
- **Price Elasticity Estimation**: Log-log linear regression model (`log(sales) ~ log(price)`) to calculate demand sensitivity.
- **Forecast Driver Analysis**: Linear regression OLS decomposition to attribute sales to trend, pricing, promotions, and holidays.
- **Anomaly Detection**: Scikit-Learn `IsolationForest` scans for statistical outliers in weekly sales volume.
- **Inventory Recommendation Engine**: Stock adjustment suggestions based on forecasted Weeks of Supply (WOS).
- **Offline Heuristic Mode**: If the local LLM server is offline, the agent runs a rule-based parser that executes tools directly in the notebook, preventing code crashes.

## Directory Structure
```text
sales-forecast-agent/
├── data/
│   └── mock_sales.csv             # Generated weekly sales database (3 years)
├── notebooks/
│   └── Sales_Forecasting_Agent.ipynb  # Interactive workspace and agent console
├── src/
│   ├── agent/
│   │   ├── agent.py               # Conversational Pydantic AI agent loop and fallbacks
│   │   ├── prompts.py             # Agent system prompt
│   │   └── tool_registry.py       # Local routing mapping helper
│   ├── forecasting/
│   │   └── prophet_model.py       # Meta Prophet forecasting model (with Holt-Winters fallback)
│   ├── tools/
│   │   ├── anomaly_tool.py        # Outlier detection (Isolation Forest)
│   │   ├── driver_analysis_tool.py# Feature regression decomposition
│   │   ├── elasticity_tool.py     # Log-log elasticity analysis
│   │   ├── forecast_tool.py       # Forecast agent tool wrapper
│   │   ├── inventory_tool.py      # Inventory safety recommendation
│   │   ├── promotion_tool.py      # Promo uplift and incremental revenue analysis
│   │   └── summary_tool.py        # Overall sales metrics aggregation
│   ├── data/
│   │   └── data_generator.py      # 3-year store-product dataset simulator
│   └── utils/
│       └── helpers.py             # File path resolving & Plotly graphing utilities
├── requirements.txt
├── README.md
└── setup.sh
```

## Setup Instructions

### 1. Install Dependencies
Run the setup script or install requirements manually:
```bash
chmod +x setup.sh
./setup.sh
```
*On Windows, you can execute the command manually:*
```powershell
pip install -r requirements.txt
python -m src.data.data_generator
```

### 2. Run the local vLLM Server
Launch the local vLLM server pointing to the Qwen3-4B model with tools enabled:
```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --max_model_len 24272
```
*Note:* The agent connects to `http://localhost:8000/v1` with the authorization header bearer key `abc-123`.

### 3. Start the Jupyter Notebook
Run the notebook interface:
```bash
jupyter notebook notebooks/Sales_Forecasting_Agent.ipynb
```

## Agent Query API
Execute the notebook cells to ask questions:
```python
# Forecast queries
agent.ask("Forecast Product_3 sales for next 12 weeks")

# Promotional impact
agent.ask("What was the impact of promotions for Product_3?")

# Inventory recommendation
agent.ask("Which products need inventory replenishment?")

# Price elasticity
agent.ask("Estimate price elasticity for Product_5")

# Driver attribution
agent.ask("Explain the drivers of sales for Product_1 at Store_2")
```
If the model is offline, the agent prints a system notification and runs the local Python tools directly, summarizing their results in clean Markdown.
