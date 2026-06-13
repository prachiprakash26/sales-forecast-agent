import os
import pandas as pd
import numpy as np

# Suppress warnings where necessary
import warnings
warnings.filterwarnings('ignore')

def get_data_path(filename: str = "mock_sales.csv") -> str:
    """
    Returns the absolute path to files in the data directory, handling notebook paths context.
    """
    # Try looking for data/ directory relative to workspace root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(base_dir, "data", filename)
    if os.path.exists(path):
        return path
    
    # Fallback to local data/ folder
    return os.path.join("data", filename)

def plot_forecast_plotly(forecast_df: pd.DataFrame, title: str = "Sales Forecast") -> object:
    """
    Generates a premium interactive Plotly chart showing:
      1. Historical actual sales (solid dark blue line)
      2. Forecasted sales (dashed purple/violet line)
      3. 95% Confidence interval (light purple shaded region)
      
    Returns a Plotly Figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        # Fallback if plotly is not installed
        print("Plotly is not installed. Returning static matplotlib chart.")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        hist = forecast_df[forecast_df['actual_sales'].notna()]
        fut = forecast_df[forecast_df['actual_sales'].isna()]
        
        plt.plot(hist['date'], hist['actual_sales'], label='Actual Sales', color='#1f77b4', linewidth=2)
        plt.plot(forecast_df['date'], forecast_df['forecasted_sales'], label='Forecast', color='#9467bd', linestyle='--')
        
        if 'lower_bound' in forecast_df.columns:
            plt.fill_between(forecast_df['date'], forecast_df['lower_bound'], forecast_df['upper_bound'], 
                             color='#9467bd', alpha=0.15, label='95% Confidence Interval')
            
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Sales Units")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return None

    # Construct the Plotly figure
    fig = go.Figure()
    
    hist_df = forecast_df[forecast_df['actual_sales'].notna()]
    # To make the forecast line continuous, we grab the last historical point
    fut_df = forecast_df[forecast_df['actual_sales'].isna()]
    if not hist_df.empty and not fut_df.empty:
        last_hist_row = hist_df.tail(1)
        fut_df = pd.concat([last_hist_row, fut_df]).sort_values('date')
    
    # Add historical actual sales
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['actual_sales'],
        name='Actual Sales',
        line=dict(color='#0F172A', width=2.5),
        mode='lines+markers',
        marker=dict(size=4)
    ))
    
    # Add historical fitted sales (optional, but keep it clean by omitting if requested, 
    # let's just plot historical forecast or fit as a subtle line if we want, or just omit it)
    
    # Add confidence interval (shaded region)
    if 'lower_bound' in forecast_df.columns and not fut_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fut_df['date'], fut_df['date'].iloc[::-1]]),
            y=pd.concat([fut_df['upper_bound'], fut_df['lower_bound'].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(139, 92, 246, 0.15)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Interval'
        ))
        
    # Add forecast line
    if not fut_df.empty:
        fig.add_trace(go.Scatter(
            x=fut_df['date'],
            y=fut_df['forecasted_sales'],
            name='Forecasted Sales',
            line=dict(color='#8B5CF6', width=2.5, dash='dash'),
            mode='lines+markers',
            marker=dict(size=4)
        ))
        
    # Premium layout styling
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Outfit, Inter, sans-serif", color='#0F172A'),
            x=0.05
        ),
        xaxis=dict(
            title="Date",
            gridcolor='rgba(226, 232, 240, 0.8)',
            showline=True,
            linecolor='#94A3B8'
        ),
        yaxis=dict(
            title="Sales Units",
            gridcolor='rgba(226, 232, 240, 0.8)',
            showline=True,
            linecolor='#94A3B8'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified"
    )
    
    return fig
