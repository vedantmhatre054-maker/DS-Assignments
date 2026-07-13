import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import config

class InteractiveVizEngine:
    
    def __init__(self):
        print("[VISUALIZATION SYSTEM] Rendering engine successfully brought online.")

    @staticmethod
    def render_cohort_retention_heatmap(matrix: pd.DataFrame):
        print("Generating interactive Plotly cohort dashboard window...")
        
        if matrix is None or matrix.empty:
            print("[VIZ ABORTED] Passed cohort matrix holds no visible values.")
            return

        z_data = matrix.values
        x_headers = [f"Month {col}" for col in matrix.columns]
        y_headers = [idx.strftime("%B %Y") for idx in matrix.index]
        
        annotation_matrix = []
        for r_idx in range(len(y_headers)):
            row_text = []
            for c_idx in range(len(x_headers)):
                cell_val = z_data[r_idx][c_idx]
                row_text.append(f"{cell_val:.1f}%" if cell_val > 0 else "0.0%")
            annotation_matrix.append(row_text)

        heatmap_trace = go.Heatmap(
            z=z_data, x=x_headers, y=y_headers,
            colorscale="YlGnBu", text=annotation_matrix, texttemplate="%{text}",
            hoverinfo="x+y+z", showscale=True
        )

        fig = go.Figure(data=[heatmap_trace])
        
        fig.update_layout(
            title=dict(
                text="<b>Cohort Lifecycle User Retention Grid Analysis</b>",
                font=dict(size=18, color="#2C3E50")
            ),
            xaxis=dict(title="Relative Lifespan Index Delta", tickmode="linear"),
            yaxis=dict(title="Acquisition Cohort Group", autorange="reversed"),
            template="plotly_white", width=950, height=500
        )
        
        fig.show()

    @staticmethod
    def render_predictive_sales_dashboard(daily_revenue: pd.DataFrame, trend: list, forecast: list):
        print("Generating interactive Plotly machine learning prediction dashboard...")
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_revenue["Timestamp"], y=daily_revenue["Net_Checkout_Price"],
            mode="lines+markers", name="Observed Daily Sales Revenue",
            line=dict(color="#34495E", width=1.5),
            marker=dict(size=4)
        ))

        fig.add_trace(go.Scatter(
            x=daily_revenue["Timestamp"], y=trend,
            mode="lines", name="Calculated Linear Trendline",
            line=dict(color="#E74C3C", width=2, dash="dash")
        ))

        last_recorded_date = daily_revenue["Timestamp"].max()
        future_timeline_dates = pd.date_range(
            start=last_recorded_date + pd.Timedelta(days=1), 
            periods=len(forecast), 
            freq="D"
        )

        fig.add_trace(go.Scatter(
            x=future_timeline_dates, y=forecast,
            mode="lines+markers", name="7-Day Forward Forecast Projection",
            line=dict(color="#2ECC71", width=3),
            marker=dict(symbol="diamond", size=7, color="#27AE60")
        ))

        fig.update_layout(
            title=dict(
                text="<b>Enterprise Revenue Forecasting Pipeline (Linear Model)</b>",
                font=dict(size=18, color="#2C3E50")
            ),
            xaxis=dict(title="Chronological Tracking Horizon", showgrid=True),
            yaxis=dict(title="Net Financial Contribution Amount (USD)", showgrid=True),
            template="plotly_white", width=1000, height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.show()