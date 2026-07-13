import sys
import pandas as pd
import numpy as np
import config
from data_engine import EcomDataEngine
from analytics import GrowthAnalyticsEngine
from viz_engine import InteractiveVizEngine

def generate_formal_html_report(kpis, ab_test, forecast):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: 'Times New Roman', Times, serif;
                margin: 50px;
                color: #000000;
                line-height: 1.6;
            }}
            h1 {{
                text-align: center;
                text-transform: uppercase;
                font-size: 24px;
                margin-bottom: 5px;
            }}
            h2 {{
                text-align: center;
                font-size: 14px;
                font-weight: normal;
                margin-top: 0;
                margin-bottom: 40px;
                border-bottom: 2px solid #000000;
                padding-bottom: 10px;
            }}
            h3 {{
                font-size: 16px;
                text-transform: uppercase;
                margin-top: 30px;
                border-bottom: 1px dashed #000000;
            }}
            p {{
                text-align: justify;
                text-justify: inter-word;
                font-size: 14px;
                text-indent: 40px;
                margin-bottom: 15px;
            }}
            .no-indent {{
                text-indent: 0px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #000000;
                padding: 8px;
                text-align: center;
                font-size: 13px;
            }}
            th {{
                background-color: #F2F2F2;
            }}
        </style>
        <title>Executive Growth Report</title>
    </head>
    <body>

        <h1>Executive Strategic Growth & Predictive Performance Report</h1>
        <h2>Operational Data Science Framework Pipeline Verification Output</h2>

        <h3>I. Foundational Platform KPI Analysis</h3>
        <p>This section provides a rigorous overview of current direct-to-consumer platform metrics. By analyzing unique user frequencies across set calendar parameters, we successfully isolate core engagement and financial velocity vectors. These figures serve as the corporate baseline for all downstream predictive transformations.</p>
        
        <table>
            <tr>
                <th>Operational Metric Category</th>
                <th>Calculated System Output Value</th>
            </tr>
            <tr>
                <td>Active Monthly Customer Footprint (MAU)</td>
                <td>{kpis['MAU']} Unique Users</td>
            </tr>
            <tr>
                <td>Product Engagement Stickiness Index (DAU/MAU)</td>
                <td>{kpis['Stickiness_Pct']}%</td>
            </tr>
            <tr>
                <td>Gross Net Revenue Run-Rate Output</td>
                <td>${kpis['Total_Revenue']:,}</td>
            </tr>
            <tr>
                <td>Average Revenue Per Unique User (ARPU)</td>
                <td>${kpis['ARPU']}</td>
            </tr>
            <tr>
                <td>Calculated Net Promoter Customer Score Proxy (NPS)</td>
                <td>{kpis['NPS_Proxy']} Point Index</td>
            </tr>
        </table>

        <h3>II. Experimental Verification & A/B Testing Validation</h3>
        <p>To optimize the core digital checkout funnel without introducing operational confounding factors, a scientific two-proportion hypothesis test was initialized. This framework systematically evaluates variant execution data against historical benchmarks to verify if observed conversion shifts represent genuine long-term trends or temporary statistical anomalies.</p>
        
        <table>
            <tr>
                <th>Experimental Metric Evaluated</th>
                <th>Calculated Performance Matrix Output</th>
            </tr>
            <tr>
                <td>Baseline Control Conversion Rate (CR)</td>
                <td>{ab_test['Control_CR']}%</td>
            </tr>
            <tr>
                <td>Treatment Experimental Variant Rate (CR)</td>
                <td>{ab_test['Treatment_CR']}%</td>
            </tr>
            <tr>
                <td>Calculated Test Z-Score Value</td>
                <td>{ab_test['Z_Score']}</td>
            </tr>
            <tr>
                <td>Calculated Two-Sided P-Value Outcome</td>
                <td>{ab_test['P_Value']}</td>
            </tr>
            <tr>
                <td>Statistical Significance Verification Status</td>
                <td><b>{"SUCCESS: DEPLOY VARIANT" if ab_test['Is_Significant'] else "FAILED: RETAIN BASELINE"}</b></td>
            </tr>
        </table>

        <h3>III. Predictive Horizon Sales Forecasting</h3>
        <p>Utilizing a customized Scikit-Learn linear regression paradigm, historical daily sum paths were calculated to extract core velocity attributes. By computing vector shifts across regular step variables, the system projects expected short-term capital growth targets, allowing management to optimize logistics and inventory layers.</p>
        
        <table>
            <tr>
                <th>Forecast Horizon Target</th>
                <th>Projected Net Revenue Contribution Target</th>
            </tr>
            <tr><td>Forecast Day +1 Projection</td><td>${forecast[0]:.2f}</td></tr>
            <tr><td>Forecast Day +2 Projection</td><td>${forecast[1]:.2f}</td></tr>
            <tr><td>Forecast Day +3 Projection</td><td>${forecast[2]:.2f}</td></tr>
            <tr><td>Forecast Day +4 Projection</td><td>${forecast[3]:.2f}</td></tr>
            <tr><td>Forecast Day +5 Projection</td><td>${forecast[4]:.2f}</td></tr>
            <tr><td>Forecast Day +6 Projection</td><td>${forecast[5]:.2f}</td></tr>
            <tr><td>Forecast Day +7 Projection</td><td>${forecast[6]:.2f}</td></tr>
        </table>

        <h3>IV. Formal System Attestation</h3>
        <p class="no-indent">Report generated dynamically by the system execution engine. All algorithmic steps, including cleaning models, data engineering layers, and statistical tests, have completed in a steady state without errors.</p>

    </body>
    </html>
    """
    with open("executive_report.html", "w") as f:
        f.write(html_content)
    print("[REPORT ENGINE] Formal 'executive_report.html' successfully generated in workspace.")

def execute_operational_pipeline():
    print(config.REPORT_LINE_BREAK)
    print("INITIALIZING D2C E-COMMERCE GROWTH ANALYTICS AND MACHINE LEARNING PIPELINE")
    print(config.REPORT_LINE_BREAK)

    try:
        data_engine = EcomDataEngine()
        raw_dataframe = data_engine.generate_raw_transactions()
    except Exception as e:
        print(f"[CRITICAL FATAL FAULT] Stage 1 Data Extraction failed: {str(e)}")
        sys.exit(1)

    try:
        processed_feature_matrix = data_engine.feature_engineering_pipeline(raw_dataframe)
        
        print("\n[MAIN PIPELINE VERIFICATION] Confirming structural feature gain maps:")
        preview_columns = ["Transaction_ID", "Is_Weekend_Order", "Net_Checkout_Price", "City_Target_Encoded", "Log_User_Purchase_History"]
        print(processed_feature_matrix[preview_columns].head(3).to_string(index=False))
    except Exception as e:
        print(f"[CRITICAL FATAL FAULT] Stage 2 Feature Engineering failed: {str(e)}")
        sys.exit(1)

    print("\nAggregating multi-layered metrics to evaluate general health baseline...")
    kpi_report = GrowthAnalyticsEngine.compute_core_kpi_metrics(processed_feature_matrix)
    
    print("-" * 65)
    print(f" 🔹 Active Monthly Customer Footprint (MAU) : {kpi_report['MAU']} unique users")
    print(f" 🔹 Product Engagement Stickiness (DAU/MAU)  : {kpi_report['Stickiness_Pct']}%")
    print(f" 🔹 Gross Net Revenue Run-Rate Output        : ${kpi_report['Total_Revenue']:,}")
    print(f" 🔹 ARPU (Average Revenue Per Unique User)    : ${kpi_report['ARPU']}")
    print(f" 🔹 Net Promoter Customer Score Proxy (NPS)  : {kpi_report['NPS_Proxy']}")
    print("-" * 65)

    print("\nConstructing relative cohort retention calculation arrays...")
    cohort_pivot_array = GrowthAnalyticsEngine.calculate_monthly_cohort_matrix(processed_feature_matrix)

    print("\nProcessing configuration metric parameters for active UI redesign test...")
    ab_test_metrics = GrowthAnalyticsEngine.run_scientific_ab_hypothesis_test()
    
    print("-" * 65)
    print(f"Control Baseline Conversion Rate     : {ab_test_metrics['Control_CR']}%")
    print(f"Treatment Experimental Variant Rate  : {ab_test_metrics['Treatment_CR']}%")
    print(f"Computed Test Z-Score Value          : {ab_test_metrics['Z_Score']} (P-Value: {ab_test_metrics['P_Value']})")
    
    if ab_test_metrics["Is_Significant"]:
        print("STRATEGIC VERDICT: Variant is statistically significant. Deploy interface changes globally.")
    else:
        print("STRATEGIC VERDICT: Insufficient evidence. Retain old template variants.")
    print("-" * 65)

    print("\nFitting daily linear regression model vectors to forecast upcoming horizons...")
    daily_revenue_dataframe, trendline_array, forward_7day_forecast = GrowthAnalyticsEngine.run_linear_regression_modeling(processed_feature_matrix)
    
    print("\n7-DAY NEXT QUARTER FINANCIAL TARGET FORECAST OUTCOME:")
    for day_index, expected_value in enumerate(forward_7day_forecast, 1):
        print(f"Forecast Target Day +{day_index} Projected Revenue: ${expected_value:.2f}")

    # Generate the requested formal corporate report file
    generate_formal_html_report(kpi_report, ab_test_metrics, forward_7day_forecast)

    print("\nTransmitting processing results to interactive visualization dashboards...")
    try:
        viz_system = InteractiveVizEngine()
        viz_system.render_cohort_retention_heatmap(cohort_pivot_array)
        viz_system.render_predictive_sales_dashboard(daily_revenue_dataframe, trendline_array, forward_7day_forecast)
    except Exception as e:
        print(f"[VISUALIZATION ALARM] Main visualization generation caught an anomaly: {str(e)}")

    print("\n" + config.REPORT_LINE_BREAK)
    print("MODULE CONTROLLER EXECUTION CYCLE TERMINATED IN STEADY STATE SUCCESS")
    print(config.REPORT_LINE_BREAK)

if __name__ == "__main__":
    execute_operational_pipeline()