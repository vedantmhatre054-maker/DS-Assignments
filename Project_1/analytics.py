import sys
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import config

class GrowthAnalyticsEngine:
    
    @staticmethod
    def compute_core_kpi_metrics(df: pd.DataFrame) -> dict:
        print("[ANALYTICS] Extracting enterprise dashboard KPIs...")
        
        try:
            df_jan_01 = df[df["Timestamp"].dt.strftime("%Y-%m-%d") == "2026-01-01"]
            dau = df_jan_01["User_ID"].nunique()
            
            df_jan_month = df[df["Timestamp"].dt.month == 1]
            mau = df_jan_month["User_ID"].nunique()
            
            stickiness_ratio = (dau / mau) * 100 if mau > 0 else 0.0
            
            successful_checkouts = df[df["Conversion_Outcome"] == 1]
            gross_revenue = successful_checkouts["Net_Checkout_Price"].sum()
            
            total_unique_users = df["User_ID"].nunique()
            arpu = gross_revenue / total_unique_users if total_unique_users > 0 else 0.0
            
            total_nps_responses = len(df)
            promoters = df[df["Customer_Satisfaction_Score"] >= 9].shape[0]
            detractors = df[df["Customer_Satisfaction_Score"] <= 6].shape[0]
            
            nps_score = ((promoters - detractors) / total_nps_responses) * 100 if total_nps_responses > 0 else 0.0
            
            return {
                "DAU": int(dau), "MAU": int(mau), "Stickiness_Pct": round(stickiness_ratio, config.ROUNDING_DECIMALS),
                "Total_Revenue": round(gross_revenue, config.ROUNDING_DECIMALS), "ARPU": round(arpu, config.ROUNDING_DECIMALS),
                "NPS_Proxy": int(nps_score)
            }
        except Exception as e:
            print(f"[ANALYTICS ACCUMULATION ERROR] Failed calculation structures: {str(e)}")
            return {"DAU": 0, "MAU": 0, "Stickiness_Pct": 0.0, "Total_Revenue": 0.0, "ARPU": 0.0, "NPS_Proxy": 0}

    @staticmethod
    def calculate_monthly_cohort_matrix(df: pd.DataFrame) -> pd.DataFrame:
        print("[ANALYTICS] Mapping customer cohorts across relative month lifetimes...")
        
        df_cohort = df.copy()
        df_cohort["Transaction_Month"] = df_cohort["Timestamp"].dt.to_period("M")
        df_cohort["Cohort_Month"] = df_cohort.groupby("User_ID")["Transaction_Month"].transform("min")
        
        df_cohort["Months_Elapsed"] = (df_cohort["Transaction_Month"].dt.year - df_cohort["Cohort_Month"].dt.year) * 12 + \
                                      (df_cohort["Transaction_Month"].dt.month - df_cohort["Cohort_Month"].dt.month)
        
        pivot_grid = df_cohort.groupby(["Cohort_Month", "Months_Elapsed"])["User_ID"].nunique().unstack()
        cohort_baselines = pivot_grid.iloc[:, 0]
        
        retention_percentage_matrix = pivot_grid.divide(cohort_baselines, axis=0) * 100
        return retention_percentage_matrix.fillna(0.0)

    @staticmethod
    def run_scientific_ab_hypothesis_test() -> dict:
        print("[A/B VALIDATION] Initiating hypothesis testing sequence...")
        
        p_A = config.CONTROL_CONVERSIONS / config.CONTROL_VISITORS
        p_B = config.TREATMENT_CONVERSIONS / config.TREATMENT_VISITORS
        
        total_conversions = config.CONTROL_CONVERSIONS + config.TREATMENT_CONVERSIONS
        total_visitors = config.CONTROL_VISITORS + config.TREATMENT_VISITORS
        pooled_p = total_conversions / total_visitors
        
        pooled_se = np.sqrt(pooled_p * (1.0 - pooled_p) * (1.0/config.CONTROL_VISITORS + 1.0/config.TREATMENT_VISITORS))
        
        z_score = (p_B - p_A) / pooled_se
        p_value = 2.0 * (1.0 - norm.cdf(abs(z_score)))
        
        is_significant = p_value <= config.ALPHA
        
        return {
            "Control_CR": round(p_A * 100, config.ROUNDING_DECIMALS),
            "Treatment_CR": round(p_B * 100, config.ROUNDING_DECIMALS),
            "Z_Score": round(z_score, 4), "P_Value": round(p_value, 5),
            "Is_Significant": is_significant
        }

    @staticmethod
    def run_linear_regression_modeling(df: pd.DataFrame) -> tuple:
        print("[MACHINE LEARNING] Training daily trend modeling vectors via LinearRegression...")
        
        sales_df = df[df["Conversion_Outcome"] == 1].copy()
        sales_df.set_index("Timestamp", inplace=True)
        
        daily_timeline = sales_df["Net_Checkout_Price"].resample("D").sum().reset_index()
        
        X_features = np.array(daily_timeline.index).reshape(-1, 1)
        y_target = daily_timeline["Net_Checkout_Price"].values
        
        reg_model = LinearRegression()
        reg_model.fit(X_features, y_target)
        
        historical_trendline = reg_model.predict(X_features)
        
        future_horizon_steps = 7
        future_indices = np.array(range(len(daily_timeline), len(daily_timeline) + future_horizon_steps)).reshape(-1, 1)
        future_revenue_forecast = reg_model.predict(future_indices)
        
        return daily_timeline, historical_trendline, future_revenue_forecast