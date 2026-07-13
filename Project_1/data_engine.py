import sys
import pandas as pd
import numpy as np
import config

class EcomDataEngine:
    def __init__(self):
        self.total_rows = config.DATA_ROWS
        self.cities = config.HIGH_CARDINALITY_CITIES
        self.weights = config.CITY_PROBABILITY_WEIGHTS
        self.user_ids = [f"USR_{i:04d}" for i in range(1, config.USER_POOL_SIZE + 1)]
        print(f"[INITIALIZATION] EcomDataEngine registered with target dimensions: {self.total_rows} rows.")

    def generate_raw_transactions(self) -> pd.DataFrame:
        print("[ETL EXTRACT] Generating synthetic relational transaction logs...")
        
        time_index = pd.date_range(
            start="2026-01-01 00:00:00", 
            periods=self.total_rows, 
            freq="95Min"
        )
        
        try:
            raw_matrix = {
                "Transaction_ID": [f"TXN_{i:06d}" for i in range(1, self.total_rows + 1)],
                "User_ID": np.random.choice(self.user_ids, size=self.total_rows),
                "Timestamp": time_index,
                "Market_City": np.random.choice(self.cities, size=self.total_rows, p=self.weights),
                "Sticker_Price": np.random.uniform(12.50, 299.99, size=self.total_rows),
                "Markdown_Pct": np.random.choice(
                    [0.0, 0.05, 0.10, 0.20, 0.35, 0.50], 
                    size=self.total_rows, 
                    p=[0.45, 0.25, 0.12, 0.10, 0.05, 0.03]
                ),
                "Historical_User_Purchases": np.random.randint(1, 75, size=self.total_rows),
                "Customer_Satisfaction_Score": np.random.choice(
                    [10, 9, 8, 7, 5, 2, 1], 
                    size=self.total_rows, 
                    p=[0.30, 0.25, 0.15, 0.12, 0.10, 0.05, 0.03]
                ),
                "Conversion_Outcome": np.random.choice([1, 0], size=self.total_rows, p=[0.35, 0.65])
            }
            
            df = pd.DataFrame(raw_matrix)
            
            print("[DATA CORRUPTION LAYER] Injecting missing timestamps and corrupt variables...")
            corrupt_time_indices = np.random.choice(df.index, size=35, replace=False)
            df.loc[corrupt_time_indices, "Timestamp"] = pd.NaT
            
            corrupt_price_indices = np.random.choice(df.index, size=15, replace=False)
            df.loc[corrupt_price_indices, "Sticker_Price"] = np.nan
            
            df.to_csv("raw_data.csv", index=False)
            print(f"[ETL COMPLETED] File 'raw_data.csv' exported. Shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to synthesize database fields: {str(e)}")
            sys.exit(1)

    def feature_engineering_pipeline(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        print("[FEATURE ENGINEERING] Initiating transformation pipeline across 5 operational strategy types...")
        
        if raw_df is None or raw_df.empty:
            raise ValueError("Passed DataFrame is completely empty. Feature pipeline aborted.")

        df = raw_df.copy()
        initial_count = len(df)
        
        df = df.dropna(subset=["Timestamp"])
        
        median_price = df["Sticker_Price"].median()
        df["Sticker_Price"] = df["Sticker_Price"].fillna(median_price)
        
        scrubbed_count = len(df)
        print(f"Imputation metrics: Removed {initial_count - scrubbed_count} missing rows. Kept {scrubbed_count} rows.")

        df["Is_Premium_Market_NY"] = (df["Market_City"] == "New York").astype(int)
        df["Is_Premium_Market_LDN"] = (df["Market_City"] == "London").astype(int)

        df["Order_Hour"] = df["Timestamp"].dt.hour
        df["Day_Of_Week"] = df["Timestamp"].dt.dayofweek
        df["Is_Weekend_Order"] = (df["Day_Of_Week"] >= 5).astype(int)
        df["Month_Numerical"] = df["Timestamp"].dt.month

        df["Absolute_Discount_Value"] = df["Sticker_Price"] * df["Markdown_Pct"]
        df["Net_Checkout_Price"] = df["Sticker_Price"] - df["Absolute_Discount_Value"]
        df["Unit_Economic_Velocity"] = df["Sticker_Price"] / (df["Historical_User_Purchases"] + 1)

        global_conversion_mean = df["Conversion_Outcome"].mean()
        city_target_probability_map = df.groupby("Market_City")["Conversion_Outcome"].mean()
        df["City_Target_Encoded"] = df["Market_City"].map(city_target_probability_map).fillna(global_conversion_mean)

        df["Log_User_Purchase_History"] = np.log10(df["Historical_User_Purchases"] + 1)

        print("Pipeline complete. 10 highly granular machine learning features engineered successfully.")
        return df