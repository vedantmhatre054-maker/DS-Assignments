import numpy as np

RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)

DATA_ROWS = 1500           
USER_POOL_SIZE = 400       
HISTORICAL_VISITORS = 50000

HIGH_CARDINALITY_CITIES = [
    "New York", "London", "Tokyo", "Paris", "Berlin", 
    "Sydney", "Mumbai", "Toronto", "Singapore", "San Francisco"
]
CITY_PROBABILITY_WEIGHTS = [0.22, 0.18, 0.14, 0.10, 0.12, 0.06, 0.06, 0.04, 0.05, 0.03]

ALPHA = 0.05               
POWER = 0.80               
CONTROL_VISITORS = 12500
CONTROL_CONVERSIONS = 1125
TREATMENT_VISITORS = 12500
TREATMENT_CONVERSIONS = 1288 

ROUNDING_DECIMALS = 2
REPORT_LINE_BREAK = "=" * 75