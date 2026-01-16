import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("Khong tim thay API Key! Kiem tra xem da co file .env chua.")

LAT = 21.0285
LON = 105.8542
DATA_PATH = "data/hanoi_weather.csv"
MODEL_PATH = "hanoi_weather_model.pkl"

COL_DATE = 'datetime'
COL_TEMP_MAX = 'tempmax'
COL_TEMP_MIN = 'tempmin'
COL_TEMP_AVG = 'temp'
COL_PRECIP = 'precip'

FEATURE_ORDER = ['sin_day', 'cos_day', 'year']
TARGETS = [COL_TEMP_MAX, COL_TEMP_MIN, COL_TEMP_AVG, COL_PRECIP]
N_LAGS = 5