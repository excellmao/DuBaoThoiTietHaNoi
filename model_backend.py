import os
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from statsmodels.tsa.statespace.sarimax import SARIMAX

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 21.0285
LON = 105.8542
MODEL_PATH = "hanoi_weather_model.pkl"

TARGETS = ['tempmax', 'tempmin', 'temp', 'precip', 'humidity', 'cloudcover']
N_LAGS = 7

# SARIMA CONFIG (NHẸ - CHẠY NHANH)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (1, 0, 1, 7)   # Chu kỳ 7 ngày (nhẹ hơn 365 rất nhiều)

MAX_HISTORY_DAYS = 120           # Chỉ dùng 4 tháng gần nhất

# ===========================================


# ---------- ICON ----------
def get_weather_icon(rain, cloud):
    if rain >= 50:
        return "⛈️", "Mưa to"
    elif rain >= 10:
        return "🌧️", "Mưa"
    elif rain > 1:
        return "🌦️", "Mưa nhỏ"

    if cloud > 80:
        return "☁️", "Nhiều mây"
    elif cloud > 50:
        return "⛅", "Có mây"
    else:
        return "☀️", "Nắng đẹp"


# ---------- SIN / COS ----------
def calculate_sin_cos(date_input):
    if isinstance(date_input, datetime.datetime):
        day_of_year = date_input.timetuple().tm_yday
    else:
        day_of_year = date_input.dt.dayofyear

    sin_d = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_d = np.cos(2 * np.pi * day_of_year / 365.25)
    return sin_d, cos_d


# ---------- PREPARE DATA ----------
def prepare_data(source):
    if isinstance(source, str):
        df = pd.read_csv(source)
    else:
        df = source.copy()

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.sort_values('datetime').reset_index(drop=True)

    for c in ['precip', 'cloudcover', 'humidity']:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    df['sin_day'], df['cos_day'] = calculate_sin_cos(df['datetime'])
    df['year'] = df['datetime'].dt.year

    lag_cols = []
    valid_targets = [c for c in TARGETS if c in df.columns]

    for col in valid_targets:
        for i in range(1, N_LAGS + 1):
            name = f"{col}_lag{i}"
            df[name] = df[col].shift(i)
            lag_cols.append(name)

    df = df.dropna().reset_index(drop=True)
    return df, lag_cols, valid_targets


# ---------- TRAIN ML ----------
def train_model(data_path):
    df_clean, lag_cols, valid_targets = prepare_data(data_path)

    features = ['sin_day', 'cos_day', 'year'] + lag_cols
    X = df_clean[features]
    y = df_clean[valid_targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump({
        'model': model,
        'features': features,
        'targets': valid_targets
    }, MODEL_PATH)

    score = model.score(X_test, y_test)
    return model, score, "OK"


# ---------- API REALTIME ----------
def fetch_realtime_lags():
    raw_history = {}

    for i in range(1, N_LAGS + 1):
        date_past = datetime.datetime.now() - datetime.timedelta(days=i)
        date_str = date_past.strftime('%Y-%m-%d')
        key = API_KEY or "fc847a0d05d7e86b8c94088a54b1a3b5"

        url = f"https://api.openweathermap.org/data/3.0/onecall/day_summary?lat={LAT}&lon={LON}&date={date_str}&appid={key}&units=metric"

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()
            t = data.get('temperature', {})
            p = data.get('precipitation', {})
            h = data.get('humidity', {})
            c = data.get('cloud_cover', {})

            vals = [t.get(k, 0) for k in ['morning', 'afternoon', 'evening', 'night']]
            t_avg = sum(vals) / 4 if vals else t.get('max', 0)

            raw_history[i] = {
                'tempmax': t.get('max', 0),
                'tempmin': t.get('min', 0),
                'temp': t_avg,
                'precip': p.get('total', 0),
                'humidity': h.get('afternoon', 70),
                'cloudcover': c.get('afternoon', 60)
            }
        except:
            return None

    try:
        saved_data = joblib.load(MODEL_PATH)
        model_targets = saved_data['targets']
    except:
        return None

    final_vector = []
    for target_col in model_targets:
        for i in range(1, N_LAGS + 1):
            final_vector.append(raw_history[i].get(target_col, 0))

    return final_vector


def update_lag_features(current_lags, new_prediction):
    updated = []
    start = 0
    for val in new_prediction:
        chunk = current_lags[start: start + N_LAGS]
        chunk.pop(-1)
        chunk.insert(0, val)
        updated.extend(chunk)
        start += N_LAGS
    return updated


def _fit_sarima_fast(series):
    try:
        model = SARIMAX(
            series,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        return model.fit(disp=False)
    except:
        return None


def predict_arima_basic(data_path, days_forecast=7):
    if isinstance(data_path, str):
        df = pd.read_csv(data_path)
    else:
        df = data_path.copy()

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.sort_values('datetime').reset_index(drop=True)

    # ===== LẤY ĐÚNG MÙA THEO THÁNG HIỆN TẠI =====
    current_month = datetime.datetime.now().month

    # Lấy +/- 1 tháng quanh tháng hiện tại
    valid_months = [
        (current_month - 1 - 1) % 12 + 1,
        current_month,
        (current_month + 1 - 1) % 12 + 1
    ]

    df = df[df['datetime'].dt.month.isin(valid_months)]

    # Giới hạn số dòng để SARIMA không quá chậm
    df = df.tail(180)
    # ==========================================

    forecast_results = {}

    for col in TARGETS:
        try:
            series = df[col].fillna(method='ffill').fillna(0).astype(float).values

            # Nếu dữ liệu ít → fallback
            if len(series) < 30 or len(set(series)) <= 1:
                forecast_results[col] = [series[-1]] * days_forecast
                continue

            model_fit = _fit_sarima_fast(series)

            if model_fit is None:
                forecast_results[col] = [np.mean(series[-7:])] * days_forecast
            else:
                forecast = model_fit.forecast(steps=days_forecast)
                forecast_results[col] = forecast.tolist()

        except:
            forecast_results[col] = [np.mean(df[col].tail(7))] * days_forecast

    today = datetime.datetime.now()
    final_days = []

    for i in range(days_forecast):
        next_date = today + datetime.timedelta(days=i + 1)

        val_max = forecast_results['tempmax'][i]
        val_min = forecast_results['tempmin'][i]
        val_rain = max(0, forecast_results['precip'][i])
        val_cloud = max(0, min(100, forecast_results['cloudcover'][i]))
        val_humid = max(0, min(100, forecast_results['humidity'][i]))

        # ===== CHẶN GIÁ TRỊ PHI THỰC TẾ =====
        noise = np.random.normal(0, 0.6)

        val_max = val_max + noise
        val_min = val_min + noise * 0.5

        # Chặn hợp lý cho Hà Nội
        val_max = min(max(val_max, 12), 38)
        val_min = min(max(val_min, 8), 30)

        if val_max < val_min:
            val_max, val_min = val_min, val_max
        # ==================================

        icon, condition = get_weather_icon(val_rain, val_cloud)

        final_days.append({
            'date': next_date.strftime('%d/%m'),
            'weekday': next_date.strftime('%A'),
            'icon': icon,
            'condition': condition,
            'max': float(val_max),
            'min': float(val_min),
            'rain': float(val_rain),
            'humid': float(val_humid)
        })

    return final_days
