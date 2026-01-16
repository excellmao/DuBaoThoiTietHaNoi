import requests
import datetime
import statistics
from src.config import API_KEY, LAT, LON, N_LAGS, TARGETS

def fetch_realtime_lags():
    print("Goi One Call API 3.0 ...")
    raw_history = {}

    for i in range(1, N_LAGS + 1):
        date_past = datetime.datetime.now() - datetime.timedelta(days=i)
        date_str = date_past.strftime("%Y-%m-%d")

        url = f"https://api.openweathermap.org/data/3.0/onecall/day_summary?lat={LAT}&lon={LON}&date={date_str}&appid={API_KEY}&units=metric"

        try:
            resp = requests.get(url)
            data = resp.json()

            if resp.status_code != 200:
                print(f"Loi API ngay {date_str}: {data.get('message')}")
                return None

            #Lay thong tin nhiet do tu API
            temp_data = data.get('temperature', {})
            t_max = temp_data.get('max', 0)
            t_min = temp_data.get('min', 0)

            #Tinh nhiet do trung binh
            t_morn = temp_data.get('morning', t_min)
            t_aft = temp_data.get('afternoon', t_max)
            t_even = temp_data.get('evening', t_min)
            t_night = temp_data.get('night', t_min)
            t_avg = (t_morn + t_aft + t_even + t_night) / 4

            #Lay thong tin mua
            precip_data = data.get('precipitation', {})
            rain_total = precip_data.get('total', 0.0)

            raw_history[i] = {
                'max': t_max,
                'min': t_min,
                'avg': t_avg,
                'rain' : rain_total
            }

            print(f"Ngày {date_str}: Max={t_max}, Min={t_min}, Rain={rain_total}mm")

        except Exception as e:
            print(f"Exception ngay -{i}: {e}")
            return None

    #Sap xep thanh vector
    final_vector = []
    for target_type in TARGETS:
        for i in range(1, N_LAGS + 1):
            if raw_history.get(i) is None:
                return None

            stats = raw_history[i]

            if 'tempmax' in target_type:
                final_vector.append(stats['max'])
            elif 'tempmin' in target_type:
                final_vector.append(stats['min'])
            elif 'precip' in target_type:
                final_vector.append(stats['rain'])
            elif 'temp' in target_type:
                final_vector.append(stats['avg'])

    return final_vector