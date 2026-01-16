import numpy as np
import pandas as pd
import datetime
from src import data_processor, model, api_client, utils
from src.config import TARGETS, N_LAGS


def update_lag_features(current_lags, new_prediction):
    """
    Cap nhat vector lag de tiep tuc du bao cac ngay tiep theo
    """
    updated_lags = []

    start_idx = 0
    for i, target_val in enumerate(new_prediction):
        chunk = current_lags[start_idx: start_idx + N_LAGS]
        chunk.pop(-1)
        chunk.insert(0, target_val)
        updated_lags.extend(chunk)

        start_idx += N_LAGS

    return updated_lags


def main():
    print("Buoc 1: Huan luyen mo hinh")
    df_clean, lag_cols = data_processor.prepare_data("data/hanoi_weather.csv")
    trained_model, feature_names = model.train_model(df_clean, lag_cols)

    print("\nBuoc 2: Lay du lieu tu API")
    current_lags_vector = api_client.fetch_realtime_lags()

    if current_lags_vector is None:
        print("Khong lay duoc du lieu API.")
        return

    #Du bao 7 ngay
    print("\nBuoc 3: Du bao 7 ngay tiep theo")
    print(f"{'NGÀY':<15} | {'MAX':<8} | {'MIN':<8} | {'AVG':<8} | {'MƯA (mm)':<8}")
    print("-" * 65)

    today = datetime.datetime.today()

    for i in range(1, 8):
        next_date = today + datetime.timedelta(days=i)

        #Tinh dac trung thoi gian cho ngay can du bao
        sin_d, cos_d = utils.calculate_sin_cos(next_date)
        year = next_date.year

        #Tao vector dau vao
        input_data = [sin_d, cos_d, year] + current_lags_vector

        #Chuyen thanh data frame
        X_pred = pd.DataFrame([input_data], columns=feature_names)

        #Du bao
        pred_result = trained_model.predict(X_pred)[0]

        #Xu ly ket qua
        p_max = round(pred_result[0], 2)
        p_min = round(pred_result[1], 2)
        p_avg = round(pred_result[2], 2)
        p_rain = round(max(0, pred_result[3]), 2)

        #In ket qua
        date_str = next_date.strftime("%d/%m/%Y")
        print(f"{date_str:<15} | {p_max:<8} | {p_min:<8} | {p_avg:<8} | {p_rain:<8}")

        #Cap nhat vector
        current_lags_vector = update_lag_features(current_lags_vector, [p_max, p_min, p_avg, p_rain])

if __name__ == "__main__":
    main()