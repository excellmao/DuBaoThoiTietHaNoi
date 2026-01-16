import pandas as pd
import numpy as np
from src.config import COL_DATE, COL_PRECIP, TARGETS, N_LAGS
from src.utils import calculate_sin_cos


def prepare_data(filepath):
    """
    Doc CSV -> lam sach -> lag features -> tra ve du lieu da train
    """
    #Doc du lieu
    df = pd.read_csv(filepath)

    #Lam sach
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')

    #Xu ly mua, NaN = 0
    if COL_PRECIP in df.columns:
        df[COL_PRECIP] = df[COL_PRECIP].fillna(0.0)

    #Sap xep theo thoi gian
    df = df.sort_values(COL_DATE).reset_index(drop=True)

    #Feature tinh
    days = df[COL_DATE].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * days / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * days / 365.25)
    df['year'] = df[COL_DATE].dt.year

    #Lag features (du lieu tinh)
    lag_cols_names = []

    for col in TARGETS:
        for i in range(1, N_LAGS + 1):
            col_name = f"{col}_lag{i}"
            df[col_name] = df[col].shift(i)
            lag_cols_names.append(col_name)

    #Xoa cac hang dau do NaN shift
    df_clean = df.dropna().reset_index(drop=True)

    return df_clean, lag_cols_names