import numpy as np
import datetime

def calculate_sin_cos(date_obj):
    """Tinh sin cos dua vao ngay trong nam de du bao chinh xac hon theo mua"""
    day_of_year = date_obj.timetuple().tm_yday
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
    return sin_day, cos_day