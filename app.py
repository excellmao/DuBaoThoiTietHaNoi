import streamlit as st
import pandas as pd
import datetime
import joblib
import model_backend as backend  # Import file backend logic

# --- 1. CẤU HÌNH TRANG & GIAO DIỆN ---
st.set_page_config(page_title="Dự Báo Thời Tiết Hà Nội", page_icon="🌤️", layout="wide")

# CSS cho giao diện thẻ (Card UI)
st.markdown("""
<style>
    .weather-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .weather-card:hover {
        transform: scale(1.02);
    }
    .big-icon { font-size: 50px; margin-bottom: 10px; }
    .temp-text { font-size: 24px; font-weight: bold; color: #ff4b4b; }
    .date-text { font-size: 18px; font-weight: bold; color: #333; }
    .sub-text { font-size: 14px; color: #555; }
    .condition-text { font-weight: bold; color: #444; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR (BẢNG ĐIỀU KHIỂN) ---
st.sidebar.header("🔧 Bảng điều khiển")
st.sidebar.info("Dự án: Nhóm 7 - Dự báo thời tiết Hà Nội")

# === SỬA LỖI Ở ĐÂY: Thêm key='csv_uploader' để tránh trùng lặp ===
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV dữ liệu (nếu có)",
    type=['csv'],
    key='csv_uploader'
)

# Chọn thuật toán
model_option = st.sidebar.radio(
    "Chọn thuật toán:",
    ("Linear Regression (Hồi quy)", "ARIMA (Chuỗi thời gian)")
)

# Nút huấn luyện (Chỉ hiện khi chọn Linear Regression)
if model_option == "Linear Regression (Hồi quy)":
    if st.sidebar.button("🚀 Huấn luyện lại mô hình"):
        # Ưu tiên dùng file upload, nếu không có thì dùng file mặc định trong backend
        data_source = uploaded_file if uploaded_file else "hanoi_weather.csv"

        with st.spinner("Đang huấn luyện mô hình..."):
            try:
                model, score, msg = backend.train_model(data_source)
                if model:
                    st.sidebar.success(f"✅ Huấn luyện xong!\nR2 Score: {score:.4f}")
                else:
                    st.sidebar.error(f"❌ Lỗi: {msg}")
            except Exception as e:
                st.sidebar.error(f"Lỗi: {str(e)}")
else:
    st.sidebar.info("ℹ️ ARIMA là mô hình thống kê, sẽ chạy trực tiếp trên dữ liệu gốc khi bạn bấm Dự báo.")

# --- 3. MÀN HÌNH CHÍNH (MAIN CONTENT) ---
st.title("🌤️ Dự Báo Thời Tiết Hà Nội")
st.write(f"Đang sử dụng mô hình: **{model_option}**")

# Nút bắt đầu dự báo
if st.button("🔮 Bắt đầu Dự báo ngay", type="primary"):

    forecast_days = []

    # === TRƯỜNG HỢP A: ARIMA ===
    if model_option == "ARIMA (Chuỗi thời gian)":
        with st.spinner("⏳ Đang chạy mô hình ARIMA (Mất khoảng 5-10s)..."):
            data_source = uploaded_file if uploaded_file else "hanoi_weather.csv"
            try:
                forecast_days = backend.predict_arima_basic(data_source)
                st.success("✅ Dự báo bằng ARIMA hoàn tất!")
            except Exception as e:
                st.error(f"❌ Lỗi khi chạy ARIMA: {e}")
                st.stop()

    # === TRƯỜNG HỢP B: LINEAR REGRESSION ===
    else:
        # 1. Load Model & Kiểm tra file .pkl cũ/mới
        try:
            saved_data = joblib.load(backend.MODEL_PATH)

            # Kiểm tra kỹ xem file model có đủ thông tin không
            if isinstance(saved_data, dict) and 'features' in saved_data:
                model = saved_data['model']
                feature_names = saved_data['features']
                targets = saved_data['targets']
            else:
                st.error(
                    "⚠️ File mô hình hiện tại là phiên bản cũ. Vui lòng bấm nút 'Huấn luyện lại mô hình' ở thanh bên trái!")
                st.stop()

        except FileNotFoundError:
            st.error("⚠️ Chưa tìm thấy file mô hình. Hãy bấm nút 'Huấn luyện lại mô hình' ở thanh bên trái trước!")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Lỗi khi đọc mô hình: {e}")
            st.stop()

        # 2. Lấy dữ liệu API
        with st.spinner("📡 Đang lấy dữ liệu thời tiết thực tế (7 ngày qua)..."):
            current_lags = backend.fetch_realtime_lags()

        if not current_lags:
            st.error("❌ Không lấy được dữ liệu API. Vui lòng kiểm tra lại kết nối mạng hoặc API Key trong file .env")
            st.stop()

        # 3. Vòng lặp Dự báo (Recursive Forecasting)
        today = datetime.datetime.now()
        progress_bar = st.progress(0)

        for i in range(1, 8):
            next_date = today + datetime.timedelta(days=i)

            # Tính toán Feature đầu vào
            sin_d, cos_d = backend.calculate_sin_cos(next_date)
            input_data = [sin_d, cos_d, next_date.year] + current_lags

            # Tạo DataFrame đúng tên cột
            X_pred = pd.DataFrame([input_data], columns=feature_names)

            # Dự báo
            pred = model.predict(X_pred)[0]

            # Map kết quả sang dictionary
            raw_result = dict(zip(targets, pred))

            # --- XỬ LÝ SỐ LIỆU (POST-PROCESSING) ---
            val_max = raw_result.get('tempmax')
            val_min = raw_result.get('tempmin')

            # Chặn số âm cho mưa
            val_rain = max(0.0, raw_result.get('precip', 0.0))

            # Chặn 0-100 cho độ ẩm/mây
            val_humid = max(0.0, min(100.0, raw_result.get('humidity', 75.0)))
            val_cloud = max(0.0, min(100.0, raw_result.get('cloudcover', 50.0)))

            # Logic: Max phải lớn hơn Min
            if val_max < val_min: val_max, val_min = val_min, val_max

            # Lấy Icon
            icon, condition = backend.get_weather_icon(val_rain, val_cloud)

            forecast_days.append({
                'date': next_date.strftime('%d/%m'),
                'weekday': next_date.strftime('%A'),
                'icon': icon,
                'condition': condition,
                'max': val_max,
                'min': val_min,
                'rain': val_rain,
                'humid': val_humid
            })

            # Cập nhật lags cho vòng lặp sau
            new_row_clean = [val_max, val_min, raw_result.get('temp'), val_rain, val_humid, val_cloud]
            current_lags = backend.update_lag_features(current_lags, new_row_clean)

            progress_bar.progress(int(i / 7 * 100))

        progress_bar.empty()

    # --- 4. HIỂN THỊ KẾT QUẢ ---
    if forecast_days:
        st.success("✅ Dự báo hoàn tất!")

        # Hàng 1
        cols = st.columns(4)
        for idx, day in enumerate(forecast_days[:4]):
            with cols[idx]:
                st.markdown(f"""
                <div class="weather-card">
                    <div class="date-text">{day['date']}</div>
                    <div class="sub-text">{day['weekday']}</div>
                    <div class="big-icon">{day['icon']}</div>
                    <div class="temp-text">{day['max']:.1f}°C</div>
                    <div class="sub-text">Min: {day['min']:.1f}°C</div>
                    <hr style="margin: 10px 0;">
                    <div class="sub-text">💧 {day['humid']:.0f}% | ☔ {day['rain']:.1f}mm</div>
                    <div class="condition-text">{day['condition']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Hàng 2
        cols2 = st.columns(4)
        for idx, day in enumerate(forecast_days[4:]):
            with cols2[idx]:
                st.markdown(f"""
                <div class="weather-card">
                    <div class="date-text">{day['date']}</div>
                    <div class="sub-text">{day['weekday']}</div>
                    <div class="big-icon">{day['icon']}</div>
                    <div class="temp-text">{day['max']:.1f}°C</div>
                    <div class="sub-text">Min: {day['min']:.1f}°C</div>
                    <hr style="margin: 10px 0;">
                    <div class="sub-text">💧 {day['humid']:.0f}% | ☔ {day['rain']:.1f}mm</div>
                    <div class="condition-text">{day['condition']}</div>
                </div>
                """, unsafe_allow_html=True)