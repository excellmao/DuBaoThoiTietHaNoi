from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # <--- 1. Thư viện dùng để lưu model
from src.config import TARGETS, MODEL_PATH # <--- 2. Nhớ import MODEL_PATH

def train_model(df, lag_cols):
    """Huan luyen va danh gia mo hinh"""
    features = ['sin_day', 'cos_day', 'year'] + lag_cols

    X = df[features]
    y = df[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    print(f"Dang huan luyen tren {len(X_train)} dong du lieu...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    #Danh gia
    preds = model.predict(X_test)
    print(f"   R2 Score: {r2_score(y_test, preds):.4f}")
    print(f"   MAE: {mean_absolute_error(y_test, preds):.2f}")

    #Luu model
    print(f"Đang lưu mô hình vào: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    return model, features

def load_trained_model():
    """Hàm tải mô hình đã lưu (nếu có)"""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Đã tải mô hình từ: {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print("Chưa có file model.pkl. Cần huấn luyện lại.")
        return None