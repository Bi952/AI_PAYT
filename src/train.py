import os
import joblib
import optuna
import shap
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import process_dataframe

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "area_m2", "lat", "lon",
    "dist_to_cluster_thanh_xuan_ha_dong",
    "dist_to_cluster_cau_giay",
    "dist_to_cluster_bach_kinh_xay",
    "dist_to_cluster_dong_da",
    "dist_to_ho_guom_center",
    "min_dist_to_uni_cluster",
    "is_ccmn", "has_ac", "has_water_heater",
    "has_elevator", "has_balcony", "no_curfew",
    "has_kitchen", "total_amenities"
]

def print_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {model_name:<28} | MAE: {mae:.3f} Tr | RMSE: {rmse:.3f} Tr | R2: {r2:.4f}")
    return mae

def main():
    raw_path = "data/raw/hanoi_rooms_raw.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError("Chưa có dữ liệu! Hãy chạy 'python src/scraper.py' trước.")

    df_raw = pd.read_csv(raw_path)
    print(f"📥 Đọc {len(df_raw)} mẫu dữ liệu thô.")

    # Làm sạch & Trích xuất đặc trưng
    df_clean = process_dataframe(df_raw, is_training=True)
    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv("data/processed/hanoi_rooms_clean.csv", index=False, encoding="utf-8-sig")
    print(f"🧹 Đã lọc nhiễu & trích xuất đặc trưng -> Còn {len(df_clean)} mẫu sạch (Lưu tại data/processed/).\n")

    X = df_clean[FEATURE_COLS]
    y = df_clean["price_million"]
    # Log-transform nhãn mục tiêu để xử lý phân phối lệch phải (Right-skewed)
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_real = np.expm1(y_test_log)

    print("=== KẾT QUẢ SO SÁNH CÁC MÔ HÌNH TRÊN TẬP TEST ===")
    # 1. Baseline: Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train_log)
    print_metrics(y_test_real, np.expm1(ridge.predict(X_test)), "1. Baseline (Ridge)")

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_log)
    print_metrics(y_test_real, np.expm1(rf.predict(X_test)), "2. Random Forest")

    # 3. LightGBM + Optuna Hyperparameter Tuning
    print("\n⚙️ Đang chạy Optuna (25 trials x 5-Fold CV) để tối ưu LightGBM...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_maes = []
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train_log.iloc[tr_idx], y_train_log.iloc[val_idx]

            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds_real = np.expm1(model.predict(X_val))
            cv_maes.append(mean_absolute_error(np.expm1(y_val), preds_real))
        return np.mean(cv_maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)

    best_lgbm = LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
    best_lgbm.fit(X_train, y_train_log)
    print_metrics(y_test_real, np.expm1(best_lgbm.predict(X_test)), "3. Tuned LightGBM (Optuna)")

    # 4. Tạo SHAP Explainer để giải thích mô hình
    explainer = shap.TreeExplainer(best_lgbm)

    # 5. Lưu toàn bộ Pipeline
    os.makedirs("models", exist_ok=True)
    model_path = "models/rent_pricing_pipeline.joblib"
    joblib.dump({
        "model": best_lgbm,
        "explainer": explainer,
        "feature_cols": FEATURE_COLS,
        "best_params": study.best_params
    }, model_path)
    print(f"\n💾 Đã đóng gói Model & SHAP Explainer tại '{model_path}'")

if __name__ == "__main__":
    main()