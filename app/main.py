import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.append("src")
from features import process_dataframe

app = FastAPI(
    title="AI_PAYT - Hanoi Rent Pricing & Deal Evaluator",
    description="Hệ thống AI định giá phòng trọ Hà Nội, phát hiện Deal Hời & giải thích bằng SHAP (XAI)",
    version="1.0.0"
)

artifact = joblib.load("models/rent_pricing_pipeline.joblib")
model = artifact["model"]
explainer = artifact["explainer"]
FEATURE_COLS = artifact["feature_cols"]

class RoomRequest(BaseModel):
    title: str = Field(..., example="Cho thuê CCMN full đồ ngõ 58 Trần Bình, Cầu Giấy")
    description: str = Field(..., example="Phòng 28m2 có điều hòa, nóng lạnh, thang máy, ban công thoáng, không chung chủ.")
    area_m2: float = Field(..., example=28.0)
    lat: float = Field(..., example=21.0315, description="Vĩ độ GPS (VD: khu Cầu Giấy ~ 21.03, Hà Đông ~ 20.98)")
    lon: float = Field(..., example=105.7805, description="Kinh độ GPS (VD: ~ 105.78)")
    asked_price_million: float = Field(..., example=3.5, description="Giá chủ nhà đang báo (Triệu VNĐ)")

@app.post("/api/v1/evaluate-deal")
def evaluate_deal(req: RoomRequest):
    df_in = pd.DataFrame([req.model_dump()])
    feat_df = process_dataframe(df_in, is_training=False)
    X_infer = feat_df[FEATURE_COLS]

    # Dự đoán giá hợp lý (quy đổi ngược từ hàm log1p)
    pred_log = model.predict(X_infer)[0]
    fair_price = float(np.expm1(pred_log))

    # Đánh giá mức độ Hời / Đắt
    diff_pct = ((req.asked_price_million - fair_price) / fair_price) * 100
    if diff_pct <= -10:
        verdict = "🔥 DEAL HỜI (Rẻ hơn thị trường)"
    elif diff_pct >= 12:
        verdict = "⚠️ GIÁ CAO (Đắt hơn mặt bằng chung)"
    else:
        verdict = "✅ HỢP LÝ (Đúng giá thị trường)"

    # Giải thích bằng SHAP Values
    shap_vals = explainer.shap_values(X_infer)[0]
    impacts = sorted(
        zip(FEATURE_COLS, X_infer.iloc[0].values, shap_vals),
        key=lambda x: abs(x[2]),
        reverse=True
    )

    top_explanations = [
        {
            "feature": f,
            "value": round(float(val), 2),
            "effect": "Làm TĂNG giá (+)" if imp > 0 else "Làm GIẢM giá (-)",
            "shap_impact": round(float(imp), 4)
        }
        for f, val, imp in impacts[:5]
    ]

    return {
        "predicted_fair_price_million": round(fair_price, 2),
        "asked_price_million": req.asked_price_million,
        "price_diff_percent": f"{diff_pct:+.1f}%",
        "verdict": verdict,
        "extracted_info": {
            "min_dist_to_uni_km": round(float(feat_df["min_dist_to_uni_cluster"].iloc[0]), 2),
            "total_amenities_detected": int(feat_df["total_amenities"].iloc[0])
        },
        "shap_explanations": top_explanations
    }

@app.get("/", response_class=HTMLResponse)
def home_ui():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>AI_PAYT - Định giá Phòng trọ Hà Nội</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 760px; margin: 30px auto; padding: 0 20px; background: #f8fafc; color: #1e293b; }
            .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
            input, textarea { width: 100%; padding: 10px; margin: 6px 0 14px; border: 1px solid #cbd5e1; border-radius: 6px; box-sizing: border-box; }
            .row { display: flex; gap: 12px; }
            .row > div { flex: 1; }
            button { background: #2563eb; color: white; border: none; padding: 12px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; width: 100%; }
            button:hover { background: #1d4ed8; }
            #result { margin-top: 20px; display: none; padding: 16px; border-radius: 8px; background: #f1f5f9; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🏠 AI Định giá & Săn Deal Phòng Trọ Hà Nội (AI_PAYT)</h2>
            <label>Tiêu đề tin đăng:</label>
            <input id="title" value="Cho thuê CCMN ngõ 58 Trần Bình, gần ĐHQG">
            <label>Mô tả chi tiết (AI tự bóc tách tiện ích):</label>
            <textarea id="desc" rows="3">Phòng 28m2 full đồ có điều hòa, nóng lạnh, thang máy, ban công thoáng, giờ giấc tự do không chung chủ.</textarea>
            <div class="row">
                <div><label>Diện tích (m²):</label><input type="number" id="area" value="28"></div>
                <div><label>Giá chủ rao (Triệu):</label><input type="number" step="0.1" id="price" value="3.5"></div>
            </div>
            <div class="row">
                <div><label>Vĩ độ (Lat - VD Hà Đông: 20.9806):</label><input type="number" step="0.0001" id="lat" value="20.9806"></div>
                <div><label>Kinh độ (Lon - VD Hà Đông: 105.7879):</label><input type="number" step="0.0001" id="lon" value="105.7879"></div>
            </div>
            <button onclick="checkDeal()">🔍 Phân tích bằng AI & SHAP</button>
            <div id="result"></div>
        </div>
        <script>
            async function checkDeal() {
                const resBox = document.getElementById('result');
                resBox.style.display = 'block';
                resBox.innerHTML = '⏳ Đang phân tích...';
                const payload = {
                    title: document.getElementById('title').value,
                    description: document.getElementById('desc').value,
                    area_m2: parseFloat(document.getElementById('area').value),
                    asked_price_million: parseFloat(document.getElementById('price').value),
                    lat: parseFloat(document.getElementById('lat').value),
                    lon: parseFloat(document.getElementById('lon').value)
                };
                const res = await fetch('/api/v1/evaluate-deal', {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
                });
                const data = await res.json();
                let shapHtml = data.shap_explanations.map(x => `<li><b>${x.feature}</b> (${x.value}): ${x.effect} [SHAP: ${x.shap_impact}]</li>`).join('');
                resBox.innerHTML = `
                    <h3>Kết luận: ${data.verdict}</h3>
                    <p>💰 <b>Giá AI định giá hợp lý:</b> ${data.predicted_fair_price_million} Triệu VNĐ (Lệch: ${data.price_diff_percent})</p>
                    <p>📍 <b>Khoảng cách tới cụm ĐH gần nhất:</b> ${data.extracted_info.min_dist_to_uni_km} km | <b>Số tiện ích phát hiện:</b> ${data.extracted_info.total_amenities_detected}/7</p>
                    <p>🧠 <b>Giải thích mô hình (Top 5 yếu tố SHAP):</b></p><ul>${shapHtml}</ul>
                `;
            }
        </script>
    </body>
    </html>
    """