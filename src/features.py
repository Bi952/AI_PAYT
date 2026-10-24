import re
import numpy as np
import pandas as pd

# Tọa độ GPS các cụm trường Đại học & Trung tâm tại Hà Nội
HANOI_HOTSPOTS = {
    "cluster_thanh_xuan_ha_dong": (20.9806, 105.7879), # Cụm PTIT, Kiến Trúc, UTT, KHTN
    "cluster_cau_giay": (21.0368, 105.7826),           # Cụm ĐHQG, Sư Phạm, Thương Mại, Báo Chí
    "cluster_bach_kinh_xay": (21.0041, 105.8438),      # Cụm Bách Khoa, NEU, Xây Dựng
    "cluster_dong_da": (21.0075, 105.8247),            # Cụm Thủy Lợi, Ngân Hàng, Công Đoàn
    "ho_guom_center": (21.0285, 105.8542)              # Trung tâm Hồ Gươm
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Công thức Haversine tính khoảng cách đường chim bay (km) giữa 2 điểm GPS."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def extract_nlp_amenities(text: str) -> dict:
    """Trích xuất tiện ích phòng trọ từ văn bản mô tả bằng Regex."""
    t = str(text).lower()
    return {
        "is_ccmn": int(bool(re.search(r"ccmn|chung cư mini|tòa nhà|toà nhà", t))),
        "has_ac": int(bool(re.search(r"điều hòa|điều hoà|máy lạnh|full đồ", t))),
        "has_water_heater": int(bool(re.search(r"nóng lạnh|bình nóng lạnh|full đồ", t))),
        "has_elevator": int(bool(re.search(r"thang máy", t))),
        "has_balcony": int(bool(re.search(r"ban công|cửa sổ thoáng|thoáng sáng", t))),
        "no_curfew": int(bool(re.search(r"không chung chủ|ko chung chủ|giờ giấc tự do", t))),
        "has_kitchen": int(bool(re.search(r"kệ bếp|bàn bếp|tủ bếp|nấu ăn", t))),
        "is_shared_room": int(bool(re.search(r"ở ghép|tìm nam|tìm nữ|share phòng|người ở cùng|nhượng phòng", t)))
    }

def process_dataframe(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    df = df.copy()

    # 1. Trích xuất đặc trưng văn bản (NLP) từ Title + Description
    combined_text = df["title"].fillna("") + " " + df["description"].fillna("")
    nlp_df = pd.DataFrame(list(combined_text.apply(extract_nlp_amenities)), index=df.index)
    df = pd.concat([df, nlp_df], axis=1)

    # 2. Làm sạch dữ liệu & lọc Outlier (Chỉ áp dụng khi chuẩn bị tập Train/Test)
    if is_training:
        df["price_million"] = df["price_vnd"] / 1e6
        
        # Loại bỏ tin tìm người ở ghép và các dòng thiếu tọa độ GPS
        df = df[df["is_shared_room"] == 0]
        df = df.dropna(subset=["lat", "lon", "area_m2", "price_million"])
        
        # Lọc ngưỡng vật lý hợp lý của phòng trọ sinh viên Hà Nội
        df = df[(df["area_m2"] >= 10.0) & (df["area_m2"] <= 75.0)]
        df = df[(df["price_million"] >= 1.2) & (df["price_million"] <= 12.0)]
        
        # Lọc Outlier theo đơn giá / m2 (Từ 70k/m2 đến 380k/m2)
        df["price_per_m2"] = df["price_million"] / df["area_m2"]
        df = df[(df["price_per_m2"] >= 0.07) & (df["price_per_m2"] <= 0.38)]

    # 3. Tính khoảng cách từ phòng trọ đến các Hotspot tại Hà Nội
    uni_dist_cols = []
    for name, (h_lat, h_lon) in HANOI_HOTSPOTS.items():
        col_name = f"dist_to_{name}"
        df[col_name] = haversine_distance(df["lat"], df["lon"], h_lat, h_lon)
        if "cluster_" in name:
            uni_dist_cols.append(col_name)

    # Khoảng cách tới cụm trường đại học gần nhất
    df["min_dist_to_uni_cluster"] = df[uni_dist_cols].min(axis=1)

    # 4. Điểm tổng hợp tiện ích
    amenity_cols = [
        "is_ccmn", "has_ac", "has_water_heater",
        "has_elevator", "has_balcony", "no_curfew", "has_kitchen"
    ]
    df["total_amenities"] = df[amenity_cols].sum(axis=1)

    return df