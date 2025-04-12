# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# --- Load dữ liệu & mô hình ---
@st.cache_data
def load_data():
    df = joblib.load("rfm_data.pkl")
    if not df.index.dtype == "object":
        df.index = df.index.astype(str)
    return df

@st.cache_resource
def load_models():
    scaler = joblib.load("rfm_scaler.pkl")
    model = joblib.load("kmeans_model.pkl")
    return scaler, model

# --- Hàm ánh xạ cụm thành nhãn ---
def interpret_cluster(cluster_id):
    mapping = {
        0: "TRUNG BÌNH / KHÁCH PHỔ THÔNG",
        1: "CHURN / KHÁCH RỜI BỎ",
        2: "VIP / KHÁCH GIÁ TRỊ CAO"
    }
    return mapping.get(cluster_id, "Không xác định")

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Phân cụm khách hàng", layout="wide")
st.title("Trung Tâm Tin Học")
st.subheader(":mortar_board: Đồ án tốt nghiệp Data Science")
st.markdown("**Thực hiện bởi: Anh Thư - Giang Sơn**")

# --- 1. Giới thiệu dự án ---
st.header("1. Giới thiệu nội dung dự án")
st.markdown("""
Dự án nhằm phân khúc khách hàng dựa trên mô hình **RFM (Recency - Frequency - Monetary)**, giúp doanh nghiệp nhận diện phân cụm các khách hàng để doanh nghiệp có chiến lược phù hợp:

**Quy trình thực hiện:**
- Tiến hành EDA để hiểu hành vi khách hàng
- Tính toán chỉ số RFM
- Chuẩn hóa dữ liệu & áp dụng KMeans clustering
- Đánh giá bằng GMM - PCA

**Hình ảnh minh họa quy trình:**
""")

image_files = [
    ("1.EDA_product.png", "EDA: Phân tích sản phẩm"),
    ("2.EDA_sales.png", "EDA: Doanh thu theo ngày"),
    ("3.top_10.eda.png", "Top 10 khách hàng theo doanh thu"),
    ("4.RFM_historgram.png", "Biểu đồ histogram RFM"),
    ("5.ebow_kmeans.png", "Lựa chọn k (elbow method)"),
    ("6.bubble_chart_kmeans.png", "Phân cụm KMeans qua bubble chart"),
    ("7.GMM-PCA.png", "Phân cụm GMM-PCA")
]

for file, caption in image_files:
    if os.path.exists(file):
        st.image(file, caption=caption, use_container_width=True)
    else:
        st.warning(f" Không tìm thấy file: `{file}`")

st.markdown("""
Dựa vào RFM (Recency - Frequency - Monetary), một số thuật toán sử dụng nhằm phân ra 3 tệp khách hàng như sau:

- **Cluster 1: CHURN / KHÁCH RỜI BỎ**: Recency cao, Frequency thấp, Monetary thấp → khách đã lâu không quay lại → **Cần remarketing**
- **Cluster 2: VIP / GIÁ TRỊ CAO**: Recency thấp, Frequency cao, Monetary cao → **Giữ chân bằng loyalty/ưu đãi đặc biệt**
- **Cluster 0: TRUNG BÌNH / PHỔ THÔNG**: Trung bình cả 3 chỉ số → **Khách tiềm năng để upsell**
""")

# --- 2. Phân cụm khách hàng theo RFM ---
st.header("2. Nhập thông tin khách hàng")
df_rfm = load_data()
scaler, model = load_models()

input_method = st.radio("Chọn cách nhập thông tin khách hàng:", 
                        ["Nhập mã khách hàng", "Nhập thông tin khách hàng vào slider", "Tải file .csv"])

# --- Cách 1: Nhập mã khách hàng ---
if input_method == "Nhập mã khách hàng":
    customer_id = st.text_input("Nhập mã khách hàng", value="1808")
    st.write(f"Mã khách hàng: **{customer_id}**")

    if customer_id in df_rfm.index:
        try:
            rfm_row = df_rfm.loc[[customer_id]][["Recency", "Frequency", "Monetary"]]
            scaled_input = scaler.transform(rfm_row)
            cluster_label = model.predict(scaled_input)[0]
            cluster_name = interpret_cluster(cluster_label)

            st.header("3. Gợi ý phân cụm khách hàng")
            st.success(f":bar_chart: Cụm khách hàng số: **{cluster_label}** – {cluster_name}")
        except Exception as e:
            st.error(f"Không thể phân cụm khách hàng này: {str(e)}")
    else:
        st.error("Mã khách hàng không tồn tại trong dữ liệu.")

# --- Cách 2: Nhập slider ---
elif input_method == "Nhập thông tin khách hàng vào slider":
    st.subheader("Nhập thông tin khách hàng")
    customer_data = []
    for i in range(5):
        st.write(f"Khách hàng {i+1}")
        r = st.slider(f"Recency (Khách {i+1})", 1, 365, 100, key=f"recency_{i}")
        f = st.slider(f"Frequency (Khách {i+1})", 1, 50, 5, key=f"frequency_{i}")
        m = st.slider(f"Monetary (Khách {i+1})", 1, 1000, 100, key=f"monetary_{i}")
        customer_data.append([r, f, m])

    df_customer = pd.DataFrame(customer_data, columns=["Recency", "Frequency", "Monetary"])

    st.header("3. Phân cụm khách hàng")
    try:
        scaled_input = scaler.transform(df_customer)
        clusters = model.predict(scaled_input)
        df_customer["Cluster"] = clusters
        df_customer["Phân nhóm"] = df_customer["Cluster"].apply(interpret_cluster)

        st.dataframe(df_customer)
        for i, row in df_customer.iterrows():
            st.success(f" Khách hàng {i+1} thuộc cụm: **{row['Cluster']} – {row['Phân nhóm']}**")

    except Exception as e:
        st.error(f"Lỗi khi phân cụm: {str(e)}")

# --- Cách 3: Tải file CSV ---
else:
    st.subheader("📂 Hoặc: Tải file dữ liệu khách hàng lên (.csv)")
    st.markdown(" **Yêu cầu:** File cần có 3 cột: `Recency`, `Frequency`, `Monetary`. Cột `CustomerID` là tùy chọn.")
    uploaded_file = st.file_uploader("Tải lên file CSV chứa thông tin RFM", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            required_cols = {"Recency", "Frequency", "Monetary"}

            if not required_cols.issubset(df_uploaded.columns):
                st.error(" File cần có đầy đủ các cột: Recency, Frequency, Monetary")
            else:
                df_input = df_uploaded[["Recency", "Frequency", "Monetary"]]
                scaled_input = scaler.transform(df_input)
                clusters = model.predict(scaled_input)
                df_uploaded["Cluster"] = clusters
                df_uploaded["Phân nhóm"] = df_uploaded["Cluster"].apply(interpret_cluster)

                st.success("✅ Phân cụm thành công!")
                st.dataframe(df_uploaded)

                for i, row in df_uploaded.iterrows():
                    st.info(f" Khách hàng {row.get('CustomerID', i+1)} thuộc cụm: **{row['Cluster']} – {row['Phân nhóm']}**")

        except Exception as e:
            st.error(f"⚠️ Lỗi xử lý file: {str(e)}")
