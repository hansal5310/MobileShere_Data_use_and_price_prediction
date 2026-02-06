import streamlit as st
import pandas as pd
import pickle
import io
import sqlite3

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="MobileSphere | Price Ratio Predictor",
    layout="wide"
)

st.title("📱 MobileSphere – Phone Price Ratio Predictor")

# -----------------------------
# LOAD DATA
# -----------------------------
data_path = r"E:\BA BI\Project BBC\project1\MobileSphere\Phone_Sales_Dataset.xlsx"
df = pd.read_excel(data_path)

brand_map = {
    b.lower(): i
    for i, b in enumerate(
        df["brand_name"].astype("category").cat.categories
    )
}

# -----------------------------
# LOAD MODEL
# -----------------------------
model_path = r"E:\BA BI\Project BBC\project1\MobileSphere\phone_sales_model.pkl"
with open(model_path, "rb") as file:
    Model = pickle.load(file)

st.success("Model loaded successfully!")

# ======================================================
# TABS
# ======================================================
tab1, tab2 = st.tabs([
    "🧪 Manual Test",
    "📦 Bulk Scanner"
])



# ======================================================
# 🧪 MANUAL TEST TAB (YOUR EXISTING CODE)
# ======================================================
with tab1:
    st.subheader("🧪 Manual Price Ratio Prediction")

    st.markdown("### 📱 Basic Info")
    brand_name = st.selectbox(
        "Brand Name",
        sorted(df["brand_name"].unique())
    )

    popularity = st.slider("Popularity", 0, 3000, 50)
    sellers_amount = st.number_input("Sellers Amount", min_value=0)

    st.markdown("### ⚙️ Technical Specs")
    screen_size = st.slider("Screen Size (inches)", 1.4, 8.1, 5.5)
    memory_size = st.selectbox(
        "Memory Size (GB)",
        [0.0032, 0.004, 0.016, 0.032, 0.064, 0.128, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    )
    battery_size = st.selectbox(
        "Battery Size (mAh)",
        sorted(df["battery_size"].dropna().astype(int).unique())
    )

    brand_encoded = df["brand_name"].astype("category").cat.categories.get_loc(brand_name)

    input_df = pd.DataFrame([{
        "brand_name": brand_encoded,
        "popularity": popularity,
        "sellers_amount": sellers_amount,
        "screen_size": screen_size,
        "memory_size": memory_size,
        "battery_size": battery_size
    }])

    if st.button("🚀 Predict Price Ratio"):
        input_df = input_df[Model.feature_names_in_]
        prediction = Model.predict(input_df)

        st.metric(
            "📈 Predicted Price Ratio",
            round(prediction[0], 3)
        )

# ======================================================
# 📦 BULK SCANNER TAB
# ======================================================
with tab2:
    st.subheader("📦 Bulk Scanner")

    # -----------------------------
    # DOWNLOAD SAMPLE FILE
    # -----------------------------
    st.markdown("### ⬇️ Download Sample File")

    sample_df = pd.DataFrame({
    "brand_name": [
        "Samsung", "Apple", "Xiaomi", "OnePlus", "Realme",
        "Oppo", "Vivo", "Motorola", "Nokia", "Infinix"
    ],
    "popularity": [
        1200, 2500, 1800, 1600, 1400,
        1300, 1250, 900, 700, 850
    ],
    "sellers_amount": [
        40, 120, 85, 60, 55,
        50, 48, 30, 25, 35
    ],
    "screen_size": [
        6.5, 6.1, 6.67, 6.55, 6.6,
        6.43, 6.44, 6.5, 6.3, 6.78
    ],
    "memory_size": [
        128, 256, 128, 256, 128,
        128, 256, 128, 64, 128
    ],
    "battery_size": [
        4500, 3200, 5000, 4800, 5000,
        4500, 4600, 5000, 3500, 6000
    ]
    })
    

    sample_format = st.selectbox(
    "Sample Format",
    ["CSV", "Excel", "JSON", "SQL"]
    )

    # -----------------------------
    # CSV SAMPLE
    # -----------------------------
    if sample_format == "CSV":
        st.download_button(
            "Download Sample CSV",
            sample_df.to_csv(index=False),
            "sample_bulk.csv",
            "text/csv"
        )

    # -----------------------------
    # EXCEL SAMPLE
    # -----------------------------
    elif sample_format == "Excel":
        buffer = io.BytesIO()
        sample_df.to_excel(buffer, index=False)
        st.download_button(
            "Download Sample Excel",
            buffer.getvalue(),
            "sample_bulk.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # -----------------------------
    # JSON SAMPLE
    # -----------------------------
    elif sample_format == "JSON":
        st.download_button(
            "Download Sample JSON",
            sample_df.to_json(orient="records"),
            "sample_bulk.json",
            "application/json"
        )

    # -----------------------------
    # SQL SAMPLE (NEW 🔥)
    # -----------------------------
    elif sample_format == "SQL":

        sql_script = """
        CREATE TABLE phone_data (
            brand_name TEXT,
            popularity INTEGER,
            sellers_amount INTEGER,
            screen_size REAL,
            memory_size INTEGER,
            battery_size INTEGER
        );

        INSERT INTO phone_data VALUES
        ('Samsung',1200,40,6.5,128,4500),
        ('Apple',2500,120,6.1,256,3200),
        ('Xiaomi',1800,85,6.67,128,5000),
        ('OnePlus',1600,60,6.55,256,4800),
        ('realme',1400,55,6.6,128,5000),
        ('OPPO',1300,50,6.43,128,4500),
        ('vivo',1250,48,6.44,256,4600),
        ('Motorola',900,30,6.5,128,5000),
        ('Nokia',700,25,6.3,64,3500),
        ('BlackBerry',850,35,6.78,128,6000);
        """

        st.download_button(
            "Download Sample SQL",
            sql_script,
            "sample_bulk.sql",
            "application/sql"
        )

        st.markdown("---")
    # -----------------------------
    # UPLOAD FILE
    # -----------------------------
    uploaded_file = st.file_uploader(
        "📤 Upload File to Scan (CSV / Excel / JSON / SQL)",
        type=["csv", "xlsx", "json", "sql"]
    )

    # -----------------------------
    # PROCESS FILE
    # -----------------------------
    if uploaded_file is not None:

        # CSV
        if uploaded_file.name.endswith(".csv"):
            bulk_df = pd.read_csv(uploaded_file)

        # Excel
        elif uploaded_file.name.endswith(".xlsx"):
            bulk_df = pd.read_excel(uploaded_file)

        # JSON
        elif uploaded_file.name.endswith(".json"):
            bulk_df = pd.read_json(uploaded_file)

        # SQL
        elif uploaded_file.name.endswith(".sql"):
            import sqlite3
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()

            sql_script = uploaded_file.read().decode("utf-8")
            cursor.executescript(sql_script)

            bulk_df = pd.read_sql("SELECT * FROM phone_data", conn)
            conn.close()

        st.success("File uploaded successfully!")
        st.dataframe(bulk_df.head())



    if uploaded_file is not None:

        # -----------------------------
        # ENCODE BRAND (SAFE & CLEAN)
        # -----------------------------
        brand_categories = df["brand_name"].astype("category").cat.categories
        brand_map = {b.lower(): i for i, b in enumerate(brand_categories)}

        if not pd.api.types.is_numeric_dtype(bulk_df["brand_name"]):
            bulk_df["brand_name"] = bulk_df["brand_name"].astype(str).str.lower()

            invalid_brands = set(bulk_df["brand_name"]) - set(brand_map.keys())
            if invalid_brands:
                st.error(f"❌ Invalid brand(s) found: {', '.join(invalid_brands)}")
                st.stop()

            bulk_df["brand_name"] = bulk_df["brand_name"].map(brand_map)

        # -----------------------------
        # RUN PREDICTION
        # -----------------------------
        if st.button("🚀 Run Bulk Prediction", key="bulk_predict"):
            bulk_df_model = bulk_df[Model.feature_names_in_]
            bulk_df["predicted_price_ratio"] = Model.predict(bulk_df_model).round(3)

            st.session_state["bulk_result"] = bulk_df
            st.success("✅ Bulk prediction completed!")


        # -----------------------------
# SHOW RESULT + DOWNLOAD
# -----------------------------
if "bulk_result" in st.session_state:
    result_df = st.session_state["bulk_result"]

    st.dataframe(result_df.head())
    st.markdown("### ⬇️ Download Results")

    output_format = st.selectbox(
        "Output Format",
        ["CSV", "Excel", "JSON", "SQL"],
        key="download_format"
    )

    # -----------------------------
    # CSV
    # -----------------------------
    if output_format == "CSV":
        st.download_button(
            "Download CSV",
            result_df.to_csv(index=False),
            "bulk_predictions.csv",
            "text/csv"
        )

    # -----------------------------
    # EXCEL
    # -----------------------------
    elif output_format == "Excel":
        buffer = io.BytesIO()
        result_df.to_excel(buffer, index=False)
        st.download_button(
            "Download Excel",
            buffer.getvalue(),
            "bulk_predictions.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # -----------------------------
    # JSON
    # -----------------------------
    elif output_format == "JSON":
        json_data = result_df.to_json(orient="records", indent=4)
        st.download_button(
            "Download JSON",
            json_data,
            "bulk_predictions.json",
            "application/json"
        )

    # -----------------------------
    # SQL
    # -----------------------------
    elif output_format == "SQL":
        table_name = "phone_price_predictions"
        sql_script = f"CREATE TABLE {table_name} (\n"

        # Create table columns
        for col, dtype in result_df.dtypes.items():
            if "int" in str(dtype):
                sql_type = "INTEGER"
            elif "float" in str(dtype):
                sql_type = "REAL"
            else:
                sql_type = "TEXT"
            sql_script += f"    {col} {sql_type},\n"

        sql_script = sql_script.rstrip(",\n") + "\n);\n\n"

        # Insert data
        for _, row in result_df.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append("NULL")
                elif isinstance(val, str):
                    values.append(f"'{val}'")
                else:
                    values.append(str(val))
            sql_script += f"INSERT INTO {table_name} VALUES ({', '.join(values)});\n"

        st.download_button(
            "Download SQL",
            sql_script,
            "bulk_predictions.sql",
            "application/sql"
        )