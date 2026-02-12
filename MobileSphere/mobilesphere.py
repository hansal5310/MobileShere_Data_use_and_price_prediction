import streamlit as st
import pandas as pd
import pickle
import io
import sqlite3
import os
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "Phone_Sales_Dataset.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "phone_sales_model.pkl")

Model = joblib.load(MODEL_PATH)

joblib.dump(Model, "phone_sales_model.pkl")

# ---- Load dataset ----
df = pd.read_excel(DATA_PATH)



# ----------------------------- 
# PAGE CONFIG 
# ----------------------------- 
st.set_page_config(
    page_title="MobileSphere | Price Ratio Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- 
# CUSTOM CSS FOR ATTRACTIVE UI
# ----------------------------- 
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content area styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    h1 {
        color: #667eea;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #764ba2;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 600;
        color: #667eea;
        border: 2px solid #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        width: 100%;
        padding: 0.5rem 1rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
        font-weight: 700;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #f8f9fa;
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------- 
# HEADER
# ----------------------------- 
st.markdown("""
<h1>📱 MobileSphere – Phone Price Ratio Predictor</h1>
<p style='text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>
    Predict phone price ratios with AI-powered machine learning
</p>
""", unsafe_allow_html=True)

# ----------------------------- 
# LOAD DATA 
# ----------------------------- 


brand_map = {
    b.lower(): i for i, b in enumerate(
        df["brand_name"].astype("category").cat.categories
    )
}

# ----------------------------- 
# LOAD MODEL 
# ----------------------------- 



# ====================================================== 
# TABS 
# ====================================================== 
tab1, tab2 = st.tabs([
    "🧪 Manual Test",
    "📦 Bulk Scanner"
])

# ====================================================== 
# 🧪 MANUAL TEST TAB
# ====================================================== 
with tab1:
    st.markdown("""<div class='card'><h3>🧪 Manual Price Ratio Prediction</h3>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📱 Basic Info")
        brand_name = st.selectbox(
            "Brand Name",
            sorted(df["brand_name"].unique())
        )
        popularity = st.slider("Popularity", 0, 3000, 50)
        sellers_amount = st.number_input("Sellers Amount", min_value=0)
    
    with col2:
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Predict Price Ratio", use_container_width=True):
            input_df = input_df[Model.feature_names_in_]
            prediction = Model.predict(input_df)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
            with col_m2:
                st.metric(
                    "📈 Predicted Price Ratio",
                    round(prediction[0], 3)
                )
            st.markdown("</div>", unsafe_allow_html=True)

# ====================================================== 
# 📦 BULK SCANNER TAB 
# ====================================================== 
with tab2:
    st.subheader("📦 Bulk Price Ratio Scanner")
    
    # Sample data
    sample_df = pd.DataFrame({
        "brand_name": [
            "Samsung", "Apple", "Xiaomi", "OnePlus", "Realme",
            "Oppo", "Vivo", "Motorola", "Nokia", "Sony"
        ],
        "popularity": [
            1200, 2500, 1800, 1600, 1400, 1300, 1250, 900, 700, 850
        ],
        "sellers_amount": [
            40, 120, 85, 60, 55, 50, 48, 30, 25, 35
        ],
        "screen_size": [
            6.5, 6.1, 6.67, 6.55, 6.6, 6.43, 6.44, 6.5, 6.3, 6.78
        ],
        "memory_size": [
            128, 256, 128, 256, 128, 128, 256, 128, 64, 128
        ],
        "battery_size": [
            4500, 3200, 5000, 4800, 5000, 4500, 4600, 5000, 3500, 6000
        ]
    })
    
    # ----------------------------- 
    # THREE COLUMN LAYOUT
    # ----------------------------- 
    col_download, col_upload, col_result = st.columns(3)
    
    # ----------------------------- 
    # COLUMN 1: DOWNLOAD SAMPLE
    # ----------------------------- 
    with col_download:
        st.markdown("""<div class='card'><h3>⬇️ Download Sample</h3>""", unsafe_allow_html=True)
        st.markdown("Get a template file to see the required format")
        
        sample_format = st.selectbox(
            "Sample Format",
            ["CSV", "Excel", "JSON", "SQL"],
            key="sample_format"
        )
        
        st.markdown("---")
        
        # CSV SAMPLE
        if sample_format == "CSV":
            st.download_button(
                "📥 Download Sample CSV",
                sample_df.to_csv(index=False),
                "sample_bulk.csv",
                "text/csv",
                use_container_width=True
            )
        
        # EXCEL SAMPLE
        elif sample_format == "Excel":
            buffer = io.BytesIO()
            sample_df.to_excel(buffer, index=False)
            st.download_button(
                "📥 Download Sample Excel",
                buffer.getvalue(),
                "sample_bulk.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # JSON SAMPLE
        elif sample_format == "JSON":
            st.download_button(
                "📥 Download Sample JSON",
                sample_df.to_json(orient="records", indent=4),
                "sample_bulk.json",
                "application/json",
                use_container_width=True
            )
        
        # SQL SAMPLE
        elif sample_format == "SQL":
            sql_script = """CREATE TABLE phone_data (
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
('Realme',1400,55,6.6,128,5000),
('Oppo',1300,50,6.43,128,4500),
('Vivo',1250,48,6.44,256,4600),
('Motorola',900,30,6.5,128,5000),
('Nokia',700,25,6.3,64,3500),
('Infinix',850,35,6.78,128,6000);
"""
            st.download_button(
                "📥 Download Sample SQL",
                sql_script,
                "sample_bulk.sql",
                "application/sql",
                use_container_width=True
            )
        
        st.markdown("---")
        st.info("💡 **Tip:** Download and modify the sample file with your data")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ----------------------------- 
    # COLUMN 2: UPLOAD FILE
    # ----------------------------- 
    with col_upload:
        st.markdown("""<div class='card'><h3>📤 Upload Your File</h3>""", unsafe_allow_html=True)
        st.markdown("Upload your data file for bulk predictions")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "json", "sql"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Process file based on type
            try:
                if uploaded_file.name.endswith(".csv"):
                    bulk_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    bulk_df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".json"):
                    bulk_df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith(".sql"):
                    conn = sqlite3.connect(":memory:")
                    cursor = conn.cursor()
                    sql_script = uploaded_file.read().decode("utf-8")
                    cursor.executescript(sql_script)
                    bulk_df = pd.read_sql("SELECT * FROM phone_data", conn)
                    conn.close()
                
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                
                st.markdown("#### 📊 Preview (First 5 rows)")
                st.dataframe(bulk_df.head(), use_container_width=True)
                
                # Store in session state
                st.session_state["uploaded_data"] = bulk_df
                
                st.markdown("---")
                
                # Encode brand names
                brand_categories = df["brand_name"].astype("category").cat.categories
                brand_map = {b.lower(): i for i, b in enumerate(brand_categories)}
                
                if not pd.api.types.is_numeric_dtype(bulk_df["brand_name"]):
                    bulk_df["brand_name"] = bulk_df["brand_name"].astype(str).str.lower()
                    invalid_brands = set(bulk_df["brand_name"]) - set(brand_map.keys())
                    
                    if invalid_brands:
                        st.error(f"❌ Invalid brand(s): {', '.join(invalid_brands)}")
                    else:
                        bulk_df["brand_name"] = bulk_df["brand_name"].map(brand_map)
                        
                        # Run prediction button
                        if st.button("🚀 Run Bulk Prediction", use_container_width=True, key="bulk_predict"):
                            bulk_df_model = bulk_df[Model.feature_names_in_]
                            bulk_df["predicted_price_ratio"] = Model.predict(bulk_df_model).round(3)
                            st.session_state["bulk_result"] = bulk_df
                            st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        else:
            st.info("📁 No file uploaded yet. Upload a file to get started!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ----------------------------- 
    # COLUMN 3: DOWNLOAD RESULTS
    # ----------------------------- 
    with col_result:
        st.markdown("""<div class='card'><h3>📊 Prediction Results</h3>""", unsafe_allow_html=True)
        
        if "bulk_result" in st.session_state:
            result_df = st.session_state["bulk_result"]
            
            st.success(f"✅ Predictions completed for {len(result_df)} records!")
            
            st.markdown("#### 📈 Results Preview")
            st.dataframe(result_df.head(), use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### ⬇️ Download Results")
            
            output_format = st.selectbox(
                "Output Format",
                ["CSV", "Excel", "JSON", "SQL"],
                key="download_format"
            )
            
            # CSV
            if output_format == "CSV":
                st.download_button(
                    "📥 Download Results (CSV)",
                    result_df.to_csv(index=False),
                    "bulk_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            # EXCEL
            elif output_format == "Excel":
                buffer = io.BytesIO()
                result_df.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Results (Excel)",
                    buffer.getvalue(),
                    "bulk_predictions.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # JSON
            elif output_format == "JSON":
                json_data = result_df.to_json(orient="records", indent=4)
                st.download_button(
                    "📥 Download Results (JSON)",
                    json_data,
                    "bulk_predictions.json",
                    "application/json",
                    use_container_width=True
                )
            
            # SQL
            elif output_format == "SQL":
                table_name = "phone_price_predictions"
                sql_script = f"CREATE TABLE {table_name} (\n"
                
                for col, dtype in result_df.dtypes.items():
                    if "int" in str(dtype):
                        sql_type = "INTEGER"
                    elif "float" in str(dtype):
                        sql_type = "REAL"
                    else:
                        sql_type = "TEXT"
                    sql_script += f"    {col} {sql_type},\n"
                
                sql_script = sql_script.rstrip(",\n") + "\n);\n\n"
                
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
                    "📥 Download Results (SQL)",
                    sql_script,
                    "bulk_predictions.sql",
                    "application/sql",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Statistics
            st.markdown("#### 📊 Quick Statistics")
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Total Records", len(result_df))
            with col_stat2:
                st.metric("Avg Price Ratio", round(result_df["predicted_price_ratio"].mean(), 3))
        
        else:
            st.info("⏳ Upload a file and run predictions to see results here")
            st.markdown("---")
            st.markdown("""
            **Steps to get results:**
            1. Download sample file
            2. Fill with your data
            3. Upload the file
            4. Click 'Run Bulk Prediction'
            5. Download results here
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- 
# FOOTER
# ----------------------------- 
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>📱 <strong>MobileSphere</strong> | Powered by Machine Learning</p>
    <p style='font-size: 0.9rem;'>© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
