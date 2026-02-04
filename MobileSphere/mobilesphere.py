import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Phone Price Ratio Predictor")
st.title("Phone Price Ratio Prediction")
st.write("Enter mobile specifications to predict price ratio")

# -----------------------------
# LOAD DATA (FOR DROPDOWNS)
# -----------------------------
data_path = r"E:\BA BI\Project BBC\project1\Project\Phone_Sales_Dataset.xlsx"
df = pd.read_excel(data_path)

# -----------------------------
# LOAD MODEL
# -----------------------------
model_path = r"E:\BA BI\Project BBC\project1\Project\phone_sales_model.pkl"
with open(model_path, "rb") as file:
    Model = pickle.load(file)

st.success("Model loaded successfully!")
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application predicts the **price ratio** of mobile phones based on various specifications.
    
    **How to use:**
    1. Select brand name
    2. Adjust popularity slider
    3. Enter number of sellers
    4. Set screen size
    5. Choose memory size
    6. Select battery capacity
    7. Click **Predict**
    """)
    
    st.header("📊 Dataset Info")
    if df is not None:
        st.metric("Total Records", len(df))
        st.metric("Total Brands", df["brand_name"].nunique())
    
    st.header("🔧 Model Info")
    st.write(f"**Model Type:** {type(Model).__name__}")

# -----------------------------
# USER INPUTS
# -----------------------------
st.markdown("---")
st.subheader("🔍 Enter Phone Specifications")

st.markdown("### 📱 Basic Info")
brand_name = st.selectbox("Brand Name", sorted(df["brand_name"].unique()))
popularity = st.slider("Popularity", 0, 3000, 50)
sellers_amount = st.number_input("Sellers Amount", min_value=0)
    

st.markdown("### ⚙️ Technical Specs")
screen_size = st.slider("Screen Size (inches)", 1.4, 8.1, 5.5)
memory_size = st.selectbox("Select Memory Size in GB",[0.0032, 0.004, 0.016, 0.032, 0.064, 0.128, 4, 8, 16, 32, 64, 128, 256, 512, 1000])
battery_size = st.selectbox("Battery Size (mAh)",sorted(df["battery_size"].dropna().astype(int).unique()))

# -----------------------------
# ENCODING (MUST MATCH TRAINING)
# -----------------------------
brand_encoded = df["brand_name"].astype("category").cat.categories.get_loc(brand_name)

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------
input_df = pd.DataFrame([{
    "brand_name": brand_encoded,
 
    "popularity": popularity,
    "sellers_amount": sellers_amount,
    "screen_size": screen_size,
    "memory_size": memory_size,
    "battery_size": battery_size
}])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Price Ratio"):

    # Enforce correct feature order
    input_df = input_df[Model.feature_names_in_]

    prediction = Model.predict(input_df)

    st.metric(
        label="📈 Predicted Price Ratio",
        value=round(prediction[0], 3)
    )