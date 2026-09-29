import streamlit as st
import pandas as pd
import numpy as np
import lifetimes as lf
from lifetimes.utils import summary_data_from_transaction_data
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value (CLV) Dashboard")
df = pd.read_csv("scanner_data.csv")
st.caption("📊 Demo dataset is loaded by default. Upload your own data to customize analysis.")
st.caption("Required columns: Customer_ID, Date, Sales_Amount")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded dataset")
else:
    try:
        df = pd.read_csv("scanner_data.csv")
        st.info("📌 Using demo dataset")
    except FileNotFoundError:
        st.error("❌ Demo dataset not found. Please upload a file.")
        st.stop()

    
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
st.sidebar.header("⚙️ Controls")
profit_rate = st.sidebar.slider("Profit Rate", 0.01, 0.5, 0.15)
time_months = st.sidebar.slider("Prediction Time (Months)", 1, 24, 12)
prob_threshold = st.sidebar.slider("At Risk Threshold (Probability Alive)", 0.0, 1.0, 0.3)

  
summary_1 = summary_data_from_transaction_data(
    df,
    customer_id_col='Customer_ID',
    datetime_col='Date',
    monetary_value_col='Sales_Amount'
)
summary_1 = summary_1[(summary_1['monetary_value'] > 0) & (summary_1['frequency'] > 0)]

bgf = lf.BetaGeoFitter(penalizer_coef=0.05)
bgf.fit(summary_1['frequency'], summary_1['recency'], summary_1['T'])
summary_1['probability_alive'] = bgf.conditional_probability_alive(
    summary_1['frequency'],
    summary_1['recency'],
    summary_1['T']
)

summary_1['Customer_type'] = np.where(
    summary_1['probability_alive'] < prob_threshold,
    'At Risk',
    'Healthy'
)

   
ggf = lf.GammaGammaFitter(penalizer_coef=0.1)
ggf.fit(summary_1['frequency'], summary_1['monetary_value'])
summary_1['predicted_sale'] = ggf.customer_lifetime_value(
    bgf,
    summary_1['frequency'],
    summary_1['recency'],
    summary_1['T'],
    summary_1['monetary_value'],
    time=time_months,
    freq='D',
    discount_rate=0.01
)
summary_1['predicted_profit'] = summary_1['predicted_sale'] * profit_rate

total_revenue = summary_1['predicted_sale'].sum()
total_profit = summary_1['predicted_profit'].sum()
avg_sales = summary_1['predicted_sale'].mean()
total_customers = len(summary_1)
st.subheader("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue", round(total_revenue, 2))
col2.metric("💸 Total Profit", round(total_profit, 2))
col3.metric("📊 Avg Customer Value", round(avg_sales, 2))
col4.metric("👥 Total Customers", total_customers)

    
st.subheader("📊 Customer Segmentation")
customer_counts = summary_1['Customer_type'].value_counts()
fig1, ax1 = plt.subplots(figsize=(4,3))
ax1.bar(customer_counts.index, customer_counts.values)
ax1.set_title("Customer Type Distribution")
st.pyplot(fig1, use_container_width=False)

st.subheader("📈 Revenue vs Profit Trend")
sorted_df = summary_1.sort_values(by='predicted_sale')
fig2, ax2 = plt.subplots(figsize=(5,3))
ax2.plot(sorted_df['predicted_sale'].values, label='Predicted Sales')
ax2.plot(sorted_df['predicted_profit'].values, label='Predicted Profit')
ax2.legend()
ax2.set_title("Sales vs Profit")
st.pyplot(fig2, use_container_width=False)

st.subheader("🏆 Top 20 Customers")
top_customers = summary_1.sort_values(by='predicted_sale', ascending=False).head(20)
st.dataframe(top_customers)
st.download_button("Download Top Customers", top_customers.to_csv(), "top_customers.csv")
st.info("👆 Upload a CSV file to start")
