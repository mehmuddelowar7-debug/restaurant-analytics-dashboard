import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# ===============================
# Load model and dataset
# ===============================
model = joblib.load('random_forest_order_total.pkl')
data = pd.read_csv('order_history_kaggle_data.csv')

# ===============================
# Preprocessing
# ===============================
data['order_placed_datetime'] = pd.to_datetime(data['Order Placed At'], errors='coerce')

def extract_discount(value):
    try:
        if pd.isna(value):
            return 0
        return float(str(value).replace('₹','').replace('%','').split()[0])
    except:
        return 0

data['total_discount'] = data['Discount construct'].apply(extract_discount)

# Dummy location coordinates
np.random.seed(42)
data['lat'] = 28.6 + np.random.randn(len(data))*0.05
data['lon'] = 77.2 + np.random.randn(len(data))*0.05

# ===============================
# App configuration
# ===============================
st.set_page_config(page_title="Restaurant Dashboard", layout="wide")

# ===============================
# CSS Styling
# ===============================
st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1 {
            color: #FF5A1F;  /* Swiggy orange */
        }
        h2, h3, h4, h5, h6, p, label, div, .stText, .stMarkdown {
            color: #ffffff !important;
        }
        .stButton>button {
            background-color: #FF5A1F;
            color: #ffffff;
            border-radius: 8px;
        }
        .stMetric>div>div {
            background-color: #2c2c2c;
            padding: 12px;
            border-radius: 12px;
        }
        .filter-card {
            background-color: #2c2c2c;
            padding: 18px 15px;
            border-radius: 12px;
            margin-bottom: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .filter-card:hover {
            transform: translateY(-3px);
            box-shadow: 0px 6px 15px rgba(0,0,0,0.4);
        }
        .stTextInput>div>div>input, 
        .stSelectbox>div>div>select, 
        .stNumberInput>div>div>input {
            color: #000000;
            background-color: #ffffff;
            border-radius: 6px;
            padding: 4px 8px;
        }
        .sidebar .sidebar-content {
            background-color: #1f1f1f;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# Dashboard Header
# ===============================
st.markdown(f"<h1 style='text-align:center;'>🍽️ Restaurant Dashboard & Order Prediction</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;'> Designed to simulate how analytics teams support pricing, operations, and growth decisions in food delivery platforms.</p> ", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;'>Interactive professional dashboard with analytics, map, and AI predictions</p>", unsafe_allow_html=True)
st.markdown("""
### 🎯 Business Context & Objective

This dashboard is designed as a **decision-support tool for food delivery operations**.
The objective is to help stakeholders **monitor revenue performance, understand customer behavior, 
and evaluate operational factors that influence order value**.

As an MBA (Final Year) student specializing in analytics, this project demonstrates how 
**data-driven insights and machine learning can support pricing, delivery efficiency, 
and restaurant performance evaluation** in a real-world food delivery ecosystem.
""")

# ===============================
# Sidebar Filters
# ===============================
st.sidebar.markdown("## 🔎 Filters & Prediction Inputs")

# City Filter
st.sidebar.markdown('<div class="filter-card">', unsafe_allow_html=True)
st.sidebar.markdown("### 🌆 Select City")
city_filter = st.sidebar.multiselect(
    "Choose Cities", 
    options=sorted(data['City'].dropna().unique()), 
    default=sorted(data['City'].dropna().unique())
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Delivery Type Filter
st.sidebar.markdown('<div class="filter-card">', unsafe_allow_html=True)
st.sidebar.markdown("### 🚚 Delivery Type")
delivery_filter = st.sidebar.multiselect(
    "Select Delivery Types", 
    options=sorted(data['Delivery'].dropna().unique()), 
    default=sorted(data['Delivery'].dropna().unique())
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Date Range Filter
st.sidebar.markdown('<div class="filter-card">', unsafe_allow_html=True)
st.sidebar.markdown("### 📅 Date Range")
min_date = data['order_placed_datetime'].min().date()
max_date = data['order_placed_datetime'].max().date()

date_range_input = st.sidebar.date_input(
    "Select Date Range", 
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_range_input, (list, tuple)) and len(date_range_input) == 2:
    start_date, end_date = date_range_input
else:
    start_date = end_date = date_range_input

# Handle single date vs range safely
if isinstance(date_range_input, (list, tuple)):
    if len(date_range_input) == 2:
        start_date, end_date = date_range_input
    else:
        start_date = end_date = date_range_input[0]
else:
    start_date = end_date = date_range_input
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Predict Order Total Inputs
st.sidebar.markdown('<div class="filter-card">', unsafe_allow_html=True)
st.sidebar.markdown("### 🤖 Predict Order Total")
num_items = st.sidebar.number_input("Number of items", min_value=1, value=1)
total_discount_input = st.sidebar.number_input("Total discount (₹)", min_value=0.0, value=0.0)
kpt_duration_minutes = st.sidebar.number_input("KPT duration (minutes)", min_value=0, value=30)
rider_wait_time_minutes = st.sidebar.number_input("Rider wait time (minutes)", min_value=0, value=5)
delivery_delay = st.sidebar.number_input("Delivery delay (minutes)", min_value=0, value=0)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Subzone Filter (if exists)
# ===============================
subzone_filter = None  # define it by default

if 'Subzone' in data.columns and not data['Subzone'].dropna().empty:
    st.sidebar.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.sidebar.markdown("### 📌 Subzone Filter")
    subzone_filter = st.sidebar.multiselect(
        "Select Subzone", 
        options=sorted(data['Subzone'].dropna().unique()), 
        default=sorted(data['Subzone'].dropna().unique())
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Apply Filters
# ===============================
filtered_data = data[
    (data['City'].isin(city_filter)) &
    (data['Delivery'].isin(delivery_filter)) &
    (data['order_placed_datetime'].dt.date >= start_date) &
    (data['order_placed_datetime'].dt.date <= end_date)
]

# Apply Subzone filter if it exists and user selected values
if subzone_filter:
    filtered_data = filtered_data[filtered_data['Subzone'].isin(subzone_filter)]
st.markdown("""
### 📌 Data Scope & Assumptions

- The analysis is based on **historical order-level data** sourced from a structured dataset.
- Geographic coordinates are **synthetically generated** to demonstrate spatial analysis.
- All monetary values are shown in **Indian Rupees (₹)**.
- Insights and predictions are intended for **analytical demonstration** and not live operational deployment.

This approach reflects common practices in **exploratory business analytics and academic projects**.
""")

st.markdown(""" Operational time variables used in predictive inputs are modeled features derived for analytical demonstration and may not represent exact system-tracked timestamps.
""")

# ===============================
# KPI Metrics
# ===============================
st.markdown("### 📈 Key Performance Indicators")
total_revenue = filtered_data['Total'].sum()
avg_order_total = filtered_data['Total'].mean()
total_orders = filtered_data.shape[0]
avg_discount = filtered_data['total_discount'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue (₹)", f"{total_revenue:,.2f}")
col2.metric("Average Order Total (₹)", f"{avg_order_total:,.2f}")
col3.metric("Total Orders", total_orders)
col4.metric("Average Discount (₹)", f"{avg_discount:.2f}")
st.markdown("""

📊 **How to Read These KPIs**

- **Total Revenue** indicates overall business volume within the selected filters.
- **Average Order Total** helps assess pricing effectiveness and customer spend behavior.
- **Total Orders** reflects demand intensity.
- **Average Discount** highlights promotional dependency and margin impact.

Together, these metrics provide a **balanced view of growth, demand, and profitability drivers**.
""")

# ===============================

st.markdown("""
---
### 🎯 Key Business Takeaways (Illustrative)

• High-revenue restaurants consistently show higher average order values rather than higher order counts, indicating pricing and menu mix effectiveness.  

• Average discounts remain relatively low compared to order value, suggesting limited margin erosion under current promotional strategies.  

• Order volume trends over time can help operations teams anticipate peak demand periods and optimize staffing and delivery capacity.  

• Subzone-level revenue differences highlight geographic demand concentration, useful for targeted promotions and rider allocation.  

• Predictive estimates enable scenario testing — for example, evaluating how increased preparation time or discounts may impact order value.
""")
# ===============================
st.markdown("""
---
### 🚀 Recommended Business Actions (Illustrative)

• Focus marketing investments on high-performing subzones to maximize ROI from demand concentration.  

• Optimize menu pricing and bundling for restaurants with high average order value rather than increasing discount depth.  

• Use order volume trends to proactively plan rider allocation and kitchen staffing during peak demand periods.  

• Monitor preparation time sensitivity in predictive estimates to identify kitchens where operational efficiency improvements can directly increase revenue.  

• Apply scenario-based predictions to evaluate the financial impact of promotional strategies before execution.
""")

# ===============================
# Dashboard Insights (Charts & Map)
# ===============================
st.markdown("### 📊 Dashboard Insights")

# Top 10 Restaurants by Revenue
revenue_by_restaurant = filtered_data.groupby('Restaurant name')['Total'].sum().sort_values(ascending=False).reset_index()
fig1 = px.bar(
    revenue_by_restaurant.head(10),
    x='Restaurant name', y='Total',
    text='Total',
    hover_data={'Total':':.2f'},
    labels={'Total':'Revenue (₹)','Restaurant name':'Restaurant'},
    title="Top 10 Restaurants by Revenue",
    color='Total',
    color_continuous_scale=px.colors.sequential.Tealgrn
)
fig1.update_traces(texttemplate='₹%{text:.2f}', textposition='outside')
fig1.update_layout(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='white')
col1_charts, col2_charts = st.columns(2)
col1_charts.plotly_chart(fig1, use_container_width=True)

# Average Ratings by City
ratings_by_city = filtered_data.groupby('City')['Rating'].mean().reset_index()
fig2 = px.bar(
    ratings_by_city,
    x='City', y='Rating',
    color='Rating',
    color_continuous_scale=px.colors.sequential.Tealgrn,
    hover_data={'Rating':':.2f'},
    title="Average Ratings by City"
)
fig2.update_layout(yaxis_range=[0,5], plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='white')
col2_charts.plotly_chart(fig2, use_container_width=True)

# Orders Over Time
order_trend = filtered_data.groupby(filtered_data['order_placed_datetime'].dt.date).size().reset_index(name='Order Count')
fig3 = px.line(order_trend, x='order_placed_datetime', y='Order Count', markers=True, title="Orders Over Time")
fig3.update_traces(line_color='#FF5A1F', marker_size=10, hovertemplate='Date: %{x}<br>Orders: %{y}<extra></extra>')
fig3.update_layout(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='white')
st.plotly_chart(fig3, use_container_width=True)

# Animated Map
fig_map = px.scatter_mapbox(
    filtered_data,
    lat="lat", lon="lon",
    hover_name="Restaurant name",
    hover_data=["Total","Rating","City","Subzone"] if 'Subzone' in data.columns else ["Total","Rating","City"],
    color="Total",
    size="Total",
    animation_frame=filtered_data['order_placed_datetime'].dt.date.astype(str),
    color_continuous_scale=px.colors.sequential.Tealgrn,
    zoom=10,
    size_max=15
)
fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# Revenue by Subzone Chart (Professional)
if 'Subzone' in filtered_data.columns and not filtered_data.empty:
    subzone_revenue = filtered_data.groupby('Subzone')['Total'].sum().reset_index()
    fig_subzone = px.bar(
        subzone_revenue,
        x='Subzone', y='Total',
        text='Total',
        title="Revenue by Subzone",
        color='Total',
        color_continuous_scale=px.colors.sequential.Tealgrn,
        labels={'Total':'Revenue (₹)', 'Subzone':'Subzone'}
    )
    fig_subzone.update_traces(texttemplate='₹%{text:.2f}', textposition='outside')
    fig_subzone.update_layout(
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font_color='white',
        xaxis_title="Subzone",
        yaxis_title="Revenue (₹)",
        margin=dict(t=50, l=50, r=50, b=50)
    )
    st.plotly_chart(fig_subzone, use_container_width=True)

# ===============================
# Restaurant Insights
# ===============================
st.markdown("### 🏷️ Restaurant Insights")
restaurant_selected = st.selectbox("Choose a restaurant", options=filtered_data['Restaurant name'].unique())
restaurant_data = filtered_data[filtered_data['Restaurant name'] == restaurant_selected]

rest_rev_trend = restaurant_data.groupby(restaurant_data['order_placed_datetime'].dt.date)['Total'].sum().reset_index()
fig_rest = px.line(rest_rev_trend, x='order_placed_datetime', y='Total', markers=True,
                   title=f"Revenue Trend - {restaurant_selected}")
fig_rest.update_traces(line_color='#FF5A1F', marker_size=10, hovertemplate='Date: %{x}<br>Revenue: ₹%{y:.2f}<extra></extra>')
fig_rest.update_layout(plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a', font_color='white')

col1_rest, col2_rest = st.columns(2)
col1_rest.metric(f"Average Rating for {restaurant_selected}", f"{restaurant_data['Rating'].mean():.2f}")
col2_rest.metric(f"Total Orders for {restaurant_selected}", restaurant_data.shape[0])
st.plotly_chart(fig_rest, use_container_width=True)
st.markdown("""
### 🤖 Predictive Analytics Overview

A **Random Forest regression model** is used to estimate the expected order value
based on operational and order-related inputs such as item count, discount, 
kitchen preparation time, rider wait time, and delivery delay.

The model captures **non-linear relationships** commonly observed in food delivery operations 
and is intended to support **scenario analysis and revenue estimation**, 
rather than exact financial forecasting.
""")

# ===============================
# Predict Order Total
# ===============================
st.markdown("### 🤖 Predict Order Total")
predict_btn = st.button("🔮 Predict Order Value")

if predict_btn:
    features = pd.DataFrame([{
        'num_items': num_items,
        'total_discount': total_discount_input,
        'kpt_duration_minutes': kpt_duration_minutes,
        'rider_wait_time_minutes': rider_wait_time_minutes,
        'delivery_delay': delivery_delay
    }])

    predicted_total = model.predict(features)[0]

    st.success(f"Predicted Order Total: ₹{predicted_total:.2f}")
st.caption("""
⚠️ **Model Disclaimer**  
Predictions are based on historical patterns and may vary under changing operational conditions. 
The output should be used as a **supporting estimate**, not a final decision metric.
""")

# ===============================
# Mini AI Assistant (Professional Cards)
# ===============================
st.markdown("### 💬 Mini AI Assistant")
user_query = st.text_input("Ask a question (e.g., 'Show me top 5 restaurants in Delhi last month')")

if user_query:
    import re

    # Default top N
    try:
        top_n = int(re.search(r'top (\d+)', user_query.lower()).group(1))
    except:
        top_n = 5

    # City filter based on user query
    city_match = [c for c in data['City'].unique() if c.lower() in user_query.lower()]
    city_name = city_match[0] if city_match else None

    # Time filter
    date_start = filtered_data['order_placed_datetime'].min()
    date_end = filtered_data['order_placed_datetime'].max()

    if 'last month' in user_query.lower():
        today = datetime.today()
        first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last_day_last_month = today.replace(day=1) - timedelta(days=1)
        date_start, date_end = first_day_last_month, last_day_last_month
    elif 'last week' in user_query.lower():
        today = datetime.today()
        date_start = today - timedelta(days=7)
        date_end = today

    # Filter dataset
    ai_filtered = filtered_data.copy()
    if city_name:
        ai_filtered = ai_filtered[ai_filtered['City'] == city_name]
    ai_filtered = ai_filtered[(ai_filtered['order_placed_datetime'].dt.date >= date_start.date()) &
                              (ai_filtered['order_placed_datetime'].dt.date <= date_end.date())]

    if ai_filtered.empty:
        st.info("No data available for this query. Try another city or date range.")
    else:
        # Prepare data for cards
        top_rev = ai_filtered.groupby('Restaurant name')['Total'].sum().sort_values(ascending=False).head(top_n).reset_index()
        top_rating = ai_filtered.groupby('Restaurant name')['Rating'].mean().sort_values(ascending=False).head(top_n).reset_index()
        top_orders = ai_filtered.groupby('Restaurant name').size().sort_values(ascending=False).head(top_n).reset_index(name='Total Orders')

        st.markdown(f"**Top {top_n} Restaurants Insights {f'in {city_name}' if city_name else ''}**")

        # Display cards
        for i in range(len(top_rev)):
            st.markdown(f"""
                <div style="background-color:#2c2c2c; padding:15px; border-radius:12px; margin-bottom:10px;">
                    <h4 style="margin:0; color:#FF5A1F;">{top_rev.loc[i,'Restaurant name']}</h4>
                    <p style="margin:0; color:#ffffff;">Revenue: ₹{top_rev.loc[i,'Total']:.2f}</p>
                    <p style="margin:0; color:#ffffff;">Avg Rating: {top_rating.loc[i,'Rating']:.2f} ⭐</p>
                    <p style="margin:0; color:#ffffff;">Total Orders: {top_orders.loc[i,'Total Orders']}</p>
                </div>
            """, unsafe_allow_html=True)

# ===============================
# Dataset Preview
# ===============================
st.markdown("### 📋 Preview Filtered Data")
st.dataframe(filtered_data.head(20), use_container_width=True)

st.markdown("""
---
### 👤 Project Context & Ownership

This dashboard was independently designed and developed as a 
Business Analytics portfolio project during my MBA (Final Year).

The objective was to demonstrate how transactional food delivery data 
can be transformed into decision-support insights for restaurant operations.

Key analytical focus areas include:

• Revenue and order volume analysis  
• Discount impact on order value  
• Restaurant and city-level performance comparison  
• Subzone-wise revenue distribution  
• Predictive modeling to estimate order value from order parameters  

The project focuses on analytical reasoning and business interpretation 
rather than production-grade system implementation.
""")
