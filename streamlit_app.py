import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_processing import load_data, clean_basic, daily_summary, category_channel_breakdown, detect_daily_spikes
from src.features import calc_rfm, add_rfm_scores
from src.modeling import detect_transaction_anomalies

st.set_page_config(page_title="fntech-payment-in-nepal", layout="wide")
st.title("📊 fntech-payment-in-nepal — Payments Insights")
st.markdown("""**Author:** Aadarsh — aspiring Data Analyst  
This dashboard explores a Nepal‑calibrated payments dataset (20k rows).""")

@st.cache_data
def load_and_prep():
    df = load_data("data/Fintech-data-for-nepal.csv")
    df = clean_basic(df)
    return df

df = load_and_prep()

# Sidebar filters
st.sidebar.header("Filters")
min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date))
if isinstance(date_range, (list, tuple)):
    start_date, end_date = date_range[0], date_range[1]
else:
    start_date, end_date = min_date, max_date

channels = st.sidebar.multiselect("Channels", options=df['channel'].unique().tolist(), default=df['channel'].unique().tolist())
categories = st.sidebar.multiselect("Categories", options=df['category'].unique().tolist(), default=df['category'].unique().tolist())

dff = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['channel'].isin(channels)) & (df['category'].isin(categories))]

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total transactions", f"{len(dff):,}")
col2.metric("Total volume", f"${dff['amount'].sum():,.2f}")
col3.metric("Average ticket", f"${dff['amount'].mean():.2f}")
mobile_share = (dff[dff['channel']=='mobile'].shape[0] / dff.shape[0]) if dff.shape[0]>0 else 0
col4.metric("Mobile tx share", f"{mobile_share:.2%}")

# Time series
daily = daily_summary(dff)
fig = px.line(daily, x='date', y='total_amount', title='Daily total amount', markers=True)
st.plotly_chart(fig, use_container_width=True)

# Category & channel
cat, chan = category_channel_breakdown(dff)
fig_cat = px.bar(cat, x='category', y='total_amount', title='Spending by Category', text='tx_count')
st.plotly_chart(fig_cat, use_container_width=True)
fig_chan = px.bar(chan, x='channel', y='total_amount', title='Spending by Channel', text='tx_count')
st.plotly_chart(fig_chan, use_container_width=True)

# Heatmap: hour vs weekday
heat = dff.groupby(['weekday','hour']).size().reset_index(name='count')
heat_pivot = heat.pivot(index='weekday', columns='hour', values='count').fillna(0)
st.subheader('Hourly heatmap (weekday vs hour)')
st.write(heat_pivot.style.background_gradient(cmap='Blues'))

# Anomalies
st.subheader('Daily spikes')
spikes = detect_daily_spikes(dff)
st.write(spikes[spikes['is_spike']])

st.subheader('Transaction-level anomalies (IsolationForest)')
dff = detect_transaction_anomalies(dff)
st.write(dff[dff['anomaly_iforest']].sort_values('amount', ascending=False).head(20))

# RFM
st.subheader('RFM segmentation sample')
rfm = calc_rfm(df)
rfm = add_rfm_scores(rfm)
st.write(rfm.sort_values('monetary', ascending=False).head(10))
