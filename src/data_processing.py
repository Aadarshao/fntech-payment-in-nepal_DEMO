import pandas as pd
import numpy as np

def load_data(path="data/Fintech-data-for-nepal.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT
    # amount numeric
    amt_cols = [c for c in df.columns if 'amount' in c.lower() or 'amt' in c.lower()]
    if amt_cols:
        df['amount'] = pd.to_numeric(df[amt_cols[0]], errors='coerce')
    else:
        df['amount'] = np.nan
    return df

def clean_basic(df, drop_negative=True):
    df = df.drop_duplicates()
    required = ['transaction_id','user_id','timestamp','amount']
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)
    if drop_negative:
        df = df[df['amount'] >= 0]
    # features
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5,6])
    return df

def daily_summary(df):
    return df.groupby('date').agg(
        total_amount=('amount','sum'),
        avg_amount=('amount','mean'),
        tx_count=('transaction_id','count')
    ).reset_index()

def category_channel_breakdown(df):
    cat = df.groupby('category').agg(
        total_amount=('amount','sum'),
        tx_count=('transaction_id','count')
    ).reset_index().sort_values('total_amount', ascending=False)
    chan = df.groupby('channel').agg(
        total_amount=('amount','sum'),
        tx_count=('transaction_id','count')
    ).reset_index().sort_values('total_amount', ascending=False)
    return cat, chan

def detect_daily_spikes(df, z_thresh=2.0):
    daily = daily_summary(df)
    mean = daily['total_amount'].mean()
    std = daily['total_amount'].std()
    daily['is_spike'] = daily['total_amount'] > (mean + z_thresh * std)
    return daily
