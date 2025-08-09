import pandas as pd

def calc_rfm(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df['timestamp'].max() + pd.Timedelta(days=1)
    agg = df.groupby('user_id').agg(
        recency_days = ('timestamp', lambda x: (snapshot_date - x.max()).days),
        frequency = ('transaction_id','count'),
        monetary = ('amount','sum')
    ).reset_index()
    return agg

def add_rfm_scores(rfm):
    rfm = rfm.copy()
    rfm['r_score'] = pd.qcut(rfm['recency_days'].rank(method='first'), 4, labels=[4,3,2,1]).astype(int)
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    return rfm
