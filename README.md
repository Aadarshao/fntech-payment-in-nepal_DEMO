# fntech-payment-in-nepal

**Author:** Aadarsh — 

## About
Nepal is rapidly moving into the digital payments era. As of mid‑January 2025, mobile banking penetration soared to **89%**, with **26.5 million mobile banking accounts**, alongside **25.8 million wallet accounts** and **13.4 million debit card accounts**. Mobile banking captures over **70% of transactions by count** but only about **6.6% of total transaction value**—highlighting frequent low‑value payments in Nepal's digital ecosystem.

This project explores a synthetic Nepal‑calibrated payments dataset (3 months, 20,000 rows). It demonstrates EDA, RFM segmentation, anomaly detection, and an interactive Streamlit dashboard.

## Files
- `data/Fintech-data-for-nepal.csv` — transaction data (20k rows)
- `src/` — data processing and feature engineering
- `notebooks/` — Jupyter notebooks for EDA, RFM, modeling
- `streamlit_app.py` — interactive dashboard
- `reports/` — exportable summaries

## Quick start (local)
```bash
# create and activate venv (macOS / Linux)
python3 -m venv venv
source venv/bin/activate

# or Windows PowerShell
# python -m venv venv
# .\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# run the Streamlit dashboard
streamlit run streamlit_app.py
```

## Next steps
- Replace dataset with real anonymized data if available
- Add SQL / database backend for larger data
- Add unit tests for preprocessing functions
