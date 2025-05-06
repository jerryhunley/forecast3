
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

def main():
    st.title("Clinical Trial Recruitment Forecasting Tool")

    uploaded_referral = st.file_uploader("📁 Upload Referral Data", type=["csv"])
    uploaded_sites = st.file_uploader("📁 Upload Site List", type=["csv"])

    if uploaded_referral and uploaded_sites:
        referrals = pd.read_csv(uploaded_referral)

        # Flexible date parsing
        if "Referral Date" in referrals.columns:
            referrals['Referral Date'] = pd.to_datetime(referrals['Referral Date'], errors='coerce')
        elif "Submission Date" in referrals.columns:
            referrals.rename(columns={'Submission Date': 'Referral Date'}, inplace=True)
            referrals['Referral Date'] = pd.to_datetime(referrals['Referral Date'], errors='coerce')
        else:
            st.error("Referral data must include either 'Referral Date' or 'Submission Date' column.")
            return

        sites = pd.read_csv(uploaded_sites)
        referrals['Signed ICF'] = referrals['Lead Stage History'].fillna('').str.contains('Signed ICF').astype(int)

        
        if referrals.empty:
            st.info("Not enough recent data to display conversion trends.")
        else:
            st.subheader("📈 Time to Contact + Recent Conversion Trends")
# Standardize site column
        site_col = None
        for col in referrals.columns:
            if 'site' in col.lower() and 'number' in col.lower():
                site_col = col
                break
        if not site_col:
            st.warning("Could not find a column for 'Site Number'.")
            return
        
        

        recent_window = st.slider("Recent Window (days)", 7, 90, 30)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_window)
        recent = referrals[referrals['Referral Date'] >= cutoff]

        if not recent.empty:
            
if site_col in recent.columns and not recent.empty:
    site_stats = recent.groupby(site_col).agg(
        Referrals=('Referral Date', 'count'),
        ICFs=('Signed ICF', 'sum')
    ).reset_index()
    site_stats['ICF Conversion Rate (%)'] = (site_stats['ICFs'] / site_stats['Referrals'] * 100).round(2)
    st.markdown("### 🔍 Recent Conversion Rates by Site")
    st.dataframe(site_stats, use_container_width=True)
else:
    st.info("No recent data available or 'Site Number' column not found.")
    main()
