import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# Platform configuration
st.set_page_config(page_title="Clinical Trial Recruitment Forecasting Tool", layout="wide")

def main():
    st.title("📊 Clinical Trial Recruitment Forecasting Tool")

    # File uploads
    referral_file = st.file_uploader("📥 Upload Referral Data (CSV)", type=["csv"])
    site_file = st.file_uploader("📥 Upload Site Master List (CSV)", type=["csv"])

    if referral_file and site_file:
        # Load data
        referrals = pd.read_csv(referral_file)
        sites = pd.read_csv(site_file)

        # Flexible date parsing
        if "Referral Date" in referrals.columns:
            referrals['Referral Date'] = pd.to_datetime(referrals['Referral Date'], errors='coerce')
        elif "Submission Date" in referrals.columns:
            referrals.rename(columns={'Submission Date':'Referral Date'}, inplace=True)
            referrals['Referral Date'] = pd.to_datetime(referrals['Referral Date'], errors='coerce')
        else:
            st.error("Referral data must include 'Referral Date' or 'Submission Date'.")
            return

        # Flag Signed ICF
        referrals['Signed ICF'] = referrals['Lead Stage History'].fillna('').str.contains('Signed ICF', case=False).astype(int)

        # --- Site Outreach Activity ---
        st.subheader("📈 Site Outreach Activity from Sent-to-Site Referrals")
        # Normalize text
        referrals['Lead Stage History'] = referrals['Lead Stage History'].fillna('').str.lower()
        referrals['Lead Status History'] = referrals['Lead Status History'].fillna('').str.lower()

        # Identify sent-to-site referrals
        sent_mask = referrals['Lead Stage History'].str.contains('sent to site')
        # Statuses indicating active outreach
        active_patterns = [
            r'site contact attempt',
            r'rescreen at a later date',
            r'site phone screening appt scheduled',
            r'pending medical records',
            r'appointment scheduled'
        ]
        contacted = referrals[sent_mask & referrals['Lead Status History'].str.contains('|'.join(active_patterns), na=False)]
        idle = referrals[sent_mask & referrals['Lead Status History'].str.contains(r'ready for site outreach phone screener passed', na=False)]
        total_sent = referrals[sent_mask]

        contacted_pct = (len(contacted) / len(total_sent) * 100) if len(total_sent)>0 else 0
        idle_pct = (len(idle) / len(total_sent) * 100) if len(total_sent)>0 else 0
        st.metric("📞 Contacted (%)", f"{contacted_pct:.1f}% of sent-to-site referrals")
        st.metric("⏳ Idle (%)", f"{idle_pct:.1f}% awaiting outreach")

        # --- Recent Conversion Trends by Site ---
        recent_window = st.slider("Recent Window (days)", 7, 90, 30)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_window)
        recent = referrals[referrals['Referral Date'] >= cutoff]

        # Detect site column
        site_col = None
        for c in referrals.columns:
            if 'site' in c.lower() and 'number' in c.lower():
                site_col = c
                break
        if site_col and not recent.empty:
            site_stats = recent.groupby(site_col).agg(
                Referrals=('Referral Date','count'),
                ICFs=('Signed ICF','sum')
            ).reset_index()
            site_stats['ICF Rate (%)'] = (site_stats['ICFs']/site_stats['Referrals']*100).round(1)
            st.markdown("### 🏅 Recent Conversion Performance by Site")
            st.dataframe(site_stats, use_container_width=True)
        else:
            st.info("No recent data or missing site column for conversion metrics.")

        # --- Study-Wide Forecast Summary ---
        st.header("📊 Study-Wide Forecast Summary")
        total_goal = st.number_input("Total ICF Goal", min_value=1, value=150)
        current_icfs = referrals['Signed ICF'].sum()
        remaining = total_goal - current_icfs
        st.write(f"**Signed ICFs to date:** {current_icfs}")
        st.write(f"**ICFs remaining:** {remaining}")

        # Time periods
        referrals['Month'] = referrals['Referral Date'].dt.to_period('M').dt.start_time
        referrals['Week'] = referrals['Referral Date'].dt.to_period('W').apply(lambda p: p.start_time)

        # Monthly & weekly aggregation
        monthly = referrals.groupby('Month').agg(Referrals=('Referral Date','count'), ICFs=('Signed ICF','sum')).reset_index()
        monthly['Cumulative ICFs'] = monthly['ICFs'].cumsum()
        weekly  = referrals.groupby('Week').agg(Referrals=('Referral Date','count'), ICFs=('Signed ICF','sum')).reset_index()
        weekly['Cumulative ICFs'] = weekly['ICFs'].cumsum()

        # Cost inputs
        cpql = st.number_input("Estimated Cost per Qualified Lead ($)", value=75)
        monthly['Spend'] = monthly['Referrals']*cpql
        monthly['CPICF'] = monthly.apply(lambda r: r['Spend']/r['ICFs'] if r['ICFs']>0 else np.nan, axis=1)
        weekly['Spend'] = weekly['Referrals']*cpql
        weekly['CPICF'] = weekly.apply(lambda r: r['Spend']/r['ICFs'] if r['ICFs']>0 else np.nan, axis=1)

        # Trailing rate and projection
        tw = st.slider("Trailing window (days)",14,120,60)
        recent_icfs = referrals[referrals['Referral Date'] >= pd.Timestamp.now()-pd.Timedelta(days=tw)]['Signed ICF'].sum()
        rate = recent_icfs/(tw/30) if tw>0 else 0
        months_needed = int(np.ceil(remaining/rate)) if rate>0 else None
        proj_date = (datetime.today()+pd.DateOffset(months=months_needed)).date() if months_needed else "N/A"
        st.write(f"📈 Trailing rate: {rate:.1f} ICFs/month")
        st.write(f"📅 Projected completion: {proj_date}")
        if isinstance(proj_date, date):
            days_left = (proj_date-datetime.today().date()).days
            st.metric("⏳ Days to goal", f"{days_left} days")

        # Visualizations
        st.markdown("### 📆 Monthly ICF & Cost Trends")
        st.line_chart(monthly.set_index('Month')[['ICFs','Cumulative ICFs']])
        st.line_chart(monthly.set_index('Month')[['Spend','CPICF']])
        st.markdown("### 📅 Weekly Pacing & Goal")
        st.line_chart(weekly.set_index('Week')[['ICFs','Referrals']])
        weekly['Pct Change ICFs'] = weekly['ICFs'].pct_change()*100
        weekly['Pct Change Referrals'] = weekly['Referrals'].pct_change()*100
        st.dataframe(weekly[['Week','Pct Change ICFs','Pct Change Referrals']].dropna(),use_container_width=True)

        # Goal line overlay
        goal_week = total_goal/(weekly['ICFs'].mean() if weekly['ICFs'].mean()>0 else 1)
        goal_line = pd.Series([goal_week]*len(weekly),index=weekly['Week'])
        goal_df = pd.DataFrame({'Actual':weekly['ICFs'],'Goal':goal_line})
        st.line_chart(goal_df)

        # Flag under-target weeks
        under = weekly[weekly['ICFs']<goal_line.values]
        st.markdown("### 🔻 Under-Target Weeks")
        st.dataframe(under[['Week','ICFs']],use_container_width=True)

        # Export
        csv_export = monthly.to_csv(index=False).encode('utf-8')
        st.download_button("📁 Download Forecast (CSV)", data=csv_export, file_name="forecast.csv", mime="text/csv")
    else:
        st.warning("⬆️ Upload both referral and site files to begin.")

if __name__ == "__main__":
    main()
