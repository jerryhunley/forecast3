# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta 
import io
from sklearn.preprocessing import MinMaxScaler # For site scoring
import traceback 

# --- Page Configuration ---
st.set_page_config(page_title="Recruitment Forecasting Tool", layout="wide")
st.title("📊 Recruitment Forecasting Tool")

# --- Helper Functions ---

@st.cache_data 
def parse_funnel_definition(uploaded_file):
    if uploaded_file is None: return None, None, None 
    try:
        bytes_data = uploaded_file.getvalue(); stringio = io.StringIO(bytes_data.decode("utf-8", errors='replace')) 
        df_funnel_def = pd.read_csv(stringio, sep='\t', header=None) 
        parsed_funnel_definition = {}; parsed_ordered_stages = []; ts_col_map = {} 
        for col_idx in df_funnel_def.columns:
            column_data = df_funnel_def[col_idx]; stage_name = column_data.iloc[0]
            if pd.isna(stage_name) or str(stage_name).strip() == "": continue
            stage_name = str(stage_name).strip().replace('"', ''); parsed_ordered_stages.append(stage_name)
            statuses = column_data.iloc[1:].dropna().astype(str).apply(lambda x: x.strip().replace('"', '')).tolist()
            statuses = [s for s in statuses if s]; 
            if stage_name not in statuses: statuses.append(stage_name)
            parsed_funnel_definition[stage_name] = statuses
            clean_ts_name = f"TS_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"; ts_col_map[stage_name] = clean_ts_name
        if not parsed_ordered_stages: st.error("Could not parse stages from Funnel Definition."); return None, None, None
        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
    except Exception as e: 
        st.error(f"Error parsing Funnel Definition file: {e}"); st.exception(e)
        return None, None, None

def parse_datetime_with_timezone(dt_str):
    if pd.isna(dt_str): return pd.NaT 
    dt_str_cleaned = str(dt_str).strip(); tz_pattern = r'\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$'
    dt_str_no_tz = re.sub(tz_pattern, '', dt_str_cleaned); parsed_dt = pd.to_datetime(dt_str_no_tz, errors='coerce') 
    return parsed_dt

def parse_history_string(history_str):
    if pd.isna(history_str) or str(history_str).strip() == "": return []
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?(?:\s+[A-Za-z]{3,}(?:T)?)?)")
    raw_lines = str(history_str).strip().split('\n'); parsed_events = []
    for line in raw_lines:
        line = line.strip();
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups(); name = name.strip(); dt_obj = parse_datetime_with_timezone(dt_str.strip()) 
            if name and pd.notna(dt_obj): 
                try: py_dt = dt_obj.to_pydatetime(); parsed_events.append((name, py_dt)) 
                except AttributeError: pass 
    try: parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) 
    except TypeError as e: pass 
    return parsed_events

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}; status_to_stage_map = {}
    if not funnel_def: return pd.Series(timestamps) 
    for stage, statuses in funnel_def.items():
        for status in statuses: status_to_stage_map[status] = stage
    all_events = []; stage_hist = row.get(parsed_stage_history_col, []); status_hist = row.get(parsed_status_history_col, [])
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist])
    try: all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) 
    except TypeError as e: pass 
    for event_name, event_dt in all_events:
        if pd.isna(event_dt): continue 
        event_stage = None; 
        if event_name in ordered_stgs: event_stage = event_name
        elif event_name in status_to_stage_map: event_stage = status_to_stage_map[event_name]
        if event_stage and event_stage in ordered_stgs:
            ts_col_name = ts_col_mapping.get(event_stage) 
            if ts_col_name and pd.isna(timestamps[ts_col_name]): timestamps[ts_col_name] = event_dt 
    return pd.Series(timestamps, dtype='datetime64[ns]')

@st.cache_data 
def preprocess_referral_data(_df_raw, funnel_def, ordered_stages, ts_col_map):
    if _df_raw is None or funnel_def is None or ordered_stages is None or ts_col_map is None: return None
    df = _df_raw.copy(); submitted_on_col = None
    if "Submitted On" in df.columns: submitted_on_col = "Submitted On"
    elif "Referral Date" in df.columns: df.rename(columns={"Referral Date": "Submitted On"}, inplace=True); submitted_on_col = "Submitted On"
    else: 
        if "Submitted On" not in df.columns: st.error("Missing 'Submitted On'/'Referral Date'."); return None
        else: submitted_on_col = "Submitted On"
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    initial_rows = len(df); df.dropna(subset=["Submitted On_DT"], inplace=True); rows_dropped = initial_rows - len(df)
    if rows_dropped > 0: st.warning(f"Dropped {rows_dropped} rows due to unparseable date.")
    if df.empty: st.error("No valid data remaining after date parsing."); return None
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    history_cols_to_parse = ['Lead Stage History', 'Lead Status History']; parsed_cols = {}
    for col_name in history_cols_to_parse:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string); parsed_cols[col_name] = parsed_col_name 
        else: st.warning(f"History column '{col_name}' not found.")
    parsed_stage_hist_col = parsed_cols.get('Lead Stage History'); parsed_status_hist_col = parsed_cols.get('Lead Status History')
    if not parsed_stage_hist_col and not parsed_status_hist_col:
        if 'Lead Stage History' not in df.columns and 'Lead Status History' not in df.columns: st.error("Neither history column found.")
        else: st.error("History columns failed to parse.")
        return None 
    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist_col, parsed_status_hist_col, funnel_def, ordered_stages, ts_col_map), axis=1)
    old_ts_cols = [col for col in df.columns if col.startswith('TS_')]; df.drop(columns=old_ts_cols, inplace=True, errors='ignore')
    df = pd.concat([df, timestamp_cols_df], axis=1)
    for stage, ts_col in ts_col_map.items():
         if ts_col in df.columns: df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce') 
    return df

def calculate_proforma_metrics(_processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    if _processed_df is None or _processed_df.empty: return pd.DataFrame()
    if not isinstance(monthly_ad_spend_input, dict): return pd.DataFrame()
    if "Submission_Month" not in _processed_df.columns: return pd.DataFrame()
    processed_df = _processed_df.copy() 
    try:
        cohort_summary = processed_df.groupby("Submission_Month").size().reset_index(name="Total Qualified Referrals")
        cohort_summary = cohort_summary.set_index("Submission_Month")
        cohort_summary["Ad Spend"] = cohort_summary.index.map(monthly_ad_spend_input).fillna(0) 
        reached_stage_cols_map = {}
        for stage_name in ordered_stages:
            ts_col = ts_col_map.get(stage_name)
            if ts_col and ts_col in processed_df.columns:
                reached_col_cleaned = f"Reached_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                reached_stage_cols_map[stage_name] = reached_col_cleaned
                if pd.api.types.is_datetime64_any_dtype(processed_df[ts_col]):
                     reached_stage_count = processed_df.dropna(subset=[ts_col]).groupby("Submission_Month").size()
                     cohort_summary[reached_col_cleaned] = reached_stage_count
                else: cohort_summary[reached_col_cleaned] = 0
        cohort_summary = cohort_summary.fillna(0)
        for col in cohort_summary.columns:
            if col != "Ad Spend": cohort_summary[col] = cohort_summary[col].astype(int)
        cohort_summary["Ad Spend"] = cohort_summary["Ad Spend"].astype(float)
        pof_reached_col = reached_stage_cols_map.get("Passed Online Form")
        base_count_col = None 
        if pof_reached_col and pof_reached_col in cohort_summary.columns:
            cohort_summary.rename(columns={pof_reached_col: "Pre-Screener Qualified"}, inplace=True, errors='ignore') 
            base_count_col = "Pre-Screener Qualified"
        elif "Total Qualified Referrals" in cohort_summary.columns :
            cohort_summary.rename(columns={"Total Qualified Referrals": "Pre-Screener Qualified"}, inplace=True, errors='ignore')
            base_count_col = "Pre-Screener Qualified"
        proforma_metrics = pd.DataFrame(index=cohort_summary.index)
        if base_count_col and base_count_col in cohort_summary.columns: 
            proforma_metrics["Ad Spend"] = cohort_summary["Ad Spend"]
            proforma_metrics["Pre-Screener Qualified"] = cohort_summary[base_count_col]
            proforma_metrics["Cost per Qualified Pre-screen"] = (cohort_summary["Ad Spend"] / cohort_summary[base_count_col].replace(0, np.nan)).round(2) 
            for stage_orig, reached_col in reached_stage_cols_map.items():
                 metric_name = "Pre-Screener Qualified" if stage_orig == "Passed Online Form" else f"Total {stage_orig}"
                 if reached_col in cohort_summary.columns and metric_name not in proforma_metrics.columns:
                     proforma_metrics[metric_name] = cohort_summary[reached_col]
            sts_col = reached_stage_cols_map.get("Sent To Site"); appt_col = reached_stage_cols_map.get("Appointment Scheduled"); icf_col = reached_stage_cols_map.get("Signed ICF")
            if sts_col in cohort_summary.columns: proforma_metrics["Qualified to StS %"] = (cohort_summary[sts_col] / cohort_summary[base_count_col].replace(0, np.nan))
            if sts_col in cohort_summary.columns and appt_col in cohort_summary.columns: proforma_metrics["StS to Appt Sched %"] = (cohort_summary[appt_col] / cohort_summary[sts_col].replace(0, np.nan))
            if appt_col in cohort_summary.columns and icf_col in cohort_summary.columns: proforma_metrics["Appt Sched to ICF %"] = (cohort_summary[icf_col] / cohort_summary[appt_col].replace(0, np.nan))
            if icf_col in cohort_summary.columns:
                proforma_metrics["Qualified to ICF %"] = (cohort_summary[icf_col] / cohort_summary[base_count_col].replace(0, np.nan))
                proforma_metrics["Cost Per ICF"] = (cohort_summary["Ad Spend"] / cohort_summary[icf_col].replace(0, np.nan)).round(2)
        return proforma_metrics
    except Exception as e: 
        st.error(f"ProForma Calc Error: {e}"); st.exception(e) 
        return pd.DataFrame() 

def calculate_avg_lag_generic(df, col_from, col_to):
    if not col_from or not col_to or col_from not in df or col_to not in df or \
       not pd.api.types.is_datetime64_any_dtype(df[col_from]) or \
       not pd.api.types.is_datetime64_any_dtype(df[col_to]):
        return np.nan
    valid_df = df.dropna(subset=[col_from, col_to])
    if valid_df.empty: return np.nan
    try:
        col_from_dt = pd.to_datetime(valid_df[col_from], errors='coerce')
        col_to_dt = pd.to_datetime(valid_df[col_to], errors='coerce')
    except Exception: return np.nan
    valid_comparison_df = valid_df.loc[col_from_dt.notna() & col_to_dt.notna()].copy()
    if valid_comparison_df.empty: return np.nan
    diff = col_to_dt.loc[valid_comparison_df.index] - col_from_dt.loc[valid_comparison_df.index]
    diff_positive = diff[diff >= pd.Timedelta(days=0)] 
    if diff_positive.empty: return np.nan
    return diff_positive.mean().total_seconds() / (60*60*24)

@st.cache_data
def calculate_overall_inter_stage_lags(_processed_df, ordered_stages, ts_col_map):
    if _processed_df is None or _processed_df.empty or not ordered_stages or not ts_col_map:
        return {}
    inter_stage_lags = {}
    for i in range(len(ordered_stages) - 1):
        stage_from = ordered_stages[i]; stage_to = ordered_stages[i+1]
        ts_col_from = ts_col_map.get(stage_from); ts_col_to = ts_col_map.get(stage_to)
        if ts_col_from and ts_col_to and ts_col_from in _processed_df.columns and ts_col_to in _processed_df.columns:
            avg_lag = calculate_avg_lag_generic(_processed_df, ts_col_from, ts_col_to)
            inter_stage_lags[f"{stage_from} -> {stage_to}"] = avg_lag
        else: inter_stage_lags[f"{stage_from} -> {stage_to}"] = np.nan
    return inter_stage_lags

# --- MODIFIED: calculate_site_metrics function ---
def calculate_site_metrics(_processed_df, ordered_stages, ts_col_map):
    if _processed_df is None or _processed_df.empty or 'Site' not in _processed_df.columns: 
        return pd.DataFrame() 
    
    processed_df = _processed_df.copy()
    site_metrics_list = []

    # Define the key stages for granular rate calculation (must match projection logic)
    # These names must exist in `ts_col_map` (derived from funnel definition)
    # and `ordered_stages`
    # POF = Passed Online Form, PSA = Pre-Screening Activities, StS = Sent To Site, Appt = Appointment Scheduled, ICF = Signed ICF
    # SF = Screen Failed
    
    # Stage names as strings (ensure these are exactly as in your funnel definition/ts_col_map keys)
    stage_pof_name = "Passed Online Form"
    stage_psa_name = "Pre-Screening Activities" 
    stage_sts_name = "Sent To Site"
    stage_appt_name = "Appointment Scheduled"
    stage_icf_name = "Signed ICF"
    stage_sf_name = "Screen Failed" # For screen fail rate

    # Get corresponding timestamp column names
    ts_pof_col = ts_col_map.get(stage_pof_name)
    ts_psa_col = ts_col_map.get(stage_psa_name)
    ts_sts_col = ts_col_map.get(stage_sts_name)
    ts_appt_col = ts_col_map.get(stage_appt_name)
    ts_icf_col = ts_col_map.get(stage_icf_name)
    ts_sf_col = ts_col_map.get(stage_sf_name)

    # Ensure all potentially needed timestamp columns exist in the DataFrame, add as NaT if not
    # This is important if some stages are optional or not always present for all referrals
    potential_ts_cols = [ts_pof_col, ts_psa_col, ts_sts_col, ts_appt_col, ts_icf_col, ts_sf_col]
    for col in potential_ts_cols:
        if col and col not in processed_df.columns:
            processed_df[col] = pd.NaT
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
    
    site_contact_attempt_statuses = ["Site Contact Attempt 1"] # As before
    post_sts_progress_stages = ["Appointment Scheduled", "Signed ICF", "Enrolled", "Screen Failed"] # As before

    try: 
        site_groups = processed_df.groupby('Site')
        for site_name, group in site_groups:
            metrics = {'Site': site_name}
            
            # Calculate counts of reaching each key stage for this site
            # Using .notna().sum() on the boolean series is robust
            count_pof = group[ts_pof_col].notna().sum() if ts_pof_col and ts_pof_col in group else 0
            count_psa = group[ts_psa_col].notna().sum() if ts_psa_col and ts_psa_col in group else 0
            count_sts = group[ts_sts_col].notna().sum() if ts_sts_col and ts_sts_col in group else 0
            count_appt = group[ts_appt_col].notna().sum() if ts_appt_col and ts_appt_col in group else 0
            count_icf = group[ts_icf_col].notna().sum() if ts_icf_col and ts_icf_col in group else 0
            count_sf = group[ts_sf_col].notna().sum() if ts_sf_col and ts_sf_col in group else 0

            metrics['Site Count POF'] = count_pof # Total referrals for site that reached POF
            metrics['Site Count PSA'] = count_psa
            metrics['Site Count StS'] = count_sts # Replaces 'Reached StS'
            metrics['Site Count Appt'] = count_appt # Replaces 'Reached Appt'
            metrics['Site Count ICF'] = count_icf # Replaces 'Reached ICF'

            # --- NEW Granular Site-Specific Conversion Rates ---
            # Denominators are counts from the 'from' stage
            metrics['Site POF -> PSA %'] = (count_psa / count_pof) if count_pof > 0 else 0.0
            metrics['Site PSA -> StS %'] = (count_sts / count_psa) if count_psa > 0 else 0.0
            # Keep existing ones for direct comparison if needed, but ensure denominators are consistent
            # The new names are more explicit for the chained projection
            metrics['Site StS -> Appt %'] = (count_appt / count_sts) if count_sts > 0 else 0.0
            metrics['Site Appt -> ICF %'] = (count_icf / count_appt) if count_appt > 0 else 0.0
            
            # Overall POF to ICF for this site (can be used for 'Qual -> ICF %' or as a separate metric)
            # 'Total Qualified' for site scoring could be based on count_pof
            metrics['Total Qualified'] = count_pof # For site scoring, 'Total Qualified' is now POF count
            metrics['Qual -> ICF %'] = (count_icf / count_pof) if count_pof > 0 else 0.0
            
            # Lag calculations (as before, but ensure ts_qual_col used for Qual->ICF lag is ts_pof_col)
            metrics['Lag Qual -> ICF (Days)'] = calculate_avg_lag_generic(group, ts_pof_col, ts_icf_col) 
            
            # TTC and Funnel Movement (as before)
            ttc_times = []; funnel_movement_steps = []
            sent_to_site_group = group.dropna(subset=[ts_sts_col]) if ts_sts_col and ts_sts_col in group else pd.DataFrame()
            parsed_status_col = f"Parsed_Lead_Status_History"; parsed_stage_col = f"Parsed_Lead_Stage_History"
            if not sent_to_site_group.empty and parsed_status_col in sent_to_site_group and parsed_stage_col in sent_to_site_group :
                for idx, row in sent_to_site_group.iterrows():
                    ts_sent = row[ts_sts_col]; first_contact_ts = pd.NaT
                    history_list = row.get(parsed_status_col, [])
                    if history_list: 
                        for status_name, event_dt in history_list:
                            if status_name in site_contact_attempt_statuses and pd.notna(event_dt) and pd.notna(ts_sent) and event_dt >= ts_sent:
                                first_contact_ts = event_dt; break
                    if pd.notna(first_contact_ts) and pd.notna(ts_sent):
                        time_diff = first_contact_ts - ts_sent
                        if time_diff >= pd.Timedelta(0): ttc_times.append(time_diff.total_seconds() / (60*60*24))                 
                    stages_reached_post_sts = set()
                    stage_history_list = row.get(parsed_stage_col, [])
                    if stage_history_list and pd.notna(ts_sent): 
                         for stage_name, event_dt in stage_history_list:
                             if stage_name in post_sts_progress_stages and pd.notna(event_dt) and event_dt > ts_sent: stages_reached_post_sts.add(stage_name)
                    funnel_movement_steps.append(len(stages_reached_post_sts))
            metrics['Avg TTC (Days)'] = np.mean(ttc_times) if ttc_times else np.nan
            metrics['Avg Funnel Movement Steps'] = np.mean(funnel_movement_steps) if funnel_movement_steps else 0
            
            # Site Screen Fail % (as before, based on count_icf if ICF is pre-SF)
            # If Screen Fail can happen *after* ICF, then denominator might need to be (count_icf + count_sf)
            # Assuming SF is a terminal state *after* attempting ICF or around the same time.
            # Current logic: SF% of those who reached ICF. 
            # A more common definition might be SF / (ICF + SF) among those who got to screening point.
            # For now, keeping existing logic:
            metrics['Site Screen Fail %'] = (count_sf / count_icf) if count_icf > 0 else 0.0 
            # Consider if denominator should be count_appt or count_sts if SF happens before ICF.
            # If ts_sf_col is available, a more robust SF rate post-appointment could be:
            # count_sf_post_appt = group.loc[group[ts_appt_col].notna() & group[ts_sf_col].notna()].shape[0]
            # metrics['Site SF Post-Appt %'] = (count_sf_post_appt / count_appt) if count_appt > 0 else 0.0

            site_metrics_list.append(metrics)
        
        site_metrics_df = pd.DataFrame(site_metrics_list)
        return site_metrics_df 
    except Exception as e: 
        st.error(f"Error calculating site metrics: {e}"); st.exception(e)
        return pd.DataFrame()
# --- End MODIFIED: calculate_site_metrics function ---


def score_sites(_site_metrics_df, weights):
    if _site_metrics_df is None or _site_metrics_df.empty: return pd.DataFrame()
    try: 
        site_metrics_df = _site_metrics_df.copy() 
        if 'Site' not in site_metrics_df.columns:
             if site_metrics_df.index.name == 'Site': 
                 site_metrics_df = site_metrics_df.reset_index() 
             else: 
                 st.error("Site Scoring: 'Site' column missing.")
                 return pd.DataFrame()
        
        site_metrics_df_indexed = site_metrics_df.set_index('Site')
        # Update metrics_to_scale and lower_is_better if new site metrics are added to scoring
        metrics_to_scale = list(weights.keys())
        lower_is_better = ["Avg TTC (Days)", "Site Screen Fail %", "Lag Qual -> ICF (Days)"]

        scaled_metrics_data = site_metrics_df_indexed.reindex(columns=metrics_to_scale).copy()
        for col in metrics_to_scale:
            if col not in scaled_metrics_data.columns: # If a weight is for a metric not in data
                scaled_metrics_data[col] = 0 if col not in lower_is_better else np.nan 
            if col in lower_is_better:
                max_val = scaled_metrics_data[col].max()
                fill_val = (max_val + 1) if pd.notna(max_val) and max_val > 0 else 999 
                scaled_metrics_data[col] = scaled_metrics_data[col].fillna(fill_val)
            else: 
                scaled_metrics_data[col] = scaled_metrics_data[col].fillna(0)
        scaled_metrics_display = pd.DataFrame(index=scaled_metrics_data.index) 
        if not scaled_metrics_data.empty and len(scaled_metrics_data) > 0: 
            for col in metrics_to_scale:
                 if col in scaled_metrics_data.columns: 
                     min_val = scaled_metrics_data[col].min(); max_val = scaled_metrics_data[col].max()
                     if min_val == max_val: 
                         scaled_metrics_display[col] = 0.5 
                     elif pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val) != 0 : 
                         scaler = MinMaxScaler()
                         scaled_values = scaler.fit_transform(scaled_metrics_data[[col]]) 
                         scaled_metrics_display[col] = scaled_values.flatten() 
                     else: 
                          scaled_metrics_display[col] = 0.5 # Fallback if scaling fails
                 else: scaled_metrics_display[col] = 0.5 # Should not happen due to reindex
            for col in lower_is_better: 
                if col in scaled_metrics_display.columns: 
                    scaled_metrics_display[col] = 1 - scaled_metrics_display[col]
        site_metrics_df_indexed['Score_Raw'] = 0.0; total_weight_applied = 0.0
        for metric, weight_pct in weights.items(): 
             weight = weight_pct 
             if metric in scaled_metrics_display.columns: 
                 current_scaled_metric_series = scaled_metrics_display.get(metric)
                 if current_scaled_metric_series is not None:
                     site_metrics_df_indexed['Score_Raw'] += current_scaled_metric_series.fillna(0.5) * weight 
                 else: # Should not happen due to reindex
                      site_metrics_df_indexed['Score_Raw'] += 0.5 * weight 
                 total_weight_applied += abs(weight) 
        if total_weight_applied > 0: 
            site_metrics_df_indexed['Score'] = (site_metrics_df_indexed['Score_Raw'] / total_weight_applied) * 100
        else: 
            site_metrics_df_indexed['Score'] = 0.0
        site_metrics_df_indexed['Score'] = site_metrics_df_indexed['Score'].fillna(0.0)
        if len(site_metrics_df_indexed) > 1: 
            site_metrics_df_indexed['Score_Rank_Percentile'] = site_metrics_df_indexed['Score'].rank(pct=True)
            bins = [0, 0.10, 0.25, 0.60, 0.85, 1.0]; labels = ['F', 'D', 'C', 'B', 'A']
            try: 
                site_metrics_df_indexed['Grade'] = pd.qcut(site_metrics_df_indexed['Score_Rank_Percentile'], q=bins, labels=labels, duplicates='drop') 
            except ValueError: 
                 st.warning("Using fixed score ranges for grading (percentile failed).")
                 def assign_grade_fallback(score_value): 
                     if pd.isna(score_value): return 'N/A'
                     score_value = round(score_value)
                     if score_value >= 90: return 'A' 
                     elif score_value >= 80: return 'B'
                     elif score_value >= 70: return 'C'
                     elif score_value >= 60: return 'D'
                     else: return 'F'
                 site_metrics_df_indexed['Grade'] = site_metrics_df_indexed['Score'].apply(assign_grade_fallback)
            site_metrics_df_indexed['Grade'] = site_metrics_df_indexed['Grade'].astype(str).replace('nan', 'N/A') 
        elif len(site_metrics_df_indexed) == 1: 
            def assign_single_site_grade(score_value):
                if pd.isna(score_value): return 'N/A'
                score_value = round(score_value)
                if score_value >= 90: return 'A' 
                elif score_value >= 80: return 'B'
                elif score_value >= 70: return 'C'
                elif score_value >= 60: return 'D'
                else: return 'F'
            site_metrics_df_indexed['Grade'] = site_metrics_df_indexed['Score'].apply(assign_single_site_grade)
        else: 
            site_metrics_df_indexed['Grade'] = None 
        final_df_output = site_metrics_df_indexed.reset_index()
        if 'Score' in final_df_output.columns:
            final_df_output = final_df_output.sort_values('Score', ascending=False)
        return final_df_output 
    except Exception as e: 
        st.error(f"Error during Site Scoring: {e}"); st.exception(e)
        if _site_metrics_df is not None and not _site_metrics_df.empty:
             if _site_metrics_df.index.name == 'Site' and 'Site' not in _site_metrics_df.columns: 
                 return _site_metrics_df.reset_index()
             return _site_metrics_df
        return pd.DataFrame()

def determine_effective_projection_rates(_processed_df, ordered_stages, ts_col_map, 
                                          rate_method_sidebar, rolling_window_sidebar, manual_rates_sidebar,
                                          inter_stage_lags_for_maturity, 
                                          sidebar_display_area=None): 
    MIN_DENOMINATOR_FOR_RATE_CALC = 5 
    DEFAULT_MATURITY_DAYS = 45 
    MATURITY_LAG_MULTIPLIER = 1.5 
    MIN_EFFECTIVE_MATURITY_DAYS = 20 
    MIN_TOTAL_DENOMINATOR_FOR_OVERALL_RATE = 20 
    if _processed_df is None or _processed_df.empty: 
        if sidebar_display_area: sidebar_display_area.caption("Using manual rates (No historical data for rolling).")
        return manual_rates_sidebar, "Manual (No History)"
    if rate_method_sidebar == 'Manual Input Below':
        if sidebar_display_area: sidebar_display_area.caption("Using manually input conversion rates for projection.")
        return manual_rates_sidebar, "Manual Input"
    calculated_rolling_rates = {}; method_description = "Manual (Error or No History for Rolling)"; substitutions_made_log = [] 
    MATURITY_PERIODS_DAYS = {}
    if inter_stage_lags_for_maturity:
        for rate_key_for_lag in manual_rates_sidebar.keys():
            avg_lag_for_key = inter_stage_lags_for_maturity.get(rate_key_for_lag)
            if pd.notna(avg_lag_for_key) and avg_lag_for_key > 0:
                calculated_maturity = round(MATURITY_LAG_MULTIPLIER * avg_lag_for_key)
                current_maturity_period = max(calculated_maturity, MIN_EFFECTIVE_MATURITY_DAYS)
                MATURITY_PERIODS_DAYS[rate_key_for_lag] = current_maturity_period
            else: MATURITY_PERIODS_DAYS[rate_key_for_lag] = DEFAULT_MATURITY_DAYS
    else: 
        for rate_key_for_lag in manual_rates_sidebar.keys(): MATURITY_PERIODS_DAYS[rate_key_for_lag] = DEFAULT_MATURITY_DAYS
        substitutions_made_log.append(f"Maturity: Inter-stage lags N/A, used default {DEFAULT_MATURITY_DAYS}d for all.")
    try:
        if "Submission_Month" not in _processed_df.columns or _processed_df["Submission_Month"].dropna().empty:
            if sidebar_display_area: st.sidebar.warning("Not enough historical submission month data. Using manual rates.")
            return manual_rates_sidebar, "Manual (No History)"
        hist_counts = _processed_df.groupby("Submission_Month").size().to_frame(name="Total Qualified Referrals") 
        reached_stage_cols_map_hist = {}
        for stage_name in ordered_stages:
            ts_col = ts_col_map.get(stage_name)
            if ts_col and ts_col in _processed_df.columns and pd.api.types.is_datetime64_any_dtype(_processed_df[ts_col]):
                reached_col_cleaned = f"Reached_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                reached_stage_cols_map_hist[stage_name] = reached_col_cleaned
                stage_monthly_counts = _processed_df.dropna(subset=[ts_col]).groupby(_processed_df['Submission_Month']).size()
                hist_counts = hist_counts.join(stage_monthly_counts.rename(reached_col_cleaned), how='left')
        hist_counts = hist_counts.fillna(0)
        pof_hist_col = reached_stage_cols_map_hist.get("Passed Online Form") 
        valid_historical_rates_found = False
        for rate_key in manual_rates_sidebar.keys():
            try: stage_from_name, stage_to_name = rate_key.split(" -> ")
            except ValueError:
                calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0); substitutions_made_log.append(f"{rate_key}: Error parsing names, used manual."); continue
            if stage_from_name == "Passed Online Form": actual_col_from = pof_hist_col if pof_hist_col and pof_hist_col in hist_counts.columns else "Total Qualified Referrals"
            else: actual_col_from = reached_stage_cols_map_hist.get(stage_from_name)
            col_to_cleaned_name = reached_stage_cols_map_hist.get(stage_to_name)
            if actual_col_from in hist_counts.columns and col_to_cleaned_name in hist_counts.columns:
                total_numerator = hist_counts[col_to_cleaned_name].sum(); total_denominator = hist_counts[actual_col_from].sum()
                overall_hist_rate_for_key = (total_numerator / total_denominator) if total_denominator > 0 else np.nan
                manual_rate_for_key = manual_rates_sidebar.get(rate_key, 0.0)
                maturity_days_for_this_rate = MATURITY_PERIODS_DAYS.get(rate_key, DEFAULT_MATURITY_DAYS)
                adjusted_monthly_rates_list = []; months_used_for_rate = [] 
                for month_period in hist_counts.index:
                    if month_period.end_time + pd.Timedelta(days=maturity_days_for_this_rate) < pd.Timestamp(datetime.now()):
                        months_used_for_rate.append(month_period)
                        numerator_val = hist_counts.loc[month_period, col_to_cleaned_name]; denominator_val = hist_counts.loc[month_period, actual_col_from]
                        rate_for_this_month = 0.0 
                        if denominator_val < MIN_DENOMINATOR_FOR_RATE_CALC:
                            rate_for_this_month = manual_rate_for_key; log_reason_detail = f"used manual rate ({manual_rate_for_key*100:.1f}%)"
                            is_manual_rate_placeholder = (manual_rate_for_key >= 0.99 or manual_rate_for_key <= 0.01)
                            if is_manual_rate_placeholder:
                                if pd.notna(overall_hist_rate_for_key) and total_denominator >= MIN_TOTAL_DENOMINATOR_FOR_OVERALL_RATE:
                                    rate_for_this_month = overall_hist_rate_for_key
                                    log_reason_detail = f"manual placeholder, used overall hist. ({overall_hist_rate_for_key*100:.1f}%, total N={total_denominator})"
                                else: log_reason_detail = f"manual placeholder, overall hist. N/A or low N ({total_denominator}), stuck with manual ({manual_rate_for_key*100:.1f}%)"
                            substitutions_made_log.append(f"Mth {month_period.strftime('%Y-%m')} for '{rate_key}': Denom ({denominator_val}) < {MIN_DENOMINATOR_FOR_RATE_CALC}, {log_reason_detail}. Mat: {maturity_days_for_this_rate}d.")
                        elif denominator_val > 0: rate_for_this_month = numerator_val / denominator_val
                        adjusted_monthly_rates_list.append(rate_for_this_month)
                    else: substitutions_made_log.append(f"Mth {month_period.strftime('%Y-%m')} for '{rate_key}': Excluded (too recent, mat: {maturity_days_for_this_rate}d).")
                if not adjusted_monthly_rates_list:
                    calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0); substitutions_made_log.append(f"{rate_key}: No mature hist. mths (mat: {maturity_days_for_this_rate}d), used manual."); continue
                adjusted_monthly_rates_series = pd.Series(adjusted_monthly_rates_list, index=pd.PeriodIndex(months_used_for_rate, freq='M'))
                actual_window_calc = min(rolling_window_sidebar, len(adjusted_monthly_rates_series))
                if actual_window_calc > 0 and not adjusted_monthly_rates_series.empty:
                    rolling_avg_rate = adjusted_monthly_rates_series.rolling(window=actual_window_calc, min_periods=1).mean()
                    if not rolling_avg_rate.empty:
                        latest_rolling_rate = rolling_avg_rate.iloc[-1]
                        calculated_rolling_rates[rate_key] = latest_rolling_rate if pd.notna(latest_rolling_rate) else 0.0
                        valid_historical_rates_found = True
                    else: calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0); substitutions_made_log.append(f"{rate_key}: Rolling avg empty (mat: {maturity_days_for_this_rate}d), used manual.")
                else: 
                    if not adjusted_monthly_rates_series.empty:
                        mean_mature_rate = adjusted_monthly_rates_series.mean()
                        calculated_rolling_rates[rate_key] = mean_mature_rate if pd.notna(mean_mature_rate) else manual_rates_sidebar.get(rate_key, 0.0)
                        substitutions_made_log.append(f"{rate_key}: Window {actual_window_calc} (mat: {maturity_days_for_this_rate}d) too small; used mean of mature or manual. Valid: {pd.notna(mean_mature_rate)}")
                        if pd.notna(mean_mature_rate): valid_historical_rates_found = True
                    else: calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0); substitutions_made_log.append(f"{rate_key}: No mature data for rolling (mat: {maturity_days_for_this_rate}d), used manual.")
            else: calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0); substitutions_made_log.append(f"{rate_key}: Stage cols N/A, used manual.")
        if sidebar_display_area and substitutions_made_log:
            with sidebar_display_area.expander("Rolling Rate Calculation Log (Adjustments & Maturity)", expanded=False):
                sidebar_display_area.caption("Maturity Periods Applied (Days):")
                if MATURITY_PERIODS_DAYS:
                    for r_key_disp, mat_days_disp in MATURITY_PERIODS_DAYS.items(): sidebar_display_area.caption(f"- {r_key_disp}: {mat_days_disp} days")
                else: sidebar_display_area.caption("Maturity periods N/A.")
                sidebar_display_area.caption("--- Substitution/Exclusion Log ---")
                for log_entry in substitutions_made_log: st.caption(log_entry)
        if not valid_historical_rates_found:
            if sidebar_display_area: sidebar_display_area.warning("No valid historical rolling rates, using manual inputs.")
            return manual_rates_sidebar, "Manual (All Rolling Calcs Failed or Invalid)"
        else: 
            if sidebar_display_area: 
                sidebar_display_area.markdown("---"); sidebar_display_area.subheader(f"Effective {rolling_window_sidebar}-Mo. Rolling Rates (Adj. & Matured):")
                for key, val in calculated_rolling_rates.items():
                    if key in manual_rates_sidebar: sidebar_display_area.text(f"- {key}: {val*100:.1f}%")
                sidebar_display_area.markdown("---")
            return calculated_rolling_rates, f"Rolling {rolling_window_sidebar}-Month Avg (Adj. & Matured)"
    except Exception as e:
        if sidebar_display_area: sidebar_display_area.error(f"Error calculating rolling rates: {e}"); sidebar_display_area.exception(e)
        return manual_rates_sidebar, "Manual (Error in Rolling)"

@st.cache_data 
def calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs): 
    default_return_tuple = pd.DataFrame(), np.nan, "N/A", "N/A", pd.DataFrame(), "N/A" 
    if _processed_df is None or _processed_df.empty: return default_return_tuple
    required_keys = ['horizon', 'spend_dict', 'cpqr_dict', 'final_conv_rates', 'goal_icf', 'site_performance_data', 'inter_stage_lags'] 
    if not isinstance(projection_inputs, dict) or not all(k in projection_inputs for k in required_keys):
        st.warning(f"Proj: Missing inputs. Need: {required_keys}. Got: {list(projection_inputs.keys())}")
        return default_return_tuple
        
    processed_df = _processed_df.copy(); horizon = projection_inputs['horizon']
    future_spend_dict = projection_inputs['spend_dict']
    assumed_cpqr_dict = projection_inputs['cpqr_dict'] 
    final_projection_conv_rates = projection_inputs['final_conv_rates'] 
    goal_total_icfs = projection_inputs['goal_icf']
    site_performance_data = projection_inputs['site_performance_data'] # This is st.session_state.site_metrics_calculated
    inter_stage_lags = projection_inputs.get('inter_stage_lags', {}) 
    
    avg_actual_lag_days_for_display = np.nan
    lag_calculation_method_message = "Lag not calculated."
    projection_segments_for_lag = [
        ("Passed Online Form", "Pre-Screening Activities"),
        ("Pre-Screening Activities", "Sent To Site"),
        ("Sent To Site", "Appointment Scheduled"),
        ("Appointment Scheduled", "Signed ICF")
    ]
    calculated_sum_of_lags = 0; valid_segments_count = 0; all_segments_found_and_valid = True
    if inter_stage_lags:
        for stage_from, stage_to in projection_segments_for_lag:
            lag_key = f"{stage_from} -> {stage_to}"; lag_value = inter_stage_lags.get(lag_key)
            if ts_col_map.get(stage_from) and ts_col_map.get(stage_to):
                if pd.notna(lag_value):
                    calculated_sum_of_lags += lag_value; valid_segments_count += 1
                else: all_segments_found_and_valid = False; break 
            else: all_segments_found_and_valid = False; break
        if all_segments_found_and_valid and valid_segments_count == len(projection_segments_for_lag):
            avg_actual_lag_days_for_display = calculated_sum_of_lags
            lag_calculation_method_message = "Using summed inter-stage lags for projection path."
        else: 
            all_segments_found_and_valid = False; lag_calculation_method_message = "Summed inter-stage lag failed. " 
    else: 
        all_segments_found_and_valid = False; lag_calculation_method_message = "Inter-stage lags not available. " 
    if not all_segments_found_and_valid or pd.isna(avg_actual_lag_days_for_display):
        start_stage_for_overall_lag = ordered_stages[0] if ordered_stages and len(ordered_stages) > 0 else None
        end_stage_for_overall_lag = "Signed ICF"; overall_lag_calc_val = np.nan
        if start_stage_for_overall_lag:
            ts_col_start_overall = ts_col_map.get(start_stage_for_overall_lag); ts_col_end_overall = ts_col_map.get(end_stage_for_overall_lag)
            if ts_col_start_overall and ts_col_end_overall and \
               ts_col_start_overall in processed_df.columns and ts_col_end_overall in processed_df.columns:
                overall_lag_calc_val = calculate_avg_lag_generic(processed_df, ts_col_start_overall, ts_col_end_overall)
        if pd.notna(overall_lag_calc_val):
            avg_actual_lag_days_for_display = overall_lag_calc_val; lag_calculation_method_message += "Used historical overall lag (first funnel stage to ICF)." 
        else: avg_actual_lag_days_for_display = 30.0; lag_calculation_method_message += "Used default lag (30 days)." 
    if pd.isna(avg_actual_lag_days_for_display): avg_actual_lag_days_for_display = 30.0; lag_calculation_method_message = "Critical Lag Error: All methods failed. Used default 30 days."
    lpi_date_str = "Goal Not Met"; ads_off_date_str = "N/A"; site_level_projections_df = pd.DataFrame() 
    try:
        last_historical_month = processed_df["Submission_Month"].max() if "Submission_Month" in processed_df and not processed_df["Submission_Month"].empty else pd.Period(datetime.now(), freq='M') - 1
        proj_start_month = last_historical_month + 1 
        future_months = pd.period_range(start=proj_start_month, periods=horizon, freq='M')
        projection_cohorts = pd.DataFrame(index=future_months)
        projection_cohorts['Forecasted_Ad_Spend'] = [future_spend_dict.get(m, 0.0) for m in future_months]
        forecasted_psq_list = []
        for month_period in future_months:
            spend = projection_cohorts.loc[month_period, 'Forecasted_Ad_Spend']
            cpqr_for_month = assumed_cpqr_dict.get(month_period, 120.0) 
            if cpqr_for_month <= 0: cpqr_for_month = 120.0 
            forecasted_psq_list.append(np.round(spend / cpqr_for_month).astype(int) if cpqr_for_month > 0 else 0)
        projection_cohorts['Forecasted_PSQ'] = forecasted_psq_list
        projection_cohorts['Forecasted_PSQ'] = projection_cohorts['Forecasted_PSQ'].fillna(0).astype(int)
        icf_stage_name_proj = "Signed ICF"; current_proj_count_col = 'Forecasted_PSQ'; icf_proj_col = ""
        for stage_from, stage_to in projection_segments_for_lag:
            conv_rate_key = f"{stage_from} -> {stage_to}"; conv_rate = final_projection_conv_rates.get(conv_rate_key, 0.0)
            proj_col_to = f"Projected_{stage_to.replace(' ', '_').replace('(', '').replace(')', '')}"
            if current_proj_count_col in projection_cohorts.columns:
                proj_counts = (projection_cohorts[current_proj_count_col] * conv_rate)
                projection_cohorts[proj_col_to] = proj_counts.round(0).fillna(0).astype(int)
                current_proj_count_col = proj_col_to
            else: projection_cohorts[proj_col_to] = 0; current_proj_count_col = proj_col_to
            if stage_to == icf_stage_name_proj: icf_proj_col = proj_col_to; break 
        projection_results = pd.DataFrame(index=future_months); projection_results['Projected_ICF_Landed'] = 0.0 
        if not icf_proj_col or icf_proj_col not in projection_cohorts.columns:
            st.error(f"Critical Error: Projected ICF column ('{icf_proj_col}') not found."); return default_return_tuple[0], default_return_tuple[1], default_return_tuple[2], default_return_tuple[3], default_return_tuple[4], "ICF Proj Col Missing"
        current_lag_days_to_use = avg_actual_lag_days_for_display; days_in_avg_month = 30.4375
        for start_month_period in projection_cohorts.index:
            icfs_from_this_cohort = projection_cohorts.loc[start_month_period, icf_proj_col]
            if icfs_from_this_cohort == 0: continue
            full_lag_months = int(np.floor(current_lag_days_to_use / days_in_avg_month))
            remaining_lag_days_component = current_lag_days_to_use - (full_lag_months * days_in_avg_month)
            fraction_for_next_month = remaining_lag_days_component / days_in_avg_month
            fraction_for_current_offset_month = 1.0 - fraction_for_next_month
            icfs_month_1 = icfs_from_this_cohort * fraction_for_current_offset_month
            icfs_month_2 = icfs_from_this_cohort * fraction_for_next_month
            landing_month_1_period = start_month_period + full_lag_months
            landing_month_2_period = start_month_period + full_lag_months + 1
            if landing_month_1_period in projection_results.index: projection_results.loc[landing_month_1_period, 'Projected_ICF_Landed'] += icfs_month_1
            if landing_month_2_period in projection_results.index: projection_results.loc[landing_month_2_period, 'Projected_ICF_Landed'] += icfs_month_2
        projection_results['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed'].round(0).fillna(0).astype(int)
        projection_cohorts['Projected_CPICF_Cohort'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col].replace(0, np.nan)).round(2)
        projection_results['Cumulative_ICF_Landed'] = projection_results['Projected_ICF_Landed'].cumsum()
        lpi_month_series = projection_results[projection_results['Cumulative_ICF_Landed'] >= goal_total_icfs]
        if not lpi_month_series.empty:
            lpi_month_period = lpi_month_series.index[0]; icfs_in_lpi_month = projection_results.loc[lpi_month_period, 'Projected_ICF_Landed']
            cumulative_before_lpi_direct = projection_results['Cumulative_ICF_Landed'].shift(1).fillna(0).loc[lpi_month_period]
            icfs_needed_in_lpi_month = goal_total_icfs - cumulative_before_lpi_direct
            if icfs_in_lpi_month > 0:
                fraction_of_lpi_month = max(0,min(1, icfs_needed_in_lpi_month / icfs_in_lpi_month))
                lpi_day_offset = int(np.ceil(fraction_of_lpi_month * days_in_avg_month)); lpi_day_offset = max(1, lpi_day_offset) 
                lpi_date_val = lpi_month_period.start_time + pd.Timedelta(days=lpi_day_offset -1); lpi_date_str = lpi_date_val.strftime('%Y-%m-%d')
            elif icfs_needed_in_lpi_month <= 0: lpi_date_str = (lpi_month_period.start_time - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            else: lpi_date_str = lpi_month_period.start_time.strftime('%Y-%m-%d') 
        projection_cohorts['Cumulative_Projected_ICF_Generated'] = projection_cohorts[icf_proj_col].cumsum()
        ads_off_month_series = projection_cohorts[projection_cohorts['Cumulative_Projected_ICF_Generated'] >= goal_total_icfs]
        if not ads_off_month_series.empty:
            ads_off_month_period = ads_off_month_series.index[0]; ads_off_date_str = ads_off_month_period.end_time.strftime('%Y-%m-%d')
        display_df = pd.DataFrame(index=future_months)
        display_df['Forecasted_Ad_Spend'] = projection_cohorts['Forecasted_Ad_Spend']
        display_df['Forecasted_Qual_Referrals'] = projection_cohorts['Forecasted_PSQ']
        display_df['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed'] 
        cpicf_cohort_series = projection_cohorts['Projected_CPICF_Cohort']; cpicf_display_series = pd.Series(index=future_months, dtype=float) 
        lag_for_cpicf_display = int(np.round(avg_actual_lag_days_for_display / 30.4375))
        for i_cohort, cohort_start_month in enumerate(projection_cohorts.index):
            cohort_cpicf = projection_cohorts.iloc[i_cohort]['Projected_CPICF_Cohort']
            primary_land_m = cohort_start_month + lag_for_cpicf_display
            if primary_land_m in cpicf_display_series.index and pd.isna(cpicf_display_series.loc[primary_land_m]): 
                cpicf_display_series.loc[primary_land_m] = cohort_cpicf
        display_df['Projected_CPICF_Cohort_Source'] = cpicf_display_series

        # --- MODIFIED: Site Level Breakdown Logic ---
        if 'Site' in _processed_df.columns and not _processed_df['Site'].empty and \
           ordered_stages and len(ordered_stages) > 0: # Simpler check
            
            # Site's share of "Forecasted_PSQ" (which are "Passed Online Form" equivalent)
            # This proportioning based on *all* historical referrals to a site might need refinement
            # if the desire is to proportion based on historical POF share.
            # For now, using existing logic for initial distribution:
            historical_site_referral_counts = _processed_df['Site'].value_counts()
            total_historical_referrals = historical_site_referral_counts.sum()
            site_proportions_overall = historical_site_referral_counts / total_historical_referrals if total_historical_referrals > 0 else pd.Series(dtype=float)

            all_sites = sorted(_processed_df['Site'].unique())
            site_data_collector = {} 
            for site_name in all_sites:
                site_data_collector[site_name] = {}
                for month_period in future_months:
                    month_str = month_period.strftime('%Y-%m')
                    site_data_collector[site_name][(month_str, 'Projected Qual. Referrals (POF)')] = 0 
                    site_data_collector[site_name][(month_str, 'Projected ICFs Landed')] = 0.0

            for start_month_period in future_months: 
                total_psq_this_cohort = projection_cohorts.loc[start_month_period, 'Forecasted_PSQ']
                for site_name in all_sites:
                    site_prop = site_proportions_overall.get(site_name, 0) 
                    site_proj_pof_cohort = total_psq_this_cohort * site_prop # Site's POF allocation
                    
                    month_str_cohort_start = start_month_period.strftime('%Y-%m')
                    site_data_collector[site_name][(month_str_cohort_start, 'Projected Qual. Referrals (POF)')] += round(site_proj_pof_cohort)
                    
                    # Get site-specific rates if available, else use overall projection rates
                    site_perf_row = site_performance_data[site_performance_data['Site'] == site_name] if not site_performance_data.empty and site_name in site_performance_data['Site'].values else pd.DataFrame()
                    
                    # Chain site-specific or fallback rates for this site
                    current_site_proj_count = site_proj_pof_cohort
                    
                    for i_seg, (stage_from_seg, stage_to_seg) in enumerate(projection_segments_for_lag):
                        # Site specific rate names (e.g. "Site POF -> PSA %")
                        site_rate_key = f"Site {stage_from_seg.split(' ')[0]} -> {stage_to_seg.split(' ')[0]} %" # Heuristic for key name like "Site POF -> PSA %"
                        if stage_from_seg == "Passed Online Form" and stage_to_seg == "Pre-Screening Activities": site_rate_key = 'Site POF -> PSA %'
                        elif stage_from_seg == "Pre-Screening Activities" and stage_to_seg == "Sent To Site": site_rate_key = 'Site PSA -> StS %'
                        elif stage_from_seg == "Sent To Site" and stage_to_seg == "Appointment Scheduled": site_rate_key = 'Site StS -> Appt %'
                        elif stage_from_seg == "Appointment Scheduled" and stage_to_seg == "Signed ICF": site_rate_key = 'Site Appt -> ICF %'
                        # This mapping is fragile; ideally site_performance_data has exact keys like manual_proj_conv_rates_sidebar
                        # For now, let's assume the keys from calculate_site_metrics are 'Site POF -> PSA %', etc.
                        
                        overall_rate_key = f"{stage_from_seg} -> {stage_to_seg}"
                        
                        rate_to_use = final_projection_conv_rates.get(overall_rate_key, 0.0) # Default to overall rate

                        if not site_perf_row.empty and site_rate_key in site_perf_row.columns:
                            site_specific_rate_val = site_perf_row[site_rate_key].iloc[0]
                            if pd.notna(site_specific_rate_val):
                                rate_to_use = site_specific_rate_val
                        
                        current_site_proj_count *= rate_to_use
                        if stage_to_seg == "Signed ICF": # This is the generated ICF for the site
                            break # Stop chaining after ICF
                    
                    site_proj_icfs_generated_this_cohort = current_site_proj_count
                    
                    # Apply lag (using overall avg_actual_lag_days_for_display)
                    if site_proj_icfs_generated_this_cohort > 0:
                        full_lag_m_site = int(np.floor(current_lag_days_to_use / days_in_avg_month))
                        remain_lag_days_comp_site = current_lag_days_to_use - (full_lag_m_site * days_in_avg_month)
                        frac_next_m_site = remain_lag_days_comp_site / days_in_avg_month; frac_curr_m_site = 1.0 - frac_next_m_site
                        icfs_m1_site = site_proj_icfs_generated_this_cohort * frac_curr_m_site; icfs_m2_site = site_proj_icfs_generated_this_cohort * frac_next_m_site
                        land_m1_p_site = start_month_period + full_lag_m_site; land_m2_p_site = land_m1_p_site + 1
                        if land_m1_p_site in future_months: site_data_collector[site_name][(land_m1_p_site.strftime('%Y-%m'), 'Projected ICFs Landed')] += icfs_m1_site
                        if land_m2_p_site in future_months: site_data_collector[site_name][(land_m2_p_site.strftime('%Y-%m'), 'Projected ICFs Landed')] += icfs_m2_site
            
            if site_data_collector:
                site_level_projections_df_temp = pd.DataFrame.from_dict(site_data_collector, orient='index')
                if not site_level_projections_df_temp.empty:
                    site_level_projections_df_temp.columns = pd.MultiIndex.from_tuples(site_level_projections_df_temp.columns, names=['Month', 'Metric'])
                    for m_period_fmt in future_months.strftime('%Y-%m'): 
                        if (m_period_fmt, 'Projected ICFs Landed') in site_level_projections_df_temp.columns: site_level_projections_df_temp[(m_period_fmt, 'Projected ICFs Landed')] = site_level_projections_df_temp[(m_period_fmt, 'Projected ICFs Landed')].round(0).astype(int)
                        if (m_period_fmt, 'Projected Qual. Referrals (POF)') in site_level_projections_df_temp.columns: site_level_projections_df_temp[(m_period_fmt, 'Projected Qual. Referrals (POF)')] = site_level_projections_df_temp[(m_period_fmt, 'Projected Qual. Referrals (POF)')].astype(int)
                    site_level_projections_df = site_level_projections_df_temp.sort_index(axis=1, level=[0,1]) 
                    if not site_level_projections_df.empty:
                        numeric_cols_site_level = []
                        for col_tuple in site_level_projections_df.columns:
                            if pd.api.types.is_numeric_dtype(site_level_projections_df[col_tuple]): numeric_cols_site_level.append(col_tuple)
                        if numeric_cols_site_level:
                            total_row_values = site_level_projections_df[numeric_cols_site_level].sum(axis=0)
                            total_row_df = pd.DataFrame([total_row_values], index=["Grand Total"])
                            site_level_projections_df = pd.concat([site_level_projections_df, total_row_df])
        return display_df, avg_actual_lag_days_for_display, lpi_date_str, ads_off_date_str, site_level_projections_df, lag_calculation_method_message
    except Exception as e: 
        st.error(f"Projection calc error (main or site-level): {e}"); st.exception(e)
        return default_return_tuple[0], default_return_tuple[1], default_return_tuple[2], default_return_tuple[3], default_return_tuple[4], f"Error: {e}"


# --- Streamlit UI ---
if 'data_processed_successfully' not in st.session_state: st.session_state.data_processed_successfully = False
if 'referral_data_processed' not in st.session_state: st.session_state.referral_data_processed = None
if 'funnel_definition' not in st.session_state: st.session_state.funnel_definition = None
if 'ordered_stages' not in st.session_state: st.session_state.ordered_stages = None
if 'ts_col_map' not in st.session_state: st.session_state.ts_col_map = None
if 'site_metrics_calculated' not in st.session_state: st.session_state.site_metrics_calculated = pd.DataFrame()
if 'inter_stage_lags' not in st.session_state: st.session_state.inter_stage_lags = None 

ad_spend_input_dict = {}
weights_normalized = {}
proj_horizon_sidebar = 12
proj_spend_dict_sidebar = {}
proj_cpqr_dict_sidebar = {}
manual_proj_conv_rates_sidebar = {} 
use_rolling_flag_sidebar = False
rolling_window_months_sidebar = 3
goal_icf_count_sidebar = 100 

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"], key="referral_uploader_main")
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (TSV)", type=["tsv"], key="funnel_uploader_main") 
    st.divider()
    with st.expander("Historical Ad Spend"):
        st.info("Enter **historical** spend for past months found in data.")
        spend_month_str_1 = st.text_input("Month 1 (YYYY-MM)", "2025-02", key="h_spend_m1_str")
        spend_val_1 = st.number_input(f"Spend {spend_month_str_1}", value=45000.0, step=1000.0, format="%.2f", key="h_spend_v1")
        spend_month_str_2 = st.text_input("Month 2 (YYYY-MM)", "2025-03", key="h_spend_m2_str")
        spend_val_2 = st.number_input(f"Spend {spend_month_str_2}", value=60000.0, step=1000.0, format="%.2f", key="h_spend_v2")
        try: ad_spend_input_dict[pd.Period(spend_month_str_1, freq='M')] = spend_val_1
        except Exception: pass 
        try: ad_spend_input_dict[pd.Period(spend_month_str_2, freq='M')] = spend_val_2
        except Exception: pass
        st.caption("Ad Spend input method needs improvement.")
    st.divider()
    with st.expander("Site Scoring Weights"):
        weights_input_local = {} 
        # --- Ensure these keys match metric names in site_metrics_df from calculate_site_metrics ---
        weights_input_local["Qual -> ICF %"] = st.slider("Qual (POF) -> ICF %", 0, 100, 20, key='w_qicf') 
        weights_input_local["Avg TTC (Days)"] = st.slider("Avg Time to Contact", 0, 100, 25, key='w_ttc') 
        weights_input_local["Avg Funnel Movement Steps"] = st.slider("Avg Funnel Movement Steps", 0, 100, 5, key='w_fms') 
        weights_input_local["Site Screen Fail %"] = st.slider("Site Screen Fail %", 0, 100, 5, key='w_sfr') 
        # Using the more granular Site StS -> Appt % and Site Appt -> ICF % for scoring if desired
        weights_input_local["Site StS -> Appt %"] = st.slider("Site StS -> Appt Sched %", 0, 100, 30, key='w_sa_site') 
        weights_input_local["Site Appt -> ICF %"] = st.slider("Site Appt Sched -> ICF %", 0, 100, 15, key='w_ai_site') 
        weights_input_local["Lag Qual -> ICF (Days)"] = st.slider("Lag Qual (POF) -> ICF (Days)", 0, 100, 0, key='w_lagqicf') 
        
        total_weight_input_local = sum(abs(w) for w in weights_input_local.values()) 
        if total_weight_input_local > 0: weights_normalized = {k: v / total_weight_input_local for k, v in weights_input_local.items()}
        else: weights_normalized = {k: 0 for k in weights_input_local} 
        st.caption(f"Weights normalized. Lower is better for TTC, Screen Fail %, Lag.")
    st.divider()
    with st.expander("Projection Assumptions", expanded=True): 
        proj_horizon_sidebar = st.number_input("Projection Horizon (Months)", min_value=1, max_value=36, value=proj_horizon_sidebar, step=1, key='proj_horizon_widget')
        goal_icf_count_sidebar = st.number_input("Goal Total ICFs", min_value=1, value=goal_icf_count_sidebar, step=1, key='goal_icf_input') 
        _proj_start_month_ui_editor = pd.Period(datetime.now(), freq='M') + 1 
        if st.session_state.data_processed_successfully and st.session_state.referral_data_processed is not None and \
           not st.session_state.referral_data_processed.empty and "Submission_Month" in st.session_state.referral_data_processed.columns:
            last_hist_month_for_ui_editor = st.session_state.referral_data_processed["Submission_Month"].max()
            if pd.notna(last_hist_month_for_ui_editor): _proj_start_month_ui_editor = last_hist_month_for_ui_editor + 1
        proj_horizon_editor = proj_horizon_sidebar if proj_horizon_sidebar > 0 else 1
        future_months_ui_for_editor = pd.period_range(start=_proj_start_month_ui_editor, periods=proj_horizon_editor, freq='M')
        st.write("Future Monthly Ad Spend:")
        spend_df_for_editor = pd.DataFrame({'Month': future_months_ui_for_editor.strftime('%Y-%m'), 'Planned_Spend': [20000.0] * proj_horizon_editor }) 
        edited_spend_df = st.data_editor(spend_df_for_editor, key='proj_spend_editor_v4', use_container_width=True, num_rows="fixed") 
        proj_spend_dict_sidebar = {m_init: 0.0 for m_init in future_months_ui_for_editor} 
        if 'Month' in edited_spend_df.columns and 'Planned_Spend' in edited_spend_df.columns:
             for index, row in edited_spend_df.iterrows():
                 try:
                     month_str = str(row['Month']).strip(); planned_spend_val = float(row['Planned_Spend']) 
                     month_period = pd.Period(month_str, freq='M') 
                     if month_period in proj_spend_dict_sidebar: proj_spend_dict_sidebar[month_period] = planned_spend_val 
                 except Exception as e: pass 
        st.write("Assumed CPQR ($) per Month:")
        default_cpqr_value = 120.0
        cpqr_df_for_editor = pd.DataFrame({'Month': future_months_ui_for_editor.strftime('%Y-%m'), 'Assumed_CPQR': [default_cpqr_value] * proj_horizon_editor })
        edited_cpqr_df = st.data_editor(cpqr_df_for_editor, key='proj_cpqr_editor_v4', use_container_width=True, num_rows="fixed") 
        proj_cpqr_dict_sidebar = {m_init: default_cpqr_value for m_init in future_months_ui_for_editor} 
        if 'Month' in edited_cpqr_df.columns and 'Assumed_CPQR' in edited_cpqr_df.columns:
            for index, row in edited_cpqr_df.iterrows():
                try:
                    month_str = str(row['Month']).strip(); cpqr_val = float(row['Assumed_CPQR'])
                    month_period = pd.Period(month_str, freq='M')
                    if cpqr_val <=0: cpqr_val = default_cpqr_value 
                    if month_period in proj_cpqr_dict_sidebar: proj_cpqr_dict_sidebar[month_period] = cpqr_val
                except Exception as e: pass
        st.write("Conversion Rate Assumption:")
        rate_assumption_method_sidebar = st.radio( "Use Rates Based On:", ('Manual Input Below', 'Rolling Historical Average'), key='rate_method', horizontal=True )
        manual_proj_conv_rates_sidebar = {} 
        cols_rate = st.columns(2)
        with cols_rate[0]:
             manual_proj_conv_rates_sidebar["Passed Online Form -> Pre-Screening Activities"] = st.slider("Manual: Qual (POF) -> PreScreen %", 0.0, 100.0, 100.0, step=0.1, format="%.1f%%", key='cr_qps') / 100.0
             manual_proj_conv_rates_sidebar["Pre-Screening Activities -> Sent To Site"] = st.slider("Manual: PreScreen -> StS %", 0.0, 100.0, 17.0, step=0.1, format="%.1f%%", key='cr_pssts') / 100.0
        with cols_rate[1]:
             manual_proj_conv_rates_sidebar["Sent To Site -> Appointment Scheduled"] = st.slider("Manual: StS -> Appt %", 0.0, 100.0, 33.0, step=0.1, format="%.1f%%", key='cr_sa') / 100.0
             manual_proj_conv_rates_sidebar["Appointment Scheduled -> Signed ICF"] = st.slider("Manual: Appt -> ICF %", 0.0, 100.0, 35.0, step=0.1, format="%.1f%%", key='cr_ai') / 100.0
        use_rolling_flag_sidebar = (rate_assumption_method_sidebar == 'Rolling Historical Average')
        if use_rolling_flag_sidebar:
            rolling_window_months_sidebar = st.selectbox("Select Rolling Window (Months):", [1, 3, 6], index=1, key='rolling_window') 
            if st.session_state.data_processed_successfully and st.session_state.referral_data_processed is not None and \
               st.session_state.ordered_stages is not None and st.session_state.ts_col_map is not None:
                determine_effective_projection_rates(
                    st.session_state.referral_data_processed, 
                    st.session_state.ordered_stages, 
                    st.session_state.ts_col_map, 
                    rate_assumption_method_sidebar, 
                    rolling_window_months_sidebar, 
                    manual_proj_conv_rates_sidebar,
                    st.session_state.get('inter_stage_lags', {}), 
                    sidebar_display_area=st.sidebar
                )
            else: st.sidebar.caption("Upload data to view calculated rolling rates.")
        else: rolling_window_months_sidebar = 0; st.caption("Using manually input rates above.")

# --- Main App Logic & Display ---
if uploaded_referral_file is not None and uploaded_funnel_def_file is not None:
    if not st.session_state.data_processed_successfully: 
        funnel_definition, ordered_stages, ts_col_map = parse_funnel_definition(uploaded_funnel_def_file)
        if funnel_definition and ordered_stages and ts_col_map: 
            st.session_state.funnel_definition = funnel_definition
            st.session_state.ordered_stages = ordered_stages
            st.session_state.ts_col_map = ts_col_map
            try:
                 bytes_data = uploaded_referral_file.getvalue()
                 try: decoded_data = bytes_data.decode("utf-8")
                 except UnicodeDecodeError: decoded_data = bytes_data.decode("latin-1") 
                 stringio = io.StringIO(decoded_data)
                 try:
                      referrals_raw_df = pd.read_csv(stringio, sep=',', header=0, on_bad_lines='warn', low_memory=False) 
                      st.session_state.referral_data_processed = preprocess_referral_data(referrals_raw_df, funnel_definition, ordered_stages, ts_col_map)
                      if st.session_state.referral_data_processed is not None and not st.session_state.referral_data_processed.empty:
                           st.session_state.data_processed_successfully = True 
                           st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(
                               st.session_state.referral_data_processed, 
                               st.session_state.ordered_stages, st.session_state.ts_col_map)
                      else: st.session_state.data_processed_successfully = False
                 except Exception as read_err: st.error(f"Error reading referral file: {read_err}"); st.exception(read_err)
            except Exception as e: st.error(f"Error loading data: {e}"); st.exception(e)

if st.session_state.data_processed_successfully:
    referral_data_processed = st.session_state.referral_data_processed 
    funnel_definition = st.session_state.funnel_definition
    ordered_stages = st.session_state.ordered_stages
    ts_col_map = st.session_state.ts_col_map
    inter_stage_lags_data = st.session_state.get('inter_stage_lags', {}) 
    if "success_message_shown" not in st.session_state and referral_data_processed is not None:
        st.success("Data loaded and preprocessed successfully!")
        st.session_state.success_message_shown = True
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📅 Monthly ProForma", "🏆 Site Performance", "📈 Projections"])
    with tab1:
        st.header("Monthly ProForma (Historical Cohorts)")
        if referral_data_processed is not None and ordered_stages is not None and ts_col_map is not None and ad_spend_input_dict is not None:
            proforma_df = calculate_proforma_metrics(referral_data_processed, ordered_stages, ts_col_map, ad_spend_input_dict) 
            if not proforma_df.empty:
                proforma_display = proforma_df.transpose(); proforma_display.columns = [str(col) for col in proforma_display.columns] 
                format_dict = {}; 
                for idx in proforma_display.index:
                     if 'Cost' in idx or 'Spend' in idx: format_dict[idx] = "${:,.2f}"
                     elif '%' in idx: format_dict[idx] = "{:.1%}"
                     else: format_dict[idx] = "{:,.0f}" 
                st.dataframe(proforma_display.style.format(format_dict, na_rep='-'))
                try:
                     csv = proforma_df.reset_index().to_csv(index=False).encode('utf-8')
                     st.download_button(label="Download ProForma Data", data=csv, file_name='monthly_proforma.csv', mime='text/csv', key='dl_proforma')
                except Exception as e: st.warning(f"Download button error: {e}")
            else: st.warning("Could not generate ProForma table.")
        else: st.warning("ProForma cannot be calculated until data is loaded and processed.")
    with tab2:
        st.header("Site Performance Ranking")
        if referral_data_processed is not None and ordered_stages is not None and ts_col_map is not None and weights_normalized is not None:
            # Ensure site_metrics_calculated uses the updated calculate_site_metrics
            site_metrics_calculated = calculate_site_metrics(referral_data_processed, ordered_stages, ts_col_map) 
            
            if not site_metrics_calculated.empty:
                st.session_state.site_metrics_calculated = site_metrics_calculated 
                ranked_sites_df = score_sites(site_metrics_calculated, weights_normalized) 
                st.subheader("Site Ranking")
                # --- MODIFIED: Update display_cols_sites to include new granular rates if desired for display ---
                display_cols_sites = ['Site', 'Score', 'Grade', 'Total Qualified', 
                                      'Site Count POF', 'Site Count PSA', 'Site Count StS', 'Site Count Appt', 'Site Count ICF',
                                      'Qual -> ICF %', # This is site POF -> ICF %
                                      'Site POF -> PSA %', 'Site PSA -> StS %', 'Site StS -> Appt %', 'Site Appt -> ICF %',
                                      'Avg TTC (Days)', 'Avg Funnel Movement Steps', 
                                      'Site Screen Fail %', 'Lag Qual -> ICF (Days)']
                # Filter down to only columns that actually exist in ranked_sites_df
                display_cols_sites_exist = [col for col in display_cols_sites if col in ranked_sites_df.columns]
                 
                final_ranked_display = ranked_sites_df[display_cols_sites_exist].copy()

                if not final_ranked_display.empty:
                    if 'Score' in final_ranked_display.columns: final_ranked_display['Score'] = final_ranked_display['Score'].round(1)
                    
                    # Update formatting logic if new column names are used
                    percent_cols_site_tab = [c for c in final_ranked_display.columns if '%' in c]
                    lag_cols_site_tab = [c for c in final_ranked_display.columns if 'Lag' in c or 'TTC' in c]
                    step_cols_site_tab = [c for c in final_ranked_display.columns if 'Steps' in c]
                    count_cols_site_tab = [c for c in final_ranked_display.columns if 'Count' in c or 'Qualified' in c] 
                    
                    for col in percent_cols_site_tab: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '-')
                    for col in lag_cols_site_tab: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
                    for col in step_cols_site_tab: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
                    for col in count_cols_site_tab: 
                        if col in final_ranked_display.columns: final_ranked_display[col] = final_ranked_display[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x==x else '-') 
                    
                    st.dataframe(final_ranked_display.style.format(na_rep='-'))
                    try:
                         csv_sites = final_ranked_display.to_csv(index=False).encode('utf-8')
                         st.download_button(label="Download Site Ranking", data=csv_sites, file_name='site_ranking.csv', mime='text/csv', key='dl_sites')
                    except Exception as e: st.warning(f"Download button error: {e}")
                else: st.warning("Site ranking display table is empty after filtering columns.")
            else: st.warning("Could not calculate site metrics."); st.session_state.site_metrics_calculated = pd.DataFrame() 
        else: st.warning("Site performance cannot be calculated until data is loaded and processed.")
    with tab3:
        st.header("Projections")
        _effective_projection_conv_rates_tab3, _method_desc_for_display_tab3 = determine_effective_projection_rates(
            referral_data_processed, ordered_stages, ts_col_map, 
            rate_assumption_method_sidebar, rolling_window_months_sidebar, 
            manual_proj_conv_rates_sidebar, inter_stage_lags_data, 
            sidebar_display_area=None)
        st.caption(f"**Projection Using: {_method_desc_for_display_tab3} Conversion Rates**")
        if "Rolling" in _method_desc_for_display_tab3 and not any(s in _method_desc_for_display_tab3 for s in ["Failed", "No History", "Error"]):
            if isinstance(_effective_projection_conv_rates_tab3, dict) and _effective_projection_conv_rates_tab3:
                st.markdown("---"); st.write("Effective Rolling Rates Applied for this Projection (Adj. & Matured):")
                for key, val in _effective_projection_conv_rates_tab3.items():
                     if key in manual_proj_conv_rates_sidebar: st.text(f"- {key}: {val*100:.1f}%")
        st.markdown("---")
        with st.expander("View Calculated Average Inter-Stage Lags & Maturity Periods Used"):
            if inter_stage_lags_data:
                lag_df_list = []
                temp_maturity_periods_display = {}
                display_maturity_lag_multiplier = 1.5 
                display_min_effective_maturity = 20
                display_default_maturity = 45
                if inter_stage_lags_data:
                    for r_key_disp in manual_proj_conv_rates_sidebar.keys(): 
                        avg_lag_disp = inter_stage_lags_data.get(r_key_disp)
                        if pd.notna(avg_lag_disp) and avg_lag_disp > 0:
                            calc_mat_disp = round(display_maturity_lag_multiplier * avg_lag_disp)
                            temp_maturity_periods_display[r_key_disp] = max(calc_mat_disp, display_min_effective_maturity)
                        else: temp_maturity_periods_display[r_key_disp] = display_default_maturity
                else:
                     for r_key_disp in manual_proj_conv_rates_sidebar.keys(): temp_maturity_periods_display[r_key_disp] = display_default_maturity
                for key, val in inter_stage_lags_data.items():
                    maturity_p_display = temp_maturity_periods_display.get(key, "N/A (Not a projection rate segment)")
                    lag_df_list.append({
                        'Stage Transition': key, 
                        'Avg Lag (Days)': f"{val:.1f}" if pd.notna(val) else "N/A",
                        'Implied Maturity Used (Days)': f"{maturity_p_display}" if isinstance(maturity_p_display, (int, float)) else maturity_p_display
                        })
                if lag_df_list: st.table(pd.DataFrame(lag_df_list))
                else: st.caption("No inter-stage lags calculated or available.")
            else: st.caption("Inter-stage lags have not been calculated.")
        st.markdown("---")

        projection_inputs = {
            'horizon': proj_horizon_sidebar, 'spend_dict': proj_spend_dict_sidebar, 
            'cpqr_dict': proj_cpqr_dict_sidebar, 'final_conv_rates': _effective_projection_conv_rates_tab3, 
            'goal_icf': goal_icf_count_sidebar,
            'site_performance_data': st.session_state.get('site_metrics_calculated', pd.DataFrame()), # This now contains more granular rates
            'inter_stage_lags': inter_stage_lags_data}
        
        projection_results_df, avg_lag_days_used_for_proj, lpi_date_str_proj, ads_off_date_str_proj, site_level_projections_df, lag_calc_msg_from_proj = calculate_projections(
            referral_data_processed, ordered_stages, ts_col_map, projection_inputs)
        
        st.markdown("---")
        col1_info, col2_info, col3_info = st.columns(3)
        with col1_info: st.metric(label="Goal Total ICFs", value=f"{goal_icf_count_sidebar:,}")
        with col2_info: st.metric(label="Estimated LPI Date", value=lpi_date_str_proj) 
        with col3_info: st.metric(label="Estimated Ads Off Date", value=ads_off_date_str_proj) 
        if pd.notna(avg_lag_days_used_for_proj): st.caption(f"Lag applied in projections: **{avg_lag_days_used_for_proj:.1f} days**. ({lag_calc_msg_from_proj})")
        else: st.caption(f"Lag could not be determined. ({lag_calc_msg_from_proj})")
        st.markdown("---")
        if projection_results_df is not None and not projection_results_df.empty and isinstance(projection_results_df, pd.DataFrame): 
            st.subheader("Projected Monthly ICFs & Cohort CPICF")
            display_cols_proj = ['Forecasted_Ad_Spend', 'Forecasted_Qual_Referrals', 'Projected_ICF_Landed', 'Projected_CPICF_Cohort_Source']
            results_display = projection_results_df[[col for col in display_cols_proj if col in projection_results_df.columns]].copy() 
            if not results_display.empty:
                results_display.index = results_display.index.strftime('%Y-%m') 
                if 'Forecasted_Ad_Spend' in results_display: results_display['Forecasted_Ad_Spend'] = results_display['Forecasted_Ad_Spend'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '-')
                if 'Projected_CPICF_Cohort_Source' in results_display: results_display['Projected_CPICF_Cohort_Source'] = results_display['Projected_CPICF_Cohort_Source'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '-')
                if 'Forecasted_Qual_Referrals' in results_display: results_display['Forecasted_Qual_Referrals'] = results_display['Forecasted_Qual_Referrals'].fillna(0).astype(int).map('{:,}'.format) 
                if 'Projected_ICF_Landed' in results_display: results_display['Projected_ICF_Landed'] = results_display['Projected_ICF_Landed'].fillna(0).astype(int).map('{:,}'.format) 
                st.dataframe(results_display.style.format(na_rep='-'))
                if 'Projected_ICF_Landed' in projection_results_df.columns:
                     st.subheader("Projected ICFs Landed Over Time")
                     chart_data = projection_results_df[['Projected_ICF_Landed']].copy()
                     if isinstance(chart_data.index, pd.PeriodIndex): chart_data.index = chart_data.index.to_timestamp() 
                     chart_data['Projected_ICF_Landed'] = pd.to_numeric(chart_data['Projected_ICF_Landed'], errors='coerce').fillna(0)
                     st.line_chart(chart_data)
                try:
                     csv_proj = results_display.reset_index().to_csv(index=False).encode('utf-8')
                     st.download_button(label="Download Projection Data", data=csv_proj, file_name='projection.csv', mime='text/csv', key='dl_proj')
                except Exception as e: st.warning(f"Download button error: {e}")
            else: st.warning("Projection results table is empty after selecting columns.")
        else: st.warning("Could not calculate projections.")
        st.markdown("---")
        st.subheader("Site-Level Monthly Projections (Editable)")
        if site_level_projections_df is not None and not site_level_projections_df.empty:
            display_site_df_main = site_level_projections_df.copy()
            if display_site_df_main.index.name == 'Site': display_site_df_main.reset_index(inplace=True)
            editable_sites_df = display_site_df_main[display_site_df_main['Site'] != 'Grand Total'] if 'Site' in display_site_df_main.columns else display_site_df_main[display_site_df_main.index != 'Grand Total']
            total_row_df_display = display_site_df_main[display_site_df_main['Site'] == 'Grand Total'] if 'Site' in display_site_df_main.columns else (display_site_df_main.loc[['Grand Total']] if 'Grand Total' in display_site_df_main.index else pd.DataFrame())
            edited_site_level_df = st.data_editor(editable_sites_df, use_container_width=True, key="site_level_editor", num_rows="dynamic")
            if not total_row_df_display.empty:
                st.caption("Totals (based on initial calculation, not live edits from above table):")
                st.dataframe(total_row_df_display.style.format("{:,.0f}", na_rep='0'), use_container_width=True)
            try:
                csv_site_proj = edited_site_level_df.to_csv(index=False).encode('utf-8') 
                st.download_button(label="Download Edited Site Projections", data=csv_site_proj, file_name='edited_site_level_projections.csv', mime='text/csv', key='dl_edited_site_proj_main')
            except Exception as e: st.warning(f"Download button error for site projections: {e}")
        else: st.info("Site-level projection data is not available or is empty.")
elif not uploaded_referral_file or not uploaded_funnel_def_file:
    st.info("👋 Welcome! Please upload both the Referral Data (CSV) and Funnel Definition (TSV) files using the sidebar to begin.")