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

# --- Constants for Stage Names (example, adjust based on your actual funnel definition) ---
STAGE_PASSED_ONLINE_FORM = "Passed Online Form"
STAGE_PRE_SCREENING_ACTIVITIES = "Pre-Screening Activities"
STAGE_SENT_TO_SITE = "Sent To Site"
STAGE_APPOINTMENT_SCHEDULED = "Appointment Scheduled"
STAGE_SIGNED_ICF = "Signed ICF"
STAGE_SCREEN_FAILED = "Screen Failed" # Example, if used in metrics

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
            if stage_name not in statuses: statuses.append(stage_name) # Ensure stage name itself is a status
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
                except AttributeError: pass # Handle NaT if dt_obj couldn't convert
    try: parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min) # Robust sort for NaT
    except TypeError as e: pass # Catch errors if sorting mixed types not handled by datetime.min
    return parsed_events

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}; status_to_stage_map = {}
    if not funnel_def: return pd.Series(timestamps) # Should not happen if funnel_def is validated
    for stage, statuses in funnel_def.items(): # funnel_def is {'StageName': [status1, status2]}
        for status in statuses: status_to_stage_map[status] = stage
    all_events = []; stage_hist = row.get(parsed_stage_history_col, []); status_hist = row.get(parsed_status_history_col, [])
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist if isinstance(name, str)]) # Ensure name is string
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist if isinstance(name, str)])
    try: all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError as e: pass
    for event_name, event_dt in all_events:
        if pd.isna(event_dt): continue
        event_stage = None;
        if event_name in ordered_stgs: event_stage = event_name # Direct match to a defined stage name
        elif event_name in status_to_stage_map: event_stage = status_to_stage_map[event_name] # Match to a status within a stage
        if event_stage and event_stage in ordered_stgs: # Ensure the derived stage is one we're tracking
            ts_col_name = ts_col_mapping.get(event_stage)
            if ts_col_name and pd.isna(timestamps[ts_col_name]): # Only fill the first time it's reached
                timestamps[ts_col_name] = event_dt
    return pd.Series(timestamps, dtype='datetime64[ns]')

@st.cache_data
def preprocess_referral_data(_df_raw, funnel_def, ordered_stages, ts_col_map):
    if _df_raw is None or funnel_def is None or ordered_stages is None or ts_col_map is None: return None
    df = _df_raw.copy(); submitted_on_col = None
    if "Submitted On" in df.columns: submitted_on_col = "Submitted On"
    elif "Referral Date" in df.columns: df.rename(columns={"Referral Date": "Submitted On"}, inplace=True); submitted_on_col = "Submitted On"
    else:
        if "Submitted On" not in df.columns: st.error("Missing 'Submitted On'/'Referral Date'."); return None
        else: submitted_on_col = "Submitted On" # Should not be reached if previous check passes
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    initial_rows = len(df); df.dropna(subset=["Submitted On_DT"], inplace=True); rows_dropped = initial_rows - len(df)
    if rows_dropped > 0: st.warning(f"Dropped {rows_dropped} rows due to unparseable 'Submitted On' date.")
    if df.empty: st.error("No valid data remaining after date parsing."); return None
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    history_cols_to_parse = ['Lead Stage History', 'Lead Status History']; parsed_cols = {}
    for col_name in history_cols_to_parse:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string); parsed_cols[col_name] = parsed_col_name
        else: st.warning(f"History column '{col_name}' not found. Timestamps might be incomplete.")
    parsed_stage_hist_col = parsed_cols.get('Lead Stage History'); parsed_status_hist_col = parsed_cols.get('Lead Status History')
    if not parsed_stage_hist_col and not parsed_status_hist_col:
        if 'Lead Stage History' not in df.columns and 'Lead Status History' not in df.columns: st.error("Neither 'Lead Stage History' nor 'Lead Status History' column found. Cannot determine stage progression.")
        else: st.error("History columns found but failed to parse into structured data.") # Should ideally not happen if parse_history_string is robust
        return None
    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist_col, parsed_status_hist_col, funnel_def, ordered_stages, ts_col_map), axis=1)
    old_ts_cols = [col for col in df.columns if col.startswith('TS_')]; df.drop(columns=old_ts_cols, inplace=True, errors='ignore') # Remove any pre-existing TS_ cols
    df = pd.concat([df, timestamp_cols_df], axis=1)
    for stage, ts_col in ts_col_map.items(): # Ensure all defined TS_ cols are datetime
         if ts_col in df.columns: df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    return df

def calculate_proforma_metrics(_processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    if _processed_df is None or _processed_df.empty: return pd.DataFrame()
    if not isinstance(monthly_ad_spend_input, dict): st.warning("ProForma: Ad spend input is not a dictionary."); return pd.DataFrame()
    if "Submission_Month" not in _processed_df.columns: st.error("ProForma: 'Submission_Month' column missing."); return pd.DataFrame()
    processed_df = _processed_df.copy()
    try:
        cohort_summary = processed_df.groupby("Submission_Month").size().reset_index(name="Total Qualified Referrals_Calc") # Use a distinct name
        cohort_summary = cohort_summary.set_index("Submission_Month")
        cohort_summary["Ad Spend"] = cohort_summary.index.map(monthly_ad_spend_input).fillna(0)
        reached_stage_cols_map = {}
        ts_pof_col_name = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)

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
        
        base_count_col_name_in_summary = None
        if ts_pof_col_name and reached_stage_cols_map.get(STAGE_PASSED_ONLINE_FORM) in cohort_summary.columns:
            cohort_summary.rename(columns={reached_stage_cols_map[STAGE_PASSED_ONLINE_FORM]: "Pre-Screener Qualified"}, inplace=True, errors='ignore')
            base_count_col_name_in_summary = "Pre-Screener Qualified"
        elif "Total Qualified Referrals_Calc" in cohort_summary.columns:
            cohort_summary.rename(columns={"Total Qualified Referrals_Calc": "Pre-Screener Qualified"}, inplace=True, errors='ignore')
            base_count_col_name_in_summary = "Pre-Screener Qualified"
        else:
            st.warning("ProForma: Could not determine base 'Pre-Screener Qualified' count column.")
            return pd.DataFrame()

        proforma_metrics = pd.DataFrame(index=cohort_summary.index)
        if base_count_col_name_in_summary and base_count_col_name_in_summary in cohort_summary.columns:
            proforma_metrics["Ad Spend"] = cohort_summary["Ad Spend"]
            proforma_metrics["Pre-Screener Qualified"] = cohort_summary[base_count_col_name_in_summary]
            proforma_metrics["Cost per Qualified Pre-screen"] = (cohort_summary["Ad Spend"] / cohort_summary[base_count_col_name_in_summary].replace(0, np.nan)).round(2)
            
            for stage_orig, reached_col_mapped in reached_stage_cols_map.items():
                 metric_name = f"Total {stage_orig}"
                 if stage_orig == STAGE_PASSED_ONLINE_FORM : metric_name = "Pre-Screener Qualified"
                 if reached_col_mapped in cohort_summary.columns and metric_name not in proforma_metrics.columns:
                     proforma_metrics[metric_name] = cohort_summary[reached_col_mapped]

            sts_col_actual = reached_stage_cols_map.get(STAGE_SENT_TO_SITE)
            appt_col_actual = reached_stage_cols_map.get(STAGE_APPOINTMENT_SCHEDULED)
            icf_col_actual = reached_stage_cols_map.get(STAGE_SIGNED_ICF)

            if sts_col_actual in cohort_summary.columns: proforma_metrics["Qualified to StS %"] = (cohort_summary[sts_col_actual] / cohort_summary[base_count_col_name_in_summary].replace(0, np.nan))
            if sts_col_actual in cohort_summary.columns and appt_col_actual in cohort_summary.columns: proforma_metrics["StS to Appt Sched %"] = (cohort_summary[appt_col_actual] / cohort_summary[sts_col_actual].replace(0, np.nan))
            if appt_col_actual in cohort_summary.columns and icf_col_actual in cohort_summary.columns: proforma_metrics["Appt Sched to ICF %"] = (cohort_summary[icf_col_actual] / cohort_summary[appt_col_actual].replace(0, np.nan))
            
            if icf_col_actual in cohort_summary.columns:
                proforma_metrics["Qualified to ICF %"] = (cohort_summary[icf_col_actual] / cohort_summary[base_count_col_name_in_summary].replace(0, np.nan))
                proforma_metrics["Cost Per ICF"] = (cohort_summary["Ad Spend"] / cohort_summary[icf_col_actual].replace(0, np.nan)).round(2)
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
        else: 
            inter_stage_lags[f"{stage_from} -> {stage_to}"] = np.nan
    return inter_stage_lags

def calculate_site_metrics(_processed_df, ordered_stages, ts_col_map):
    if _processed_df is None or _processed_df.empty or 'Site' not in _processed_df.columns:
        return pd.DataFrame()

    processed_df = _processed_df.copy(); site_metrics_list = []
    ts_pof_col = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
    ts_psa_col = ts_col_map.get(STAGE_PRE_SCREENING_ACTIVITIES)
    ts_sts_col = ts_col_map.get(STAGE_SENT_TO_SITE)
    ts_appt_col = ts_col_map.get(STAGE_APPOINTMENT_SCHEDULED)
    ts_icf_col = ts_col_map.get(STAGE_SIGNED_ICF)
    ts_sf_col = ts_col_map.get(STAGE_SCREEN_FAILED)

    potential_ts_cols = [ts_pof_col, ts_psa_col, ts_sts_col, ts_appt_col, ts_icf_col, ts_sf_col]
    for col_name in potential_ts_cols:
        if col_name and col_name not in processed_df.columns: 
            processed_df[col_name] = pd.NaT
            processed_df[col_name] = pd.to_datetime(processed_df[col_name], errors='coerce')
    
    site_contact_attempt_statuses = ["Site Contact Attempt 1"] 
    post_sts_progress_stages = [STAGE_APPOINTMENT_SCHEDULED, STAGE_SIGNED_ICF, "Enrolled", STAGE_SCREEN_FAILED]

    projection_segments_for_site_lag = [
        (STAGE_PASSED_ONLINE_FORM, STAGE_PRE_SCREENING_ACTIVITIES),
        (STAGE_PRE_SCREENING_ACTIVITIES, STAGE_SENT_TO_SITE),
        (STAGE_SENT_TO_SITE, STAGE_APPOINTMENT_SCHEDULED),
        (STAGE_APPOINTMENT_SCHEDULED, STAGE_SIGNED_ICF)
    ]
    try:
        site_groups = processed_df.groupby('Site')
        for site_name, group in site_groups:
            metrics = {'Site': site_name}
            count_pof = group[ts_pof_col].notna().sum() if ts_pof_col and ts_pof_col in group else 0
            count_psa = group[ts_psa_col].notna().sum() if ts_psa_col and ts_psa_col in group else 0
            count_sts = group[ts_sts_col].notna().sum() if ts_sts_col and ts_sts_col in group else 0
            count_appt = group[ts_appt_col].notna().sum() if ts_appt_col and ts_appt_col in group else 0
            count_icf = group[ts_icf_col].notna().sum() if ts_icf_col and ts_icf_col in group else 0
            count_sf = group[ts_sf_col].notna().sum() if ts_sf_col and ts_sf_col in group else 0

            metrics['Total Qualified'] = count_pof
            metrics['PSA Count'] = count_psa
            metrics['StS Count'] = count_sts
            metrics['Appt Count'] = count_appt
            metrics['ICF Count'] = count_icf
            
            metrics['POF -> PSA %'] = (count_psa / count_pof) if count_pof > 0 else 0.0
            metrics['PSA -> StS %'] = (count_sts / count_psa) if count_psa > 0 else 0.0
            metrics['StS -> Appt %'] = (count_appt / count_sts) if count_sts > 0 else 0.0
            metrics['Appt -> ICF %'] = (count_icf / count_appt) if count_appt > 0 else 0.0
            metrics['Qual -> ICF %'] = (count_icf / count_pof) if count_pof > 0 else 0.0
            
            site_total_projection_lag = 0.0
            valid_lag_segments = 0
            for seg_from_name, seg_to_name in projection_segments_for_site_lag:
                ts_seg_from = ts_col_map.get(seg_from_name)
                ts_seg_to = ts_col_map.get(seg_to_name)
                if ts_seg_from and ts_seg_to and ts_seg_from in group.columns and ts_seg_to in group.columns:
                    segment_lag = calculate_avg_lag_generic(group, ts_seg_from, ts_seg_to)
                    if pd.notna(segment_lag):
                        site_total_projection_lag += segment_lag
                        valid_lag_segments +=1
                    else: site_total_projection_lag = np.nan; break 
                else: site_total_projection_lag = np.nan; break 
            
            if valid_lag_segments < len(projection_segments_for_site_lag):
                 site_total_projection_lag = np.nan
            metrics['Site Projection Lag (Days)'] = site_total_projection_lag
            metrics['Lag Qual -> ICF (Days)'] = calculate_avg_lag_generic(group, ts_pof_col, ts_icf_col) if ts_pof_col and ts_icf_col else np.nan
            
            ttc_times = []; funnel_movement_steps = []
            parsed_status_col_name = f"Parsed_Lead_Status_History"
            parsed_stage_col_name = f"Parsed_Lead_Stage_History"
            sent_to_site_group = group.dropna(subset=[ts_sts_col]) if ts_sts_col and ts_sts_col in group else pd.DataFrame()
            
            if not sent_to_site_group.empty and parsed_status_col_name in sent_to_site_group.columns and parsed_stage_col_name in sent_to_site_group.columns:
                for idx, row in sent_to_site_group.iterrows():
                    ts_sent = row[ts_sts_col]; first_contact_ts = pd.NaT
                    history_list_status = row.get(parsed_status_col_name, []) 
                    if history_list_status: 
                        for status_name, event_dt in history_list_status:
                            if status_name in site_contact_attempt_statuses and pd.notna(event_dt) and pd.notna(ts_sent) and event_dt >= ts_sent:
                                first_contact_ts = event_dt; break
                    if pd.notna(first_contact_ts) and pd.notna(ts_sent): 
                        time_diff = first_contact_ts - ts_sent
                        if time_diff >= pd.Timedelta(0): ttc_times.append(time_diff.total_seconds() / (60*60*24))
                                    
                    stages_reached_post_sts = set()
                    history_list_stage = row.get(parsed_stage_col_name, [])
                    if history_list_stage and pd.notna(ts_sent): 
                         for stage_name_hist, event_dt_hist in history_list_stage:
                             if stage_name_hist in post_sts_progress_stages and pd.notna(event_dt_hist) and event_dt_hist > ts_sent:
                                 stages_reached_post_sts.add(stage_name_hist)
                    funnel_movement_steps.append(len(stages_reached_post_sts))
            metrics['Avg TTC (Days)'] = np.mean(ttc_times) if ttc_times else np.nan
            metrics['Avg Funnel Movement Steps'] = np.mean(funnel_movement_steps) if funnel_movement_steps else 0.0
            metrics['Site Screen Fail %'] = (count_sf / count_icf) if count_icf > 0 else 0.0
            site_metrics_list.append(metrics)
        
        site_metrics_df_final = pd.DataFrame(site_metrics_list)
        return site_metrics_df_final
    except Exception as e:
        st.error(f"Error calculating site metrics: {e}"); st.exception(e)
        return pd.DataFrame()

def score_sites(_site_metrics_df, weights):
    if _site_metrics_df is None or _site_metrics_df.empty: return pd.DataFrame()
    try:
        site_metrics_df = _site_metrics_df.copy();
        if 'Site' not in site_metrics_df.columns:
             if site_metrics_df.index.name == 'Site': site_metrics_df = site_metrics_df.reset_index()
             else: st.error("Site Scoring: 'Site' column missing."); return pd.DataFrame()
        site_metrics_df_indexed = site_metrics_df.set_index('Site')
        metrics_to_scale = list(weights.keys())
        lower_is_better = ["Avg TTC (Days)", "Site Screen Fail %", "Lag Qual -> ICF (Days)", "Site Projection Lag (Days)"]
        scaled_metrics_data = site_metrics_df_indexed.reindex(columns=metrics_to_scale).copy()
        for col in metrics_to_scale: 
            if col not in scaled_metrics_data.columns: 
                scaled_metrics_data[col] = 0 if col not in lower_is_better else np.nan
            if col in lower_is_better:
                max_val = scaled_metrics_data[col].max(skipna=True);
                fill_val = (max_val + scaled_metrics_data[col].std(skipna=True)) if pd.notna(max_val) and max_val > 0 and pd.notna(scaled_metrics_data[col].std(skipna=True)) else 999
                scaled_metrics_data[col] = scaled_metrics_data[col].fillna(fill_val)
            else: 
                scaled_metrics_data[col] = scaled_metrics_data[col].fillna(0)

        scaled_metrics_display = pd.DataFrame(index=scaled_metrics_data.index)
        if not scaled_metrics_data.empty:
            for col in metrics_to_scale: 
                 if col in scaled_metrics_data.columns:
                     min_val = scaled_metrics_data[col].min(); max_val = scaled_metrics_data[col].max()
                     if min_val == max_val: 
                         scaled_metrics_display[col] = 0.5 
                     elif pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val) != 0 :
                         scaler = MinMaxScaler(); 
                         scaled_values = scaler.fit_transform(scaled_metrics_data[[col]]);
                         scaled_metrics_display[col] = scaled_values.flatten()
                     else: 
                         scaled_metrics_display[col] = 0.5 
                 else: 
                     scaled_metrics_display[col] = 0.5
            for col in lower_is_better:
                if col in scaled_metrics_display.columns:
                    scaled_metrics_display[col] = 1 - scaled_metrics_display[col]

        site_metrics_df_indexed['Score_Raw'] = 0.0; total_weight_applied = 0.0
        for metric, weight_value in weights.items(): 
             if metric in scaled_metrics_display.columns:
                 current_scaled_metric_series = scaled_metrics_display.get(metric) 
                 if current_scaled_metric_series is not None:
                     site_metrics_df_indexed['Score_Raw'] += current_scaled_metric_series.fillna(0.5) * weight_value
                 else: 
                     site_metrics_df_indexed['Score_Raw'] += 0.5 * weight_value 
                 total_weight_applied += abs(weight_value) 
        
        if total_weight_applied > 0:
            site_metrics_df_indexed['Score'] = (site_metrics_df_indexed['Score_Raw'] / total_weight_applied) * 100
        else: 
            site_metrics_df_indexed['Score'] = 0.0
        site_metrics_df_indexed['Score'] = site_metrics_df_indexed['Score'].fillna(0.0)

        if len(site_metrics_df_indexed) > 1:
            site_metrics_df_indexed['Score_Rank_Percentile'] = site_metrics_df_indexed['Score'].rank(pct=True)
            bins = [0, 0.10, 0.25, 0.60, 0.85, 1.0]; labels = ['F', 'D', 'C', 'B', 'A']
            try: site_metrics_df_indexed['Grade'] = pd.qcut(site_metrics_df_indexed['Score_Rank_Percentile'], q=bins, labels=labels, duplicates='drop')
            except ValueError: 
                 st.warning("Using fixed score ranges for grading (percentile method failed).")
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
            def assign_single_site_grade(score_value): # CORRECTED THIS FUNCTION
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
        if 'Score' in final_df_output.columns: final_df_output = final_df_output.sort_values('Score', ascending=False)
        return final_df_output
    except Exception as e:
        st.error(f"Error during Site Scoring: {e}"); st.exception(e)
        if _site_metrics_df is not None and not _site_metrics_df.empty: 
             if _site_metrics_df.index.name == 'Site' and 'Site' not in _site_metrics_df.columns: return _site_metrics_df.reset_index()
             return _site_metrics_df
        return pd.DataFrame()

def determine_effective_projection_rates(_processed_df, ordered_stages, ts_col_map,
                                          rate_method_sidebar, rolling_window_sidebar, manual_rates_sidebar,
                                          inter_stage_lags_for_maturity,
                                          sidebar_display_area=None):
    MIN_DENOMINATOR_FOR_RATE_CALC = 5; DEFAULT_MATURITY_DAYS = 45; MATURITY_LAG_MULTIPLIER = 1.5
    MIN_EFFECTIVE_MATURITY_DAYS = 20; MIN_TOTAL_DENOMINATOR_FOR_OVERALL_RATE = 20
    
    if ts_col_map is None:
        if sidebar_display_area: sidebar_display_area.error("TS Column Map not available for rate calculation.")
        return manual_rates_sidebar, "Manual (TS Col Map Missing)"

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
                MATURITY_PERIODS_DAYS[rate_key_for_lag] = max(calculated_maturity, MIN_EFFECTIVE_MATURITY_DAYS)
            else: MATURITY_PERIODS_DAYS[rate_key_for_lag] = DEFAULT_MATURITY_DAYS
    else: 
        for rate_key_for_lag in manual_rates_sidebar.keys(): MATURITY_PERIODS_DAYS[rate_key_for_lag] = DEFAULT_MATURITY_DAYS
        substitutions_made_log.append(f"Maturity: Inter-stage lags N/A or empty, used default {DEFAULT_MATURITY_DAYS}d for all.")

    try:
        if "Submission_Month" not in _processed_df.columns or _processed_df["Submission_Month"].dropna().empty:
            if sidebar_display_area: sidebar_display_area.warning("Not enough historical submission month data. Using manual rates.")
            return manual_rates_sidebar, "Manual (No Submission Month History)"
        
        hist_counts = _processed_df.groupby("Submission_Month").size().to_frame(name="Total_Submissions_Calc")
        reached_stage_cols_map_hist = {}
        for stage_name_iter in ordered_stages: 
            ts_col_iter = ts_col_map.get(stage_name_iter)
            if ts_col_iter and ts_col_iter in _processed_df.columns and pd.api.types.is_datetime64_any_dtype(_processed_df[ts_col_iter]):
                cleaned_stage_name_for_col = f"Reached_{stage_name_iter.replace(' ', '_').replace('(', '').replace(')', '')}"
                reached_stage_cols_map_hist[stage_name_iter] = cleaned_stage_name_for_col
                stage_monthly_counts = _processed_df.dropna(subset=[ts_col_iter]).groupby(_processed_df['Submission_Month']).size()
                hist_counts = hist_counts.join(stage_monthly_counts.rename(cleaned_stage_name_for_col), how='left')
        hist_counts = hist_counts.fillna(0)

        pof_hist_col_mapped_name = reached_stage_cols_map_hist.get(STAGE_PASSED_ONLINE_FORM)
        valid_historical_rates_found = False

        for rate_key in manual_rates_sidebar.keys(): 
            try: stage_from_name, stage_to_name = rate_key.split(" -> ")
            except ValueError: 
                calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                substitutions_made_log.append(f"{rate_key}: Error parsing stage names, used manual rate."); continue

            if stage_from_name == STAGE_PASSED_ONLINE_FORM: 
                actual_col_from = pof_hist_col_mapped_name if pof_hist_col_mapped_name else "Total_Submissions_Calc"
            else:
                actual_col_from = reached_stage_cols_map_hist.get(stage_from_name)
            col_to_cleaned_name = reached_stage_cols_map_hist.get(stage_to_name)

            if actual_col_from and col_to_cleaned_name and actual_col_from in hist_counts.columns and col_to_cleaned_name in hist_counts.columns:
                total_numerator = hist_counts[col_to_cleaned_name].sum(); total_denominator = hist_counts[actual_col_from].sum()
                overall_hist_rate_for_key = (total_numerator / total_denominator) if total_denominator > 0 else np.nan
                manual_rate_for_key = manual_rates_sidebar.get(rate_key, 0.0) 
                maturity_days_for_this_rate = MATURITY_PERIODS_DAYS.get(rate_key, DEFAULT_MATURITY_DAYS)
                
                adjusted_monthly_rates_list = []; months_used_for_rate = []
                for month_period_loop in hist_counts.index: 
                    if month_period_loop.end_time + pd.Timedelta(days=maturity_days_for_this_rate) < pd.Timestamp(datetime.now()):
                        months_used_for_rate.append(month_period_loop)
                        numerator_val = hist_counts.loc[month_period_loop, col_to_cleaned_name]
                        denominator_val = hist_counts.loc[month_period_loop, actual_col_from]
                        rate_for_this_month_calc = 0.0
                        
                        if denominator_val < MIN_DENOMINATOR_FOR_RATE_CALC:
                            rate_for_this_month_calc = manual_rate_for_key
                            log_reason_detail = f"used manual rate ({manual_rate_for_key*100:.1f}%)"
                            is_manual_rate_placeholder = (manual_rate_for_key >= 0.99 or manual_rate_for_key <= 0.01) 
                            if is_manual_rate_placeholder:
                                if pd.notna(overall_hist_rate_for_key) and total_denominator >= MIN_TOTAL_DENOMINATOR_FOR_OVERALL_RATE:
                                    rate_for_this_month_calc = overall_hist_rate_for_key
                                    log_reason_detail = f"manual placeholder, used overall hist. ({overall_hist_rate_for_key*100:.1f}%, total N={total_denominator})"
                            substitutions_made_log.append(f"Mth {month_period_loop.strftime('%Y-%m')} for '{rate_key}': Denom ({denominator_val}) < {MIN_DENOMINATOR_FOR_RATE_CALC}, {log_reason_detail}. Maturity: {maturity_days_for_this_rate}d.")
                        elif denominator_val > 0: 
                            rate_for_this_month_calc = numerator_val / denominator_val
                        adjusted_monthly_rates_list.append(rate_for_this_month_calc)
                    else: 
                        substitutions_made_log.append(f"Mth {month_period_loop.strftime('%Y-%m')} for '{rate_key}': Excluded (too recent, maturity: {maturity_days_for_this_rate}d).")
                
                if not adjusted_monthly_rates_list: 
                    calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0) 
                    substitutions_made_log.append(f"{rate_key}: No mature hist. mths (mat: {maturity_days_for_this_rate}d), used manual rate."); continue

                adjusted_monthly_rates_series = pd.Series(adjusted_monthly_rates_list, index=pd.PeriodIndex(months_used_for_rate, freq='M'))
                actual_window_to_calc = min(rolling_window_sidebar, len(adjusted_monthly_rates_series)) 

                if actual_window_to_calc > 0 and not adjusted_monthly_rates_series.empty:
                    rolling_avg_rate_series = adjusted_monthly_rates_series.rolling(window=actual_window_to_calc, min_periods=1).mean()
                    if not rolling_avg_rate_series.empty:
                        latest_rolling_rate = rolling_avg_rate_series.iloc[-1]
                        calculated_rolling_rates[rate_key] = latest_rolling_rate if pd.notna(latest_rolling_rate) else 0.0
                        valid_historical_rates_found = True 
                    else: 
                        calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                        substitutions_made_log.append(f"{rate_key}: Rolling avg empty (mat: {maturity_days_for_this_rate}d), used manual rate.")
                else: 
                    if not adjusted_monthly_rates_series.empty: 
                        mean_mature_rate_val = adjusted_monthly_rates_series.mean()
                        calculated_rolling_rates[rate_key] = mean_mature_rate_val if pd.notna(mean_mature_rate_val) else manual_rates_sidebar.get(rate_key, 0.0)
                        substitutions_made_log.append(f"{rate_key}: Window {actual_window_to_calc} (mat: {maturity_days_for_this_rate}d) too small or no data; used mean of mature or manual. Valid: {pd.notna(mean_mature_rate_val)}")
                        if pd.notna(mean_mature_rate_val): valid_historical_rates_found = True
                    else: 
                        calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                        substitutions_made_log.append(f"{rate_key}: No mature data for rolling (mat: {maturity_days_for_this_rate}d), used manual rate.")
            else: 
                calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                substitutions_made_log.append(f"{rate_key}: Stage columns N/A in historical summary, used manual rate.")

        if sidebar_display_area and substitutions_made_log:
            with sidebar_display_area.expander("Rolling Rate Calculation Log (Adjustments & Maturity)", expanded=False):
                sidebar_display_area.caption("Maturity Periods Applied (Days):")
                if MATURITY_PERIODS_DAYS:
                    for r_key_disp_log, mat_days_disp_log in MATURITY_PERIODS_DAYS.items(): sidebar_display_area.caption(f"- {r_key_disp_log}: {mat_days_disp_log} days")
                else: sidebar_display_area.caption("Maturity periods N/A.")
                sidebar_display_area.caption("--- Substitution/Exclusion Log ---")
                for log_entry_disp in substitutions_made_log: sidebar_display_area.caption(log_entry_disp)
        
        if not valid_historical_rates_found: 
            if sidebar_display_area: sidebar_display_area.warning("No valid historical rolling rates could be calculated, using manual inputs provided.")
            return manual_rates_sidebar, "Manual (All Rolling Calcs Failed or Invalid)"
        else:
            if sidebar_display_area:
                sidebar_display_area.markdown("---"); sidebar_display_area.subheader(f"Effective {rolling_window_sidebar}-Mo. Rolling Rates (Adj. & Matured):")
                for key_disp, val_disp in calculated_rolling_rates.items():
                    if key_disp in manual_rates_sidebar: sidebar_display_area.text(f"- {key_disp}: {val_disp*100:.1f}%") 
                sidebar_display_area.markdown("---")
            return calculated_rolling_rates, f"Rolling {rolling_window_sidebar}-Month Avg (Adj. & Matured)"
    except Exception as e:
        if sidebar_display_area: sidebar_display_area.error(f"Error calculating rolling rates: {e}"); sidebar_display_area.exception(e)
        return manual_rates_sidebar, "Manual (Error in Rolling Calc)"

@st.cache_data
def calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs):
    default_return_tuple = pd.DataFrame(), np.nan, "N/A", "N/A", pd.DataFrame(), "N/A"
    if _processed_df is None or _processed_df.empty: return default_return_tuple
    required_keys = ['horizon', 'spend_dict', 'cpqr_dict', 'final_conv_rates',
                     'goal_icf', 'site_performance_data', 'inter_stage_lags',
                     'icf_variation_percentage']
    if not isinstance(projection_inputs, dict) or not all(k in projection_inputs for k in required_keys):
        missing_keys = [k for k in required_keys if k not in projection_inputs]
        st.warning(f"Proj: Missing inputs. Need: {missing_keys}. Got: {list(projection_inputs.keys())}")
        return default_return_tuple

    processed_df = _processed_df.copy(); horizon = projection_inputs['horizon']
    future_spend_dict = projection_inputs['spend_dict']
    assumed_cpqr_dict = projection_inputs['cpqr_dict']
    final_projection_conv_rates = projection_inputs['final_conv_rates']
    goal_total_icfs = projection_inputs['goal_icf']
    site_performance_data = projection_inputs['site_performance_data'] 
    inter_stage_lags = projection_inputs.get('inter_stage_lags', {})
    icf_variation_percent = projection_inputs.get('icf_variation_percentage', 0)
    variation_factor = icf_variation_percent / 100.0

    avg_actual_lag_days_for_display = np.nan; lag_calculation_method_message = "Lag not calculated."
    projection_segments_for_lag_path = [
        (STAGE_PASSED_ONLINE_FORM, STAGE_PRE_SCREENING_ACTIVITIES),
        (STAGE_PRE_SCREENING_ACTIVITIES, STAGE_SENT_TO_SITE),
        (STAGE_SENT_TO_SITE, STAGE_APPOINTMENT_SCHEDULED),
        (STAGE_APPOINTMENT_SCHEDULED, STAGE_SIGNED_ICF)
    ]
    calculated_sum_of_lags = 0; valid_segments_count = 0; all_segments_found_and_valid = True

    if inter_stage_lags: 
        for stage_from, stage_to in projection_segments_for_lag_path:
            lag_key = f"{stage_from} -> {stage_to}"; lag_value = inter_stage_lags.get(lag_key)
            if ts_col_map.get(stage_from) and ts_col_map.get(stage_to):
                if pd.notna(lag_value):
                    calculated_sum_of_lags += lag_value; valid_segments_count += 1
                else: 
                    all_segments_found_and_valid = False; break
            else: 
                all_segments_found_and_valid = False; break
        if all_segments_found_and_valid and valid_segments_count == len(projection_segments_for_lag_path):
            avg_actual_lag_days_for_display = calculated_sum_of_lags
            lag_calculation_method_message = "Using summed inter-stage lags for POF->ICF projection path."
        else: 
            all_segments_found_and_valid = False
            lag_calculation_method_message = "Summed inter-stage lag for POF->ICF path failed. "
    else: 
        all_segments_found_and_valid = False
        lag_calculation_method_message = "Inter-stage lags not available. "

    if not all_segments_found_and_valid or pd.isna(avg_actual_lag_days_for_display):
        start_stage_for_overall_lag = ordered_stages[0] if ordered_stages and len(ordered_stages) > 0 else None
        end_stage_for_overall_lag = STAGE_SIGNED_ICF 
        overall_lag_calc_val = np.nan
        if start_stage_for_overall_lag and ts_col_map.get(start_stage_for_overall_lag) and ts_col_map.get(end_stage_for_overall_lag):
            ts_col_start_overall = ts_col_map[start_stage_for_overall_lag]
            ts_col_end_overall = ts_col_map[end_stage_for_overall_lag]
            if ts_col_start_overall in processed_df.columns and ts_col_end_overall in processed_df.columns:
                overall_lag_calc_val = calculate_avg_lag_generic(processed_df, ts_col_start_overall, ts_col_end_overall)
        
        if pd.notna(overall_lag_calc_val):
            avg_actual_lag_days_for_display = overall_lag_calc_val
            lag_calculation_method_message += f"Used historical overall lag ({start_stage_for_overall_lag} -> {end_stage_for_overall_lag})."
        else: 
            avg_actual_lag_days_for_display = 30.0 
            lag_calculation_method_message += "Used default lag (30 days)."
            
    if pd.isna(avg_actual_lag_days_for_display): 
        avg_actual_lag_days_for_display = 30.0
        lag_calculation_method_message = "Critical Lag Error: All methods failed. Used default 30 days."

    lpi_date_str = "Goal Not Met"; ads_off_date_str = "N/A"; site_level_projections_df_final = pd.DataFrame()
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

        icf_proj_col_name_base = "" 
        current_stage_count_col = 'Forecasted_PSQ' 

        for stage_from, stage_to in projection_segments_for_lag_path: 
            conv_rate_key = f"{stage_from} -> {stage_to}"
            conv_rate = final_projection_conv_rates.get(conv_rate_key, 0.0) 
            proj_col_to_name = f"Projected_{stage_to.replace(' ', '_').replace('(', '').replace(')', '')}"
            if current_stage_count_col in projection_cohorts.columns:
                proj_counts_for_to_stage = (projection_cohorts[current_stage_count_col] * conv_rate)
                projection_cohorts[proj_col_to_name] = proj_counts_for_to_stage.round(0).fillna(0).astype(int)
                current_stage_count_col = proj_col_to_name 
            else: 
                projection_cohorts[proj_col_to_name] = 0
                current_stage_count_col = proj_col_to_name
            if stage_to == STAGE_SIGNED_ICF: 
                icf_proj_col_name_base = proj_col_to_name; break 
        
        projection_results = pd.DataFrame(index=future_months); projection_results['Projected_ICF_Landed'] = 0.0
        if not icf_proj_col_name_base or icf_proj_col_name_base not in projection_cohorts.columns:
            st.error(f"Critical Error: Projected ICF column ('{icf_proj_col_name_base}') not found after funnel progression.");
            return default_return_tuple[0], avg_actual_lag_days_for_display, "Error", "Error", pd.DataFrame(), "ICF Proj Col Missing after funnel calc"

        icf_proj_col_low = f"{icf_proj_col_name_base}_low"
        icf_proj_col_high = f"{icf_proj_col_name_base}_high"
        projection_cohorts[icf_proj_col_low] = (projection_cohorts[icf_proj_col_name_base] * (1 - variation_factor)).round(0).astype(int).clip(lower=0)
        projection_cohorts[icf_proj_col_high] = (projection_cohorts[icf_proj_col_name_base] * (1 + variation_factor)).round(0).astype(int).clip(lower=0)

        projection_cohorts['Projected_CPICF_Cohort_Mean'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_name_base].replace(0, np.nan)).round(2)
        projection_cohorts['Projected_CPICF_Cohort_Low'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_high].replace(0, np.nan)).round(2)
        projection_cohorts['Projected_CPICF_Cohort_High'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_low].replace(0, np.nan)).round(2)

        overall_current_lag_days_to_use = avg_actual_lag_days_for_display
        days_in_avg_month = 30.4375 

        for cohort_start_month_iter, row_data in projection_cohorts.iterrows():
            icfs_from_this_cohort_val = row_data[icf_proj_col_name_base] 
            if icfs_from_this_cohort_val == 0: continue
            full_lag_months_val = int(np.floor(overall_current_lag_days_to_use / days_in_avg_month))
            remaining_lag_days_comp_val = overall_current_lag_days_to_use - (full_lag_months_val * days_in_avg_month)
            fraction_for_next_month_val = remaining_lag_days_comp_val / days_in_avg_month
            fraction_for_current_offset_month_val = 1.0 - fraction_for_next_month_val
            icfs_month_1_val = icfs_from_this_cohort_val * fraction_for_current_offset_month_val
            icfs_month_2_val = icfs_from_this_cohort_val * fraction_for_next_month_val
            landing_month_1_period_val = cohort_start_month_iter + full_lag_months_val
            landing_month_2_period_val = cohort_start_month_iter + full_lag_months_val + 1
            if landing_month_1_period_val in projection_results.index:
                projection_results.loc[landing_month_1_period_val, 'Projected_ICF_Landed'] += icfs_month_1_val
            if landing_month_2_period_val in projection_results.index: 
                projection_results.loc[landing_month_2_period_val, 'Projected_ICF_Landed'] += icfs_month_2_val
        projection_results['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed'].round(0).fillna(0).astype(int)
        projection_results['Cumulative_ICF_Landed'] = projection_results['Projected_ICF_Landed'].cumsum()

        lpi_month_series_val = projection_results[projection_results['Cumulative_ICF_Landed'] >= goal_total_icfs]
        if not lpi_month_series_val.empty:
            lpi_month_period_val = lpi_month_series_val.index[0]
            icfs_in_lpi_month_val = projection_results.loc[lpi_month_period_val, 'Projected_ICF_Landed']
            cumulative_before_lpi_direct_val = projection_results['Cumulative_ICF_Landed'].shift(1).fillna(0).loc[lpi_month_period_val]
            icfs_needed_in_lpi_month_val = goal_total_icfs - cumulative_before_lpi_direct_val
            if icfs_in_lpi_month_val > 0:
                fraction_of_lpi_month_val = max(0,min(1, icfs_needed_in_lpi_month_val / icfs_in_lpi_month_val))
                lpi_day_offset_val = int(np.ceil(fraction_of_lpi_month_val * days_in_avg_month)); lpi_day_offset_val = max(1, lpi_day_offset_val)
                lpi_date_val_calc = lpi_month_period_val.start_time + pd.Timedelta(days=lpi_day_offset_val -1)
                lpi_date_str = lpi_date_val_calc.strftime('%Y-%m-%d')
            elif icfs_needed_in_lpi_month_val <= 0: 
                lpi_date_str = (lpi_month_period_val.start_time - pd.Timedelta(days=1)).strftime('%Y-%m-%d') 
            else: 
                lpi_date_str = lpi_month_period_val.start_time.strftime('%Y-%m-%d')
        
        projection_cohorts['Cumulative_Projected_ICF_Generated'] = projection_cohorts[icf_proj_col_name_base].cumsum()
        ads_off_month_series_val = projection_cohorts[projection_cohorts['Cumulative_Projected_ICF_Generated'] >= goal_total_icfs]
        if not ads_off_month_series_val.empty:
            ads_off_month_period_val = ads_off_month_series_val.index[0]
            ads_off_date_str = ads_off_month_period_val.end_time.strftime('%Y-%m-%d') 

        display_df_out = pd.DataFrame(index=future_months)
        display_df_out['Forecasted_Ad_Spend'] = projection_cohorts['Forecasted_Ad_Spend']
        display_df_out['Forecasted_Qual_Referrals'] = projection_cohorts['Forecasted_PSQ'] 
        display_df_out['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed']

        cpicf_display_mean_val = pd.Series(index=future_months, dtype=float)
        cpicf_display_low_val = pd.Series(index=future_months, dtype=float)
        cpicf_display_high_val = pd.Series(index=future_months, dtype=float)
        lag_for_cpicf_display_val = int(np.round(overall_current_lag_days_to_use / days_in_avg_month))

        for i_cohort_idx, cohort_start_month_cpicf in enumerate(projection_cohorts.index):
            primary_land_m_cpicf = cohort_start_month_cpicf + lag_for_cpicf_display_val
            if primary_land_m_cpicf in cpicf_display_mean_val.index:
                if pd.isna(cpicf_display_mean_val.loc[primary_land_m_cpicf]):
                    cpicf_display_mean_val.loc[primary_land_m_cpicf] = projection_cohorts.iloc[i_cohort_idx]['Projected_CPICF_Cohort_Mean']
                    cpicf_display_low_val.loc[primary_land_m_cpicf] = projection_cohorts.iloc[i_cohort_idx]['Projected_CPICF_Cohort_Low']
                    cpicf_display_high_val.loc[primary_land_m_cpicf] = projection_cohorts.iloc[i_cohort_idx]['Projected_CPICF_Cohort_High']
        
        display_df_out['Projected_CPICF_Cohort_Source_Mean'] = cpicf_display_mean_val
        display_df_out['Projected_CPICF_Cohort_Source_Low'] = cpicf_display_low_val
        display_df_out['Projected_CPICF_Cohort_Source_High'] = cpicf_display_high_val
        
        if 'Site' in _processed_df.columns and not _processed_df['Site'].empty and ordered_stages:
            if site_performance_data.empty or 'Site' not in site_performance_data.columns:
                 st.warning("Site performance data is empty or missing 'Site' column for site-level projections.")
                 site_level_projections_df_final = pd.DataFrame() 
            else:
                historical_site_pof_proportions = _processed_df[_processed_df[ts_col_map.get(STAGE_PASSED_ONLINE_FORM)].notna()]['Site'].value_counts(normalize=True) if ts_col_map.get(STAGE_PASSED_ONLINE_FORM) in _processed_df else pd.Series(dtype=float)
                all_sites_proj = site_performance_data['Site'].unique() 
                site_data_collector_proj = {}

                for site_name_s_proj in all_sites_proj:
                    site_data_collector_proj[site_name_s_proj] = {}
                    for month_period_s_proj in future_months: 
                        month_str_s_proj = month_period_s_proj.strftime('%Y-%m')
                        site_data_collector_proj[site_name_s_proj][(month_str_s_proj, 'Projected Qual. Referrals (POF)')] = 0
                        site_data_collector_proj[site_name_s_proj][(month_str_s_proj, 'Projected ICFs Landed')] = 0.0

                for cohort_start_month_s_proj, cohort_row_s_proj in projection_cohorts.iterrows():
                    total_psq_this_cohort_s_proj = cohort_row_s_proj['Forecasted_PSQ']
                    if total_psq_this_cohort_s_proj <= 0: continue

                    for site_name_iter_s_proj in all_sites_proj:
                        site_prop_s_proj = historical_site_pof_proportions.get(site_name_iter_s_proj, 0)
                        site_proj_pof_cohort_s_proj = total_psq_this_cohort_s_proj * site_prop_s_proj
                        
                        month_str_cohort_start_s_proj = cohort_start_month_s_proj.strftime('%Y-%m')
                        site_data_collector_proj[site_name_iter_s_proj][(month_str_cohort_start_s_proj, 'Projected Qual. Referrals (POF)')] += round(site_proj_pof_cohort_s_proj)
                        
                        site_perf_row_s_proj = site_performance_data[site_performance_data['Site'] == site_name_iter_s_proj]
                        current_site_proj_count_s_proj = site_proj_pof_cohort_s_proj 

                        for i_seg_s, (stage_from_seg_s, stage_to_seg_s) in enumerate(projection_segments_for_lag_path):
                            site_rate_key_s = "" 
                            if stage_from_seg_s == STAGE_PASSED_ONLINE_FORM and stage_to_seg_s == STAGE_PRE_SCREENING_ACTIVITIES: site_rate_key_s = 'POF -> PSA %'
                            elif stage_from_seg_s == STAGE_PRE_SCREENING_ACTIVITIES and stage_to_seg_s == STAGE_SENT_TO_SITE: site_rate_key_s = 'PSA -> StS %'
                            elif stage_from_seg_s == STAGE_SENT_TO_SITE and stage_to_seg_s == STAGE_APPOINTMENT_SCHEDULED: site_rate_key_s = 'StS -> Appt %'
                            elif stage_from_seg_s == STAGE_APPOINTMENT_SCHEDULED and stage_to_seg_s == STAGE_SIGNED_ICF: site_rate_key_s = 'Appt -> ICF %'
                            overall_rate_key_s = f"{stage_from_seg_s} -> {stage_to_seg_s}"
                            rate_to_use_s = final_projection_conv_rates.get(overall_rate_key_s, 0.0) 
                            if not site_perf_row_s_proj.empty and site_rate_key_s and site_rate_key_s in site_perf_row_s_proj.columns:
                                site_specific_rate_val_s = site_perf_row_s_proj[site_rate_key_s].iloc[0]
                                if pd.notna(site_specific_rate_val_s): rate_to_use_s = site_specific_rate_val_s
                            current_site_proj_count_s_proj *= rate_to_use_s
                            if stage_to_seg_s == STAGE_SIGNED_ICF: break 
                        site_proj_icfs_generated_this_cohort_s = current_site_proj_count_s_proj
                        
                        lag_to_use_for_site_s = overall_current_lag_days_to_use 
                        if not site_perf_row_s_proj.empty and 'Site Projection Lag (Days)' in site_perf_row_s_proj.columns:
                            site_specific_lag_val_s = site_perf_row_s_proj['Site Projection Lag (Days)'].iloc[0]
                            if pd.notna(site_specific_lag_val_s): lag_to_use_for_site_s = site_specific_lag_val_s
                        
                        if site_proj_icfs_generated_this_cohort_s > 0:
                            full_lag_m_site_s = int(np.floor(lag_to_use_for_site_s / days_in_avg_month))
                            remain_lag_days_comp_site_s = lag_to_use_for_site_s - (full_lag_m_site_s * days_in_avg_month)
                            frac_next_m_site_s = remain_lag_days_comp_site_s / days_in_avg_month
                            frac_curr_m_site_s = 1.0 - frac_next_m_site_s
                            icfs_m1_site_s = site_proj_icfs_generated_this_cohort_s * frac_curr_m_site_s
                            icfs_m2_site_s = site_proj_icfs_generated_this_cohort_s * frac_next_m_site_s
                            land_m1_p_site_s = cohort_start_month_s_proj + full_lag_m_site_s
                            land_m2_p_site_s = land_m1_p_site_s + 1
                            if land_m1_p_site_s in future_months:
                                site_data_collector_proj[site_name_iter_s_proj][(land_m1_p_site_s.strftime('%Y-%m'), 'Projected ICFs Landed')] += icfs_m1_site_s
                            if land_m2_p_site_s in future_months:
                                site_data_collector_proj[site_name_iter_s_proj][(land_m2_p_site_s.strftime('%Y-%m'), 'Projected ICFs Landed')] += icfs_m2_site_s
                
                if site_data_collector_proj:
                    site_level_projections_df_final = pd.DataFrame.from_dict(site_data_collector_proj, orient='index')
                    site_level_projections_df_final.columns = pd.MultiIndex.from_tuples(site_level_projections_df_final.columns, names=['Month', 'Metric'])
                    for m_period_fmt_s_proj in future_months.strftime('%Y-%m'):
                        if (m_period_fmt_s_proj, 'Projected ICFs Landed') in site_level_projections_df_final.columns:
                            site_level_projections_df_final[(m_period_fmt_s_proj, 'Projected ICFs Landed')] = site_level_projections_df_final[(m_period_fmt_s_proj, 'Projected ICFs Landed')].round(0).astype(int)
                        if (m_period_fmt_s_proj, 'Projected Qual. Referrals (POF)') in site_level_projections_df_final.columns:
                            site_level_projections_df_final[(m_period_fmt_s_proj, 'Projected Qual. Referrals (POF)')] = site_level_projections_df_final[(m_period_fmt_s_proj, 'Projected Qual. Referrals (POF)')].astype(int)
                    site_level_projections_df_final = site_level_projections_df_final.sort_index(axis=1, level=[0,1])
                    if not site_level_projections_df_final.empty: 
                        numeric_cols_slp = [col for col in site_level_projections_df_final.columns if pd.api.types.is_numeric_dtype(site_level_projections_df_final[col])]
                        if numeric_cols_slp:
                            total_row_vals_slp = site_level_projections_df_final[numeric_cols_slp].sum(axis=0)
                            total_row_df_slp = pd.DataFrame([total_row_vals_slp], index=["Grand Total"])
                            site_level_projections_df_final = pd.concat([site_level_projections_df_final, total_row_df_slp])
        return display_df_out, avg_actual_lag_days_for_display, lpi_date_str, ads_off_date_str, site_level_projections_df_final, lag_calculation_method_message
    except Exception as e:
        st.error(f"Projection calc error (main or site-level): {e}"); st.exception(e)
        return default_return_tuple[0], avg_actual_lag_days_for_display if pd.notna(avg_actual_lag_days_for_display) else 30.0, "Error", "Error", pd.DataFrame(), f"Error: {e}"

# --- NEW: AI Forecast Core Function ---
def calculate_ai_forecast_core(
    goal_lpi_date_dt: datetime,
    goal_icf_number: int,
    estimated_cpql: float, 
    icf_variation_percent: float,
    processed_df: pd.DataFrame,
    ordered_stages: list,
    ts_col_map: dict,
    effective_projection_conv_rates: dict, 
    avg_overall_lag_days: float, 
    site_metrics_df: pd.DataFrame,
    projection_horizon_months: int 
):
    default_return_ai = pd.DataFrame(), pd.DataFrame(), "N/A", "Not Calculated", True 
    if not all([processed_df is not None, not processed_df.empty, ordered_stages, ts_col_map, effective_projection_conv_rates, site_metrics_df is not None]):
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Missing critical base data for AI Forecast.", True
    if goal_icf_number <= 0: return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Goal ICF number must be positive.", True
    if estimated_cpql <= 0: return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Estimated CPQL must be positive.", True
    if pd.isna(avg_overall_lag_days) or avg_overall_lag_days < 0:
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Average POF to ICF lag is invalid or not calculated.", True

    main_funnel_path_segments = [
        f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}",
        f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}",
        f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}",
        f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"
    ]
    overall_pof_to_icf_rate = 1.0
    for segment in main_funnel_path_segments:
        rate = effective_projection_conv_rates.get(segment)
        if rate is None or rate < 0: # Added check for negative rate
            return default_return_ai[0], default_return_ai[1], default_return_ai[2], f"Conversion rate for segment '{segment}' invalid or not found. Cannot calculate overall POF to ICF rate.", True
        overall_pof_to_icf_rate *= rate
    
    if overall_pof_to_icf_rate <= 1e-9: # Check for effectively zero rate
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Overall POF to ICF conversion rate is zero or negligible. Cannot project ICFs.", True

    last_hist_month = processed_df["Submission_Month"].max() if "Submission_Month" in processed_df and not processed_df["Submission_Month"].empty else pd.Period(datetime.now(), freq='M') - 1
    proj_start_month_period = last_hist_month + 1
    goal_lpi_month_period = pd.Period(goal_lpi_date_dt, freq='M')
    
    avg_lag_months_approx = int(round(avg_overall_lag_days / 30.4375))
    first_possible_landing_month = proj_start_month_period + avg_lag_months_approx
    actual_landing_start_month = max(proj_start_month_period, first_possible_landing_month) 
    
    landing_window_months = pd.period_range(start=actual_landing_start_month, end=goal_lpi_month_period, freq='M')

    if landing_window_months.empty or goal_lpi_month_period < actual_landing_start_month:
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], f"Goal LPI date ({goal_lpi_month_period.strftime('%Y-%m')}) is too soon. Earliest ICFs can land is {actual_landing_start_month.strftime('%Y-%m')}.", True
    
    num_months_for_landing = len(landing_window_months)
    target_icfs_to_land_per_month = goal_icf_number / num_months_for_landing

    max_projection_end_month = goal_lpi_month_period + avg_lag_months_approx + 3 
    limited_proj_end_month = proj_start_month_period + projection_horizon_months -1 
    final_proj_end_month = min(max_projection_end_month, limited_proj_end_month)
    projection_calc_months = pd.period_range(start=proj_start_month_period, end=final_proj_end_month, freq='M')
    if projection_calc_months.empty : 
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Projection calculation window is invalid.", True

    ai_gen_df = pd.DataFrame(index=projection_calc_months)
    ai_gen_df['Generated_ICF_Mean'] = 0.0

    for l_month in landing_window_months:
        g_month = l_month - avg_lag_months_approx 
        if g_month in ai_gen_df.index:
            ai_gen_df.loc[g_month, 'Generated_ICF_Mean'] += target_icfs_to_land_per_month
        elif g_month < proj_start_month_period : # Check if generation is needed before proj_start_month
             return default_return_ai[0], default_return_ai[1], default_return_ai[2], f"To meet LPI, ICF generation required in {g_month.strftime('%Y-%m')}, which is before projection start {proj_start_month_period.strftime('%Y-%m')}. Review LPI or goals.", True
        # If g_month is outside ai_gen_df.index but not before proj_start_month_period, it means LPI is too far for current horizon (handled later by feasibility)


    variation_f = icf_variation_percent / 100.0
    ai_gen_df['Generated_ICF_Low'] = (ai_gen_df['Generated_ICF_Mean'] * (1 - variation_f)).round(0).clip(lower=0)
    ai_gen_df['Generated_ICF_High'] = (ai_gen_df['Generated_ICF_Mean'] * (1 + variation_f)).round(0).clip(lower=0)
    ai_gen_df['Required_QLs_POF'] = (ai_gen_df['Generated_ICF_Mean'] / overall_pof_to_icf_rate).round(0).astype(int) # Round QLs
    ai_gen_df['Implied_Ad_Spend'] = ai_gen_df['Required_QLs_POF'] * estimated_cpql
    
    ai_gen_df['Projected_CPICF_Mean'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_Mean'].replace(0, np.nan)).round(2)
    ai_gen_df['Projected_CPICF_Low'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_High'].replace(0, np.nan)).round(2)
    ai_gen_df['Projected_CPICF_High'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_Low'].replace(0, np.nan)).round(2)

    ai_results_df = pd.DataFrame(index=projection_calc_months)
    ai_results_df['Projected_ICF_Landed'] = 0.0
    days_in_avg_m = 30.4375

    for cohort_g_month, cohort_row in ai_gen_df.iterrows():
        icfs_gen_this_cohort = cohort_row['Generated_ICF_Mean']
        if icfs_gen_this_cohort <= 0: continue
        full_lag_mths = int(np.floor(avg_overall_lag_days / days_in_avg_m))
        rem_lag_days = avg_overall_lag_days - (full_lag_mths * days_in_avg_m)
        frac_next = rem_lag_days / days_in_avg_m; frac_curr = 1.0 - frac_next
        land_m1 = cohort_g_month + full_lag_mths; land_m2 = land_m1 + 1
        if land_m1 in ai_results_df.index: ai_results_df.loc[land_m1, 'Projected_ICF_Landed'] += icfs_gen_this_cohort * frac_curr
        if land_m2 in ai_results_df.index: ai_results_df.loc[land_m2, 'Projected_ICF_Landed'] += icfs_gen_this_cohort * frac_next
    ai_results_df['Projected_ICF_Landed'] = ai_results_df['Projected_ICF_Landed'].round(0).astype(int)
    ai_results_df['Cumulative_ICF_Landed'] = ai_results_df['Projected_ICF_Landed'].cumsum()

    ai_results_df['Target_QLs_POF'] = ai_gen_df['Required_QLs_POF'].reindex(ai_results_df.index).fillna(0).round(0).astype(int)
    ai_results_df['Implied_Ad_Spend'] = ai_gen_df['Implied_Ad_Spend'].reindex(ai_results_df.index).fillna(0)
    
    cpicf_m = pd.Series(index=projection_calc_months, dtype=float); cpicf_l = pd.Series(index=projection_calc_months, dtype=float); cpicf_h = pd.Series(index=projection_calc_months, dtype=float)
    for g_m, g_data in ai_gen_df.iterrows():
        if g_data['Generated_ICF_Mean'] > 0:
            display_land_m = g_m + avg_lag_months_approx 
            if display_land_m in cpicf_m.index and pd.isna(cpicf_m.loc[display_land_m]):
                cpicf_m.loc[display_land_m] = g_data['Projected_CPICF_Mean']
                cpicf_l.loc[display_land_m] = g_data['Projected_CPICF_Low']
                cpicf_h.loc[display_land_m] = g_data['Projected_CPICF_High']
    ai_results_df['Projected_CPICF_Cohort_Source_Mean'] = cpicf_m
    ai_results_df['Projected_CPICF_Cohort_Source_Low'] = cpicf_l
    ai_results_df['Projected_CPICF_Cohort_Source_High'] = cpicf_h

    ai_gen_df['Cumulative_Generated_ICF'] = ai_gen_df['Generated_ICF_Mean'].cumsum()
    ads_off_s = ai_gen_df[ai_gen_df['Cumulative_Generated_ICF'] >= goal_icf_number]
    ads_off_date_str_calc = "Goal Not Met by End of Projection"
    if not ads_off_s.empty: ads_off_date_str_calc = ads_off_s.index[0].end_time.strftime('%Y-%m-%d')

    ai_site_proj_df = pd.DataFrame() 
    if 'Site' in processed_df.columns and not site_metrics_df.empty and STAGE_PASSED_ONLINE_FORM in ts_col_map:
        ts_pof_col_for_prop = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
        hist_site_pof_prop = processed_df[processed_df[ts_pof_col_for_prop].notna()]['Site'].value_counts(normalize=True) if ts_pof_col_for_prop and ts_pof_col_for_prop in processed_df else pd.Series(dtype=float)
        
        all_sites_for_ai_tab = site_metrics_df['Site'].unique()
        site_data_coll_ai = {site: { (m.strftime('%Y-%m'), 'Projected QLs (POF)'): 0,
                                     (m.strftime('%Y-%m'), 'Projected ICFs Landed'): 0.0 }
                               for site in all_sites_for_ai_tab for m in projection_calc_months}

        for g_m_site, cohort_r_site in ai_gen_df.iterrows():
            total_qls_cohort = cohort_r_site['Required_QLs_POF']
            if total_qls_cohort <= 0: continue
            for site_n in all_sites_for_ai_tab:
                site_p = hist_site_pof_prop.get(site_n, 0)
                site_qls = total_qls_cohort * site_p
                site_data_coll_ai[site_n][(g_m_site.strftime('%Y-%m'), 'Projected QLs (POF)')] += round(site_qls)
                site_perf_r = site_metrics_df[site_metrics_df['Site'] == site_n]
                site_pof_icf_rate = site_perf_r['Qual -> ICF %'].iloc[0] if not site_perf_r.empty and 'Qual -> ICF %' in site_perf_r and pd.notna(site_perf_r['Qual -> ICF %'].iloc[0]) else overall_pof_to_icf_rate
                site_gen_icfs = site_qls * site_pof_icf_rate
                
                s_full_lag_m = int(np.floor(avg_overall_lag_days / days_in_avg_m)) # Using overall lag for site landing in step 1
                s_rem_lag_d = avg_overall_lag_days - (s_full_lag_m * days_in_avg_m)
                s_frac_next = s_rem_lag_d / days_in_avg_m; s_frac_curr = 1.0 - s_frac_next
                s_land_m1 = g_m_site + s_full_lag_m; s_land_m2 = s_land_m1 + 1
                if s_land_m1 in projection_calc_months: site_data_coll_ai[site_n][(s_land_m1.strftime('%Y-%m'), 'Projected ICFs Landed')] += site_gen_icfs * s_frac_curr
                if s_land_m2 in projection_calc_months: site_data_coll_ai[site_n][(s_land_m2.strftime('%Y-%m'), 'Projected ICFs Landed')] += site_gen_icfs * s_frac_next
        if site_data_coll_ai:
            ai_site_proj_df = pd.DataFrame.from_dict(site_data_coll_ai, orient='index')
            if not ai_site_proj_df.empty: # Check if DF is not empty before processing columns
                ai_site_proj_df.columns = pd.MultiIndex.from_tuples(ai_site_proj_df.columns)
                ai_site_proj_df = ai_site_proj_df.sort_index(axis=1, level=[0,1])
                for m_fmt in projection_calc_months.strftime('%Y-%m'):
                    if (m_fmt, 'Projected ICFs Landed') in ai_site_proj_df.columns:
                        ai_site_proj_df[(m_fmt, 'Projected ICFs Landed')] = ai_site_proj_df[(m_fmt, 'Projected ICFs Landed')].round(0).astype(int)
                num_cols_site_ai = [c for c in ai_site_proj_df.columns if pd.api.types.is_numeric_dtype(ai_site_proj_df[c])]
                if num_cols_site_ai:
                    total_r_ai = ai_site_proj_df[num_cols_site_ai].sum(axis=0)
                    # ai_site_proj_df.loc["Grand Total"] = total_r_ai # This can cause issues if index is not Site
                    total_df_ai = pd.DataFrame(total_r_ai).T # Transpose to a row
                    total_df_ai.index = ["Grand Total"]
                    ai_site_proj_df = pd.concat([ai_site_proj_df, total_df_ai])
    else: 
        ai_site_proj_df = pd.DataFrame()

    final_cum_landed_icfs = ai_results_df['Cumulative_ICF_Landed'].max() if 'Cumulative_ICF_Landed' in ai_results_df else 0
    goal_met_on_time = False
    actual_lpi_month_for_msg = goal_lpi_month_period # Default to goal LPI for messages
    if 'Cumulative_ICF_Landed' in ai_results_df and not ai_results_df[ai_results_df['Cumulative_ICF_Landed'] >= goal_icf_number].empty:
        actual_lpi_month_for_msg = ai_results_df[ai_results_df['Cumulative_ICF_Landed'] >= goal_icf_number].index.min()
        if actual_lpi_month_for_msg <= goal_lpi_month_period:
            goal_met_on_time = True
    
    feasibility_msg_final = "Projection complete."
    is_unfeasible_final = False
    if not goal_met_on_time:
        feasibility_msg_final = f"Goal of {goal_icf_number} ICFs by {goal_lpi_month_period.strftime('%Y-%m')} appears unfeasible with current model. "
        if final_cum_landed_icfs < goal_icf_number:
            feasibility_msg_final += f"Projected to land only {final_cum_landed_icfs:.0f} ICFs. The goal is met by {actual_lpi_month_for_msg.strftime('%Y-%m')} if projection extends."
        else: # Goal met, but late
             feasibility_msg_final += f"ICF goal met, but by {actual_lpi_month_for_msg.strftime('%Y-%m')} (later than target)."
        is_unfeasible_final = True
    
    display_end_month = final_proj_end_month
    if not ai_results_df.empty and 'Cumulative_ICF_Landed' in ai_results_df and not ai_results_df[ai_results_df['Cumulative_ICF_Landed'] >= goal_icf_number].empty:
        # Show projection up to a few months after LPI or end of horizon, whichever is earlier
        lpi_achieved_month_for_trim = ai_results_df[ai_results_df['Cumulative_ICF_Landed'] >= goal_icf_number].index.min()
        display_end_month = min(final_proj_end_month, lpi_achieved_month_for_trim + 3) 
    
    ai_results_df_final_display = ai_results_df.loc[proj_start_month_period:display_end_month].copy() if not ai_results_df.empty else pd.DataFrame()
    return ai_results_df_final_display, ai_site_proj_df, ads_off_date_str_calc, feasibility_msg_final, is_unfeasible_final

# --- Streamlit UI ---
if 'data_processed_successfully' not in st.session_state: st.session_state.data_processed_successfully = False
if 'referral_data_processed' not in st.session_state: st.session_state.referral_data_processed = None
if 'funnel_definition' not in st.session_state: st.session_state.funnel_definition = None
if 'ordered_stages' not in st.session_state: st.session_state.ordered_stages = None
if 'ts_col_map' not in st.session_state: st.session_state.ts_col_map = None
if 'site_metrics_calculated' not in st.session_state: st.session_state.site_metrics_calculated = pd.DataFrame()
if 'inter_stage_lags' not in st.session_state: st.session_state.inter_stage_lags = None
if 'historical_spend_df' not in st.session_state: 
            st.session_state.historical_spend_df = pd.DataFrame([
                {'Month (YYYY-MM)': (datetime.now() - pd.DateOffset(months=2)).strftime('%Y-%m'), 'Historical Spend': 45000.0},
                {'Month (YYYY-MM)': (datetime.now() - pd.DateOffset(months=1)).strftime('%Y-%m'), 'Historical Spend': 60000.0}
            ])

ad_spend_input_dict = {}
weights_normalized = {}; proj_horizon_sidebar = 12
proj_spend_dict_sidebar = {}; proj_cpqr_dict_sidebar = {}; manual_proj_conv_rates_sidebar = {}
use_rolling_flag_sidebar = False; rolling_window_months_sidebar = 3; goal_icf_count_sidebar = 100
proj_icf_variation_percent_sidebar = 10

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"], key="referral_uploader_main")
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (TSV)", type=["tsv"], key="funnel_uploader_main")
    st.divider()

    with st.expander("Historical Ad Spend"):
        st.info("Enter **historical** ad spend for past months. Add rows as needed.")
        edited_historical_spend_df = st.data_editor(
            st.session_state.historical_spend_df, num_rows="dynamic", key="hist_spend_editor_main", use_container_width=True,
            column_config={
                "Month (YYYY-MM)": st.column_config.TextColumn(help="Enter month in YYYY-MM format",required=True),
                "Historical Spend": st.column_config.NumberColumn(help="Enter ad spend amount", min_value=0.0, format="$%.2f", required=True),
            }
        )
        temp_ad_spend_input_dict = {}
        valid_hist_spend_entries = True
        for index, row in edited_historical_spend_df.iterrows():
            month_str = str(row['Month (YYYY-MM)']).strip()
            spend_val = row['Historical Spend']
            if month_str and pd.notna(spend_val) and spend_val >=0 : 
                try:
                    month_period = pd.Period(month_str, freq='M')
                    temp_ad_spend_input_dict[month_period] = float(spend_val)
                except ValueError: st.sidebar.error(f"Invalid month format for hist. spend: '{month_str}'. Use YYYY-MM."); valid_hist_spend_entries = False; break
                except Exception as e: st.sidebar.error(f"Error processing hist. spend row ({month_str}, {spend_val}): {e}"); valid_hist_spend_entries = False; break
            elif not month_str and pd.notna(spend_val): 
                 st.sidebar.warning(f"Row {index+1} in historical spend has amount but no month. Ignoring.");
        if valid_hist_spend_entries: ad_spend_input_dict = temp_ad_spend_input_dict; st.session_state.historical_spend_df = edited_historical_spend_df
    st.divider()
    with st.expander("Site Scoring Weights"):
        weights_input_local = {}
        weights_input_local["Qual -> ICF %"] = st.slider("Qual (POF) -> ICF %", 0, 100, 20, key='w_qicf_v2')
        weights_input_local["Avg TTC (Days)"] = st.slider("Avg Time to Contact", 0, 100, 25, key='w_ttc_v2')
        weights_input_local["Avg Funnel Movement Steps"] = st.slider("Avg Funnel Movement Steps", 0, 100, 5, key='w_fms_v2')
        weights_input_local["Site Screen Fail %"] = st.slider("Site Screen Fail %", 0, 100, 5, key='w_sfr_v2')
        weights_input_local["StS -> Appt %"] = st.slider("StS -> Appt Sched %", 0, 100, 30, key='w_sa_site_score_v2')
        weights_input_local["Appt -> ICF %"] = st.slider("Appt Sched -> ICF %", 0, 100, 15, key='w_ai_site_score_v2')
        weights_input_local["Lag Qual -> ICF (Days)"] = st.slider("Lag Qual (POF) -> ICF (Days)", 0, 100, 0, key='w_lagqicf_v2')
        weights_input_local["Site Projection Lag (Days)"] = st.slider("Site Projection Lag (Days)", 0, 100, 0, key='w_siteprojlag_v2', help="Sum of average lags for key funnel segments specific to this site; used in site-level ICF landing projections.")
        total_weight_input_local = sum(abs(w) for w in weights_input_local.values())
        if total_weight_input_local > 0: weights_normalized = {k: v / total_weight_input_local for k, v in weights_input_local.items()}
        else: weights_normalized = {k: 0 for k in weights_input_local} 
        st.caption(f"Weights normalized. Lower is better for TTC, Screen Fail %, Lags.")
    st.divider()
    with st.expander("Projection Assumptions", expanded=True): 
        proj_horizon_sidebar = st.number_input("Projection Horizon (Months)", min_value=1, max_value=48, value=proj_horizon_sidebar, step=1, key='proj_horizon_widget_v2', help="Max months for the 'Projections' tab.")
        goal_icf_count_sidebar = st.number_input("Goal Total ICFs (for Projections Tab)", min_value=1, value=goal_icf_count_sidebar, step=1, key='goal_icf_input_v2', help="Target ICFs for standard projections.")
        _proj_start_month_ui_editor = pd.Period(datetime.now(), freq='M') + 1
        if st.session_state.data_processed_successfully and st.session_state.referral_data_processed is not None and \
           not st.session_state.referral_data_processed.empty and "Submission_Month" in st.session_state.referral_data_processed.columns:
            last_hist_month_for_ui_editor = st.session_state.referral_data_processed["Submission_Month"].max()
            if pd.notna(last_hist_month_for_ui_editor): _proj_start_month_ui_editor = last_hist_month_for_ui_editor + 1
        proj_horizon_editor_val = proj_horizon_sidebar if proj_horizon_sidebar > 0 else 1
        future_months_ui_for_editor = pd.period_range(start=_proj_start_month_ui_editor, periods=proj_horizon_editor_val, freq='M')
        st.write("Future Monthly Ad Spend (Projections Tab):")
        current_spend_df_rows = len(st.session_state.get('proj_spend_df_cache', []))
        if current_spend_df_rows != proj_horizon_editor_val or 'proj_spend_df_cache' not in st.session_state:
            st.session_state.proj_spend_df_cache = pd.DataFrame({'Month': future_months_ui_for_editor.strftime('%Y-%m'), 'Planned_Spend': [20000.0] * proj_horizon_editor_val})
        edited_spend_df = st.data_editor(st.session_state.proj_spend_df_cache, key='proj_spend_editor_v5', use_container_width=True, num_rows="fixed")
        if not edited_spend_df.equals(st.session_state.proj_spend_df_cache): st.session_state.proj_spend_df_cache = edited_spend_df 
        proj_spend_dict_sidebar = {m_init: 0.0 for m_init in future_months_ui_for_editor}
        if 'Month' in edited_spend_df.columns and 'Planned_Spend' in edited_spend_df.columns:
             for index, row_spend in edited_spend_df.iterrows():
                 try:
                     month_str_spend = str(row_spend['Month']).strip(); planned_spend_val = float(row_spend['Planned_Spend'])
                     month_period_spend = pd.Period(month_str_spend, freq='M')
                     if month_period_spend in proj_spend_dict_sidebar: proj_spend_dict_sidebar[month_period_spend] = planned_spend_val
                 except Exception: pass 
        st.write("Assumed CPQR ($) per Month (Projections Tab):")
        default_cpqr_value = 120.0
        if current_spend_df_rows != proj_horizon_editor_val or 'proj_cpqr_df_cache' not in st.session_state: 
             st.session_state.proj_cpqr_df_cache = pd.DataFrame({'Month': future_months_ui_for_editor.strftime('%Y-%m'), 'Assumed_CPQR': [default_cpqr_value] * proj_horizon_editor_val})
        edited_cpqr_df = st.data_editor(st.session_state.proj_cpqr_df_cache, key='proj_cpqr_editor_v5', use_container_width=True, num_rows="fixed")
        if not edited_cpqr_df.equals(st.session_state.proj_cpqr_df_cache): st.session_state.proj_cpqr_df_cache = edited_cpqr_df 
        proj_cpqr_dict_sidebar = {m_init_cpqr: default_cpqr_value for m_init_cpqr in future_months_ui_for_editor}
        if 'Month' in edited_cpqr_df.columns and 'Assumed_CPQR' in edited_cpqr_df.columns:
            for index, row_cpqr in edited_cpqr_df.iterrows():
                try:
                    month_str_cpqr = str(row_cpqr['Month']).strip(); cpqr_val = float(row_cpqr['Assumed_CPQR'])
                    month_period_cpqr = pd.Period(month_str_cpqr, freq='M')
                    if cpqr_val <=0: cpqr_val = default_cpqr_value 
                    if month_period_cpqr in proj_cpqr_dict_sidebar: proj_cpqr_dict_sidebar[month_period_cpqr] = cpqr_val
                except Exception: pass
        st.write("Conversion Rate Assumption (Projections Tab):")
        rate_assumption_method_sidebar = st.radio( "Use Rates Based On:", ('Manual Input Below', 'Rolling Historical Average'), key='rate_method_v2', horizontal=True )
        manual_proj_conv_rates_sidebar = {}
        cols_rate = st.columns(2)
        with cols_rate[0]:
             manual_proj_conv_rates_sidebar[f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}"] = st.slider("Manual: POF -> PreScreen %", 0.0, 100.0, 100.0, step=0.1, format="%.1f%%", key='cr_qps_v2') / 100.0
             manual_proj_conv_rates_sidebar[f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}"] = st.slider("Manual: PreScreen -> StS %", 0.0, 100.0, 17.0, step=0.1, format="%.1f%%", key='cr_pssts_v2') / 100.0
        with cols_rate[1]:
             manual_proj_conv_rates_sidebar[f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}"] = st.slider("Manual: StS -> Appt %", 0.0, 100.0, 33.0, step=0.1, format="%.1f%%", key='cr_sa_v2') / 100.0
             manual_proj_conv_rates_sidebar[f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"] = st.slider("Manual: Appt -> ICF %", 0.0, 100.0, 35.0, step=0.1, format="%.1f%%", key='cr_ai_v2') / 100.0
        use_rolling_flag_sidebar = (rate_assumption_method_sidebar == 'Rolling Historical Average')
        if use_rolling_flag_sidebar:
            rolling_window_months_sidebar = st.selectbox("Select Rolling Window (Months):", [1, 3, 6], index=1, key='rolling_window_v2')
            if st.session_state.data_processed_successfully and st.session_state.referral_data_processed is not None and \
               st.session_state.ordered_stages is not None and st.session_state.ts_col_map is not None:
                determine_effective_projection_rates( 
                    st.session_state.referral_data_processed, st.session_state.ordered_stages,
                    st.session_state.ts_col_map, rate_assumption_method_sidebar,
                    rolling_window_months_sidebar, manual_proj_conv_rates_sidebar, 
                    st.session_state.get('inter_stage_lags', {}), sidebar_display_area=st.sidebar)
            else: st.sidebar.caption("Upload data to view calculated rolling rates.")
        else: rolling_window_months_sidebar = 0; st.sidebar.caption("Using manually input rates above for Projections tab.")
        st.markdown("---")
        st.subheader("Projection Sensitivity (Shared)")
        proj_icf_variation_percent_sidebar = st.slider(
            "Projected ICF Variation (+/- %)", min_value=0, max_value=50, value=proj_icf_variation_percent_sidebar, step=1,
            key='proj_icf_variation_widget_v2',
            help="This percentage variation is applied to the number of ICFs generated by each cohort to estimate a Low-Mean-High range for the Cohort CPICF on both Projections and AI Forecast tabs."
        )

# --- Main App Logic & Display ---
if uploaded_referral_file is not None and uploaded_funnel_def_file is not None:
    if not st.session_state.data_processed_successfully:
        funnel_definition, ordered_stages, ts_col_map = parse_funnel_definition(uploaded_funnel_def_file)
        if funnel_definition and ordered_stages and ts_col_map:
            st.session_state.funnel_definition = funnel_definition; st.session_state.ordered_stages = ordered_stages; st.session_state.ts_col_map = ts_col_map
            try: 
                 bytes_data_ref = uploaded_referral_file.getvalue()
                 try: decoded_data_ref = bytes_data_ref.decode("utf-8")
                 except UnicodeDecodeError: decoded_data_ref = bytes_data_ref.decode("latin-1", errors="replace") 
                 stringio_ref = io.StringIO(decoded_data_ref)
                 try:
                      referrals_raw_df_main = pd.read_csv(stringio_ref, sep=',', header=0, on_bad_lines='warn', low_memory=False)
                      st.session_state.referral_data_processed = preprocess_referral_data(referrals_raw_df_main, funnel_definition, ordered_stages, ts_col_map)
                      if st.session_state.referral_data_processed is not None and not st.session_state.referral_data_processed.empty:
                           st.session_state.data_processed_successfully = True
                           st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(st.session_state.referral_data_processed, ordered_stages, ts_col_map)
                           st.session_state.site_metrics_calculated = calculate_site_metrics(st.session_state.referral_data_processed, ordered_stages, ts_col_map)
                      else: st.session_state.data_processed_successfully = False; st.error("Referral data preprocessing failed or returned empty.")
                 except pd.errors.ParserError as pe: st.error(f"Error parsing referral CSV file: {pe}. Check file format and delimiter."); st.exception(pe)
                 except Exception as read_err: st.error(f"Error reading referral file: {read_err}"); st.exception(read_err)
            except Exception as e: st.error(f"Error loading referral data: {e}"); st.exception(e)
        else: st.error("Funnel definition parsing failed. Cannot proceed with referral data.")

if st.session_state.data_processed_successfully:
    referral_data_processed = st.session_state.referral_data_processed
    ordered_stages = st.session_state.ordered_stages
    ts_col_map = st.session_state.ts_col_map
    inter_stage_lags_data = st.session_state.get('inter_stage_lags', {})
    site_metrics_calculated_data = st.session_state.get('site_metrics_calculated', pd.DataFrame())

    if "success_message_shown" not in st.session_state and referral_data_processed is not None:
        st.success("Data loaded and preprocessed successfully!"); st.session_state.success_message_shown = True
    st.markdown("---")
    tab1, tab2, tab3, tab_ai = st.tabs(["📅 Monthly ProForma", "🏆 Site Performance", "📈 Projections", "🤖 AI Forecast"])

    with tab1: 
        st.header("Monthly ProForma (Historical Cohorts)")
        if referral_data_processed is not None and ordered_stages is not None and ts_col_map is not None and ad_spend_input_dict:
            proforma_df_tab1 = calculate_proforma_metrics(referral_data_processed, ordered_stages, ts_col_map, ad_spend_input_dict)
            if not proforma_df_tab1.empty:
                proforma_display_tab1 = proforma_df_tab1.transpose(); proforma_display_tab1.columns = [str(col) for col in proforma_display_tab1.columns]
                format_dict_tab1 = {idx: ("${:,.2f}" if 'Cost' in idx or 'Spend' in idx else ("{:.1%}" if '%' in idx else "{:,.0f}")) for idx in proforma_display_tab1.index}
                st.dataframe(proforma_display_tab1.style.format(format_dict_tab1, na_rep='-'))
                try: csv_tab1 = proforma_df_tab1.reset_index().to_csv(index=False).encode('utf-8'); st.download_button(label="Download ProForma Data", data=csv_tab1, file_name='monthly_proforma.csv', mime='text/csv', key='dl_proforma_v2')
                except Exception as e_dl1: st.warning(f"ProForma download button error: {e_dl1}")
            else: st.warning("Could not generate ProForma table (check historical ad spend for relevant months).")
        else: st.warning("ProForma cannot be calculated until data is loaded and historical ad spend is entered.")

    with tab2: 
        st.header("Site Performance Ranking")
        if referral_data_processed is not None and ordered_stages is not None and ts_col_map is not None and weights_normalized and not site_metrics_calculated_data.empty:
            ranked_sites_df_tab2 = score_sites(site_metrics_calculated_data, weights_normalized)
            st.subheader("Site Ranking")
            display_cols_sites_tab2 = ['Site', 'Score', 'Grade', 'Total Qualified', 'PSA Count', 'StS Count', 'Appt Count', 'ICF Count',
                                   'Qual -> ICF %', 'POF -> PSA %', 'PSA -> StS %', 'StS -> Appt %', 'Appt -> ICF %',
                                   'Avg TTC (Days)', 'Avg Funnel Movement Steps', 'Site Screen Fail %', 'Lag Qual -> ICF (Days)', 'Site Projection Lag (Days)']
            display_cols_sites_exist_tab2 = [col for col in display_cols_sites_tab2 if col in ranked_sites_df_tab2.columns]
            final_ranked_display_tab2 = ranked_sites_df_tab2[display_cols_sites_exist_tab2].copy()
            if not final_ranked_display_tab2.empty:
                if 'Score' in final_ranked_display_tab2.columns: final_ranked_display_tab2['Score'] = final_ranked_display_tab2['Score'].round(1)
                for col_fmt in final_ranked_display_tab2.columns:
                    if '%' in col_fmt: final_ranked_display_tab2[col_fmt] = final_ranked_display_tab2[col_fmt].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '-')
                    elif 'Lag' in col_fmt or 'TTC' in col_fmt or 'Steps' in col_fmt: final_ranked_display_tab2[col_fmt] = final_ranked_display_tab2[col_fmt].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
                    elif 'Count' in col_fmt or 'Qualified' in col_fmt : final_ranked_display_tab2[col_fmt] = final_ranked_display_tab2[col_fmt].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x==x else '-') 
                st.dataframe(final_ranked_display_tab2.style.format(na_rep='-'))
                try: csv_tab2 = final_ranked_display_tab2.to_csv(index=False).encode('utf-8'); st.download_button(label="Download Site Ranking", data=csv_tab2, file_name='site_ranking.csv', mime='text/csv', key='dl_sites_v2')
                except Exception as e_dl2: st.warning(f"Site ranking download error: {e_dl2}")
            else: st.warning("Site ranking display table is empty after filtering columns.")
        elif site_metrics_calculated_data.empty: st.warning("Site metrics have not been calculated (e.g. no 'Site' column in data or error during calculation). Site performance cannot be shown.")
        else: st.warning("Site performance cannot be ranked until data is loaded and weights are set.")

    with tab3: 
        st.header("Projections (Based on Future Spend)")
        effective_proj_rates_tab3, method_desc_tab3 = determine_effective_projection_rates(
            referral_data_processed, ordered_stages, ts_col_map, rate_assumption_method_sidebar,
            rolling_window_months_sidebar, manual_proj_conv_rates_sidebar, inter_stage_lags_data, sidebar_display_area=None) 
        st.caption(f"**Projection Using: {method_desc_tab3} Conversion Rates**")
        if "Rolling" in method_desc_tab3 and not any(s in method_desc_tab3 for s in ["Failed", "No History", "Error"]):
            if isinstance(effective_proj_rates_tab3, dict) and effective_proj_rates_tab3:
                st.markdown("---"); st.write("Effective Rolling Rates Applied for this Projection (Adj. & Matured):")
                for key_r, val_r in effective_proj_rates_tab3.items():
                     if key_r in manual_proj_conv_rates_sidebar: st.text(f"- {key_r}: {val_r*100:.1f}%") 
        st.markdown("---")
        with st.expander("View Calculated Average Inter-Stage Lags & Maturity Periods Used (for Rolling Rates)"):
            if inter_stage_lags_data:
                lag_df_list_tab3 = []
                temp_maturity_periods_display_tab3 = {} 
                display_maturity_lag_multiplier_tab3 = 1.5; display_min_effective_maturity_tab3 = 20; display_default_maturity_tab3 = 45
                if inter_stage_lags_data: 
                    for r_key_disp_tab3 in manual_proj_conv_rates_sidebar.keys(): 
                        avg_lag_disp_tab3 = inter_stage_lags_data.get(r_key_disp_tab3)
                        if pd.notna(avg_lag_disp_tab3) and avg_lag_disp_tab3 > 0:
                            calc_mat_disp_tab3 = round(display_maturity_lag_multiplier_tab3 * avg_lag_disp_tab3)
                            temp_maturity_periods_display_tab3[r_key_disp_tab3] = max(calc_mat_disp_tab3, display_min_effective_maturity_tab3)
                        else: temp_maturity_periods_display_tab3[r_key_disp_tab3] = display_default_maturity_tab3
                for key_lag, val_lag in inter_stage_lags_data.items():
                    maturity_p_disp_tab3 = temp_maturity_periods_display_tab3.get(key_lag, "N/A (Not a projection rate segment)")
                    lag_df_list_tab3.append({'Stage Transition': key_lag, 'Avg Lag (Days)': f"{val_lag:.1f}" if pd.notna(val_lag) else "N/A", 
                                          'Implied Maturity Used (Days)': f"{maturity_p_disp_tab3}" if isinstance(maturity_p_disp_tab3, (int, float)) else maturity_p_disp_tab3})
                if lag_df_list_tab3: st.table(pd.DataFrame(lag_df_list_tab3))
                else: st.caption("No inter-stage lags calculated or available.")
            else: st.caption("Inter-stage lags have not been calculated (required for rolling rate maturity).")
        st.markdown("---")
        projection_inputs_tab3 = {
            'horizon': proj_horizon_sidebar, 'spend_dict': proj_spend_dict_sidebar,
            'cpqr_dict': proj_cpqr_dict_sidebar, 'final_conv_rates': effective_proj_rates_tab3, 
            'goal_icf': goal_icf_count_sidebar, 'site_performance_data': site_metrics_calculated_data,
            'inter_stage_lags': inter_stage_lags_data,
            'icf_variation_percentage': proj_icf_variation_percent_sidebar
        }
        projection_results_df_tab3, avg_lag_used_tab3, lpi_date_tab3, ads_off_tab3, site_level_proj_tab3, lag_msg_tab3 = calculate_projections(
            referral_data_processed, ordered_stages, ts_col_map, projection_inputs_tab3
        )
        st.markdown("---")
        col1_info_tab3, col2_info_tab3, col3_info_tab3 = st.columns(3)
        with col1_info_tab3: st.metric(label="Goal Total ICFs (Projections Tab)", value=f"{goal_icf_count_sidebar:,}")
        with col2_info_tab3: st.metric(label="Estimated LPI Date", value=lpi_date_tab3)
        with col3_info_tab3: st.metric(label="Estimated Ads Off Date", value=ads_off_tab3)
        if pd.notna(avg_lag_used_tab3): st.caption(f"Lag applied in projections: **{avg_lag_used_tab3:.1f} days**. ({lag_msg_tab3})")
        else: st.caption(f"Lag could not be determined. ({lag_msg_tab3})")
        st.markdown("---")
        if projection_results_df_tab3 is not None and not projection_results_df_tab3.empty:
            st.subheader("Projected Monthly ICFs & Cohort CPICF")
            cols_to_check_cpicf_tab3 = ['Projected_CPICF_Cohort_Source_Low', 'Projected_CPICF_Cohort_Source_Mean', 'Projected_CPICF_Cohort_Source_High']
            base_display_cols_proj_tab3 = ['Forecasted_Ad_Spend', 'Forecasted_Qual_Referrals', 'Projected_ICF_Landed']
            results_display_tab3 = projection_results_df_tab3.copy()
            if all(c in results_display_tab3.columns for c in cols_to_check_cpicf_tab3):
                results_display_tab3['Projected CPICF (Low-Mean-High)'] = results_display_tab3.apply(
                    lambda row: (f"${row['Projected_CPICF_Cohort_Source_Low']:,.2f} - ${row['Projected_CPICF_Cohort_Source_Mean']:,.2f} - ${row['Projected_CPICF_Cohort_Source_High']:,.2f}"
                                 if pd.notna(row['Projected_CPICF_Cohort_Source_Low']) and pd.notna(row['Projected_CPICF_Cohort_Source_Mean']) and pd.notna(row['Projected_CPICF_Cohort_Source_High'])
                                 else (f"${row['Projected_CPICF_Cohort_Source_Mean']:,.2f} (Range N/A)" if pd.notna(row['Projected_CPICF_Cohort_Source_Mean']) else "-"))
                                if proj_icf_variation_percent_sidebar > 0 else (f"${row['Projected_CPICF_Cohort_Source_Mean']:,.2f}" if pd.notna(row['Projected_CPICF_Cohort_Source_Mean']) else "-"), axis=1)
                final_display_cols_proj_tab3 = base_display_cols_proj_tab3 + ['Projected CPICF (Low-Mean-High)']
            elif 'Projected_CPICF_Cohort_Source_Mean' in results_display_tab3.columns: 
                results_display_tab3.rename(columns={'Projected_CPICF_Cohort_Source_Mean': 'Projected CPICF (Mean)'}, inplace=True)
                results_display_tab3['Projected CPICF (Mean)'] = results_display_tab3['Projected CPICF (Mean)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '-')
                final_display_cols_proj_tab3 = base_display_cols_proj_tab3 + ['Projected CPICF (Mean)']
            else: final_display_cols_proj_tab3 = base_display_cols_proj_tab3
            results_display_filtered_tab3 = results_display_tab3[[col for col in final_display_cols_proj_tab3 if col in results_display_tab3.columns]].copy()
            if not results_display_filtered_tab3.empty:
                results_display_filtered_tab3.index = results_display_filtered_tab3.index.strftime('%Y-%m')
                for col_name_fmt_tab3 in results_display_filtered_tab3.columns:
                    if 'Ad_Spend' in col_name_fmt_tab3: results_display_filtered_tab3[col_name_fmt_tab3] = results_display_filtered_tab3[col_name_fmt_tab3].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int,float)) and pd.notna(x) else (x if isinstance(x, str) else '-'))
                    elif 'Qual_Referrals' in col_name_fmt_tab3 or 'ICF_Landed' in col_name_fmt_tab3 : results_display_filtered_tab3[col_name_fmt_tab3] = results_display_filtered_tab3[col_name_fmt_tab3].apply(lambda x: f"{int(x):,}" if pd.notna(x) and isinstance(x, (int,float)) else (x if isinstance(x,str) else '-'))
                st.dataframe(results_display_filtered_tab3.style.format(na_rep='-'))
                if proj_icf_variation_percent_sidebar > 0 and 'Projected CPICF (Low-Mean-High)' in results_display_filtered_tab3.columns:
                    st.caption(f"Note: CPICF range based on +/- {proj_icf_variation_percent_sidebar}% ICF variation (set in sidebar).")
                if 'Projected_ICF_Landed' in projection_results_df_tab3.columns:
                     st.subheader("Projected ICFs Landed Over Time (Projections Tab)"); chart_data_tab3 = projection_results_df_tab3[['Projected_ICF_Landed']].copy()
                     if isinstance(chart_data_tab3.index, pd.PeriodIndex): chart_data_tab3.index = chart_data_tab3.index.to_timestamp()
                     chart_data_tab3['Projected_ICF_Landed'] = pd.to_numeric(chart_data_tab3['Projected_ICF_Landed'], errors='coerce').fillna(0)
                     st.line_chart(chart_data_tab3)
                try: csv_tab3_proj = results_display_filtered_tab3.reset_index().to_csv(index=False).encode('utf-8'); st.download_button(label="Download Projection Data", data=csv_tab3_proj, file_name='projection_spend_based.csv', mime='text/csv', key='dl_proj_v2')
                except Exception as e_dl3: st.warning(f"Projection download error: {e_dl3}")
            else: st.warning("Projection results table is empty after selecting columns.")
        else: st.warning("Could not calculate projections for Projections Tab.")
        st.markdown("---"); st.subheader("Site-Level Monthly Projections (Projections Tab - Editable)")
        if site_level_proj_tab3 is not None and not site_level_proj_tab3.empty:
            display_site_df_main_tab3 = site_level_proj_tab3.copy()
            if display_site_df_main_tab3.index.name == 'Site': display_site_df_main_tab3.reset_index(inplace=True) 
            else: display_site_df_main_tab3.index.name = 'Site'; display_site_df_main_tab3.reset_index(inplace=True)
            editable_sites_df_tab3 = display_site_df_main_tab3[display_site_df_main_tab3['Site'] != 'Grand Total'].copy() if 'Site' in display_site_df_main_tab3.columns else display_site_df_main_tab3.drop(index='Grand Total', errors='ignore').copy()
            total_row_df_display_tab3 = display_site_df_main_tab3[display_site_df_main_tab3['Site'] == 'Grand Total'] if 'Site' in display_site_df_main_tab3.columns else (display_site_df_main_tab3.loc[['Grand Total']] if 'Grand Total' in display_site_df_main_tab3.index else pd.DataFrame())
            if 'Site' in editable_sites_df_tab3.columns: 
                cols_site_editor_tab3 = ['Site'] + [col for col in editable_sites_df_tab3.columns if col != 'Site']
                editable_sites_df_tab3 = editable_sites_df_tab3[cols_site_editor_tab3]
            edited_site_level_df_tab3 = st.data_editor(editable_sites_df_tab3, use_container_width=True, key="site_level_editor_v2", num_rows="dynamic")
            if not total_row_df_display_tab3.empty:
                st.caption("Totals (based on initial calculation, not live edits from above table):")
                st.dataframe(total_row_df_display_tab3.style.format("{:,.0f}", na_rep='0'), use_container_width=True)
            try: csv_site_proj_tab3 = edited_site_level_df_tab3.to_csv(index=False).encode('utf-8'); st.download_button(label="Download Edited Site Projections", data=csv_site_proj_tab3, file_name='edited_site_level_projections.csv', mime='text/csv', key='dl_edited_site_proj_main_v2')
            except Exception as e_dl_site3: st.warning(f"Site projection download error: {e_dl_site3}")
        else: st.info("Site-level projection data (Projections Tab) is not available or is empty.")

    with tab_ai:
        st.header("🤖 AI Forecast (Goal-Based)")
        st.info("Define your recruitment goals, and the tool will estimate a plan. Assumes Cost Per Qualified Lead (POF) is consistent unless advanced options are used (future feature).")
        ai_cols = st.columns(3)
        with ai_cols[0]:
            ai_goal_lpi_date = st.date_input(
                "Target LPI Date", value=datetime.now() + pd.DateOffset(months=12), 
                min_value=datetime.now() + pd.DateOffset(months=1), key="ai_lpi_date"
            )
        with ai_cols[1]:
            ai_goal_icf_num = st.number_input("Target Total ICFs", min_value=1, value=100, step=10, key="ai_icf_num")
        with ai_cols[2]:
            ai_cpql_estimate = st.number_input(
                "Estimated Cost Per Qualified Lead (POF)", min_value=1.0, value=75.0, step=5.0, format="%.2f", key="ai_cpql",
                help="Your average cost to get one lead to the 'Passed Online Form' (or equivalent first funnel) stage."
            )
        ai_effective_rates, ai_rates_method_desc = determine_effective_projection_rates(
            referral_data_processed, ordered_stages, ts_col_map, 'Rolling Historical Average', 6, 
            manual_proj_conv_rates_sidebar, inter_stage_lags_data, sidebar_display_area=None
        )
        if "Error" in ai_rates_method_desc or "Failed" in ai_rates_method_desc or "No History" in ai_rates_method_desc or not ai_effective_rates or all(v == 0 for v in ai_effective_rates.values()):
            st.warning(f"Could not determine reliable historical rates for AI Forecast ({ai_rates_method_desc}). Using manual rates from sidebar as fallback.")
            ai_effective_rates = manual_proj_conv_rates_sidebar
            ai_rates_method_desc = "Manual (Fallback)"
        
        pof_ts_col_ai = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
        icf_ts_col_ai = ts_col_map.get(STAGE_SIGNED_ICF)
        avg_pof_icf_lag_ai = np.nan
        if pof_ts_col_ai and icf_ts_col_ai and pof_ts_col_ai in referral_data_processed.columns and icf_ts_col_ai in referral_data_processed.columns:
            avg_pof_icf_lag_ai = calculate_avg_lag_generic(referral_data_processed, pof_ts_col_ai, icf_ts_col_ai)
        if pd.isna(avg_pof_icf_lag_ai): 
            avg_pof_icf_lag_ai = avg_lag_used_tab3 if 'avg_lag_used_tab3' in locals() and pd.notna(avg_lag_used_tab3) else 30.0
            st.caption(f"AI Forecast using fallback POF-ICF lag: {avg_pof_icf_lag_ai:.1f} days. Rates: {ai_rates_method_desc}")
        else:
            st.caption(f"AI Forecast using calculated POF-ICF lag: {avg_pof_icf_lag_ai:.1f} days. Rates based on: {ai_rates_method_desc}")

        if st.button("🚀 Generate AI Forecast", key="run_ai_forecast"):
            if pd.isna(avg_pof_icf_lag_ai):
                st.error("Cannot run AI Forecast: Average POF to ICF lag is not available.")
            else:
                ai_results_df, ai_site_df, ai_ads_off, ai_message, ai_unfeasible = calculate_ai_forecast_core(
                    goal_lpi_date_dt=ai_goal_lpi_date, goal_icf_number=ai_goal_icf_num,
                    estimated_cpql=ai_cpql_estimate, icf_variation_percent=proj_icf_variation_percent_sidebar,
                    processed_df=referral_data_processed, ordered_stages=ordered_stages, ts_col_map=ts_col_map,
                    effective_projection_conv_rates=ai_effective_rates, avg_overall_lag_days=avg_pof_icf_lag_ai,
                    site_metrics_df=site_metrics_calculated_data, projection_horizon_months = proj_horizon_sidebar
                )
                st.markdown("---")
                if ai_unfeasible: st.warning(f"Feasibility Note: {ai_message}")
                else: st.success(f"Forecast Status: {ai_message}")
                ai_col1, ai_col2, ai_col3 = st.columns(3)
                ai_col1.metric("Target LPI Date", ai_goal_lpi_date.strftime("%Y-%m-%d"))
                ai_col2.metric("Target ICFs", f"{ai_goal_icf_num:,}")
                ai_col3.metric("Est. Ads Off Date", ai_ads_off if ai_ads_off != "N/A" else "Past LPI/Goal Unmet")

                if not ai_results_df.empty:
                    st.subheader("AI Forecasted Monthly Performance")
                    ai_display_df = ai_results_df.copy()
                    ai_display_df.index = ai_display_df.index.strftime('%Y-%m')
                    if all(c in ai_display_df.columns for c in ['Projected_CPICF_Cohort_Source_Low', 'Projected_CPICF_Cohort_Source_Mean', 'Projected_CPICF_Cohort_Source_High']):
                        ai_display_df['Projected CPICF (Low-Mean-High)'] = ai_display_df.apply(
                            lambda row: (f"${row['Projected_CPICF_Cohort_Source_Low']:,.2f} - ${row['Projected_CPICF_Cohort_Source_Mean']:,.2f} - ${row['Projected_CPICF_Cohort_Source_High']:,.2f}"
                                 if pd.notna(row['Projected_CPICF_Cohort_Source_Low']) and pd.notna(row['Projected_CPICF_Cohort_Source_Mean']) and pd.notna(row['Projected_CPICF_Cohort_Source_High'])
                                 else (f"${row['Projected_CPICF_Cohort_Source_Mean']:,.2f} (Range N/A)" if pd.notna(row['Projected_CPICF_Cohort_Source_Mean']) else "-"))
                                if proj_icf_variation_percent_sidebar > 0 else (f"${row['Projected_CPICF_Cohort_Source_Mean']:,.2f}" if pd.notna(row['Projected_CPICF_Cohort_Source_Mean']) else "-"), axis=1)
                        cols_to_show_ai = ['Target_QLs_POF', 'Implied_Ad_Spend', 'Projected_ICF_Landed', 'Projected CPICF (Low-Mean-High)']
                    else: 
                        cols_to_show_ai = ['Target_QLs_POF', 'Implied_Ad_Spend', 'Projected_ICF_Landed', 'Projected_CPICF_Cohort_Source_Mean']
                        if 'Projected_CPICF_Cohort_Source_Mean' in ai_display_df.columns:
                             ai_display_df.rename(columns={'Projected_CPICF_Cohort_Source_Mean':'Projected CPICF (Mean)'}, inplace=True)
                             ai_display_df['Projected CPICF (Mean)'] = ai_display_df['Projected CPICF (Mean)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else '-')
                    ai_display_df_filtered = ai_display_df[[col for col in cols_to_show_ai if col in ai_display_df.columns]].copy()
                    for col_n_ai in ai_display_df_filtered.columns:
                        if 'Ad_Spend' in col_n_ai : ai_display_df_filtered[col_n_ai] = ai_display_df_filtered[col_n_ai].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int,float)) and pd.notna(x) else (x if isinstance(x,str) else '-'))
                        elif 'QLs_POF' in col_n_ai or 'ICF_Landed' in col_n_ai: ai_display_df_filtered[col_n_ai] = ai_display_df_filtered[col_n_ai].apply(lambda x: f"{int(x):,}" if pd.notna(x) and isinstance(x,(int,float)) else (x if isinstance(x,str) else '-'))
                    st.dataframe(ai_display_df_filtered.style.format(na_rep='-'))
                    if proj_icf_variation_percent_sidebar > 0 and 'Projected CPICF (Low-Mean-High)' in ai_display_df_filtered.columns:
                        st.caption(f"Note: CPICF range based on +/- {proj_icf_variation_percent_sidebar}% ICF variation (set in sidebar).")
                    st.subheader("Projected ICFs Landed Over Time (AI Forecast)")
                    ai_chart_data = ai_results_df[['Projected_ICF_Landed']].copy()
                    if isinstance(ai_chart_data.index, pd.PeriodIndex): ai_chart_data.index = ai_chart_data.index.to_timestamp()
                    ai_chart_data['Projected_ICF_Landed'] = pd.to_numeric(ai_chart_data['Projected_ICF_Landed'], errors='coerce').fillna(0)
                    st.line_chart(ai_chart_data)
                if not ai_site_df.empty:
                    st.subheader("AI Forecasted Site-Level Performance")
                    ai_site_df_displayable = ai_site_df.copy()
                    # Make sure index is 'Site' if it's not already for display
                    if ai_site_df_displayable.index.name != 'Site' and 'Site' in ai_site_df_displayable.columns:
                        ai_site_df_displayable.set_index('Site', inplace=True)
                    elif ai_site_df_displayable.index.name != 'Site' and 'Site' not in ai_site_df_displayable.columns:
                         pass # Index is likely already site names, or it's 'Grand Total' only
                    
                    # Basic formatting for site table - can be enhanced
                    formatted_site_df_ai = ai_site_df_displayable.copy()
                    for col_site_ai in formatted_site_df_ai.columns:
                        if isinstance(col_site_ai, tuple) and 'Projected QLs (POF)' in col_site_ai[1] or 'Projected ICFs Landed' in col_site_ai[1]:
                            formatted_site_df_ai[col_site_ai] = formatted_site_df_ai[col_site_ai].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "-")

                    st.dataframe(formatted_site_df_ai.style.format(na_rep='-'))
                else: st.info("Site-level AI forecast not available or sites not defined.")
        else: st.caption("Click the button above to generate the AI forecast based on your goals.")
elif not uploaded_referral_file or not uploaded_funnel_def_file:
    st.info("👋 Welcome! Please upload both the Referral Data (CSV) and Funnel Definition (TSV) files using the sidebar to begin.")
