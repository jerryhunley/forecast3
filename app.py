# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
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
STAGE_SCREEN_FAILED = "Screen Failed"
# New constants for terminal stages
STAGE_ENROLLED = "Enrolled"
STAGE_LOST = "Lost"


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
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist if isinstance(name, str)])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist if isinstance(name, str)])
    try: all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError as e: pass
    for event_name, event_dt in all_events:
        if pd.isna(event_dt): continue
        event_stage = None;
        if event_name in ordered_stgs: event_stage = event_name
        elif event_name in status_to_stage_map: event_stage = status_to_stage_map[event_name]
        if event_stage and event_stage in ordered_stgs:
            ts_col_name = ts_col_mapping.get(event_stage)
            if ts_col_name and pd.isna(timestamps[ts_col_name]):
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
        else: submitted_on_col = "Submitted On"
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
        else: st.error("History columns found but failed to parse into structured data.")
        return None
    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist_col, parsed_status_hist_col, funnel_def, ordered_stages, ts_col_map), axis=1)
    old_ts_cols = [col for col in df.columns if col.startswith('TS_')]; df.drop(columns=old_ts_cols, inplace=True, errors='ignore')
    df = pd.concat([df, timestamp_cols_df], axis=1)
    for stage, ts_col in ts_col_map.items():
         if ts_col in df.columns: df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

    # Ensure UTM columns exist for Ad Performance Tab
    if "UTM Source" not in df.columns:
        # st.info("'UTM Source' column not found in data. Ad Performance tab may not function fully.") # User already warned if file missing
        df["UTM Source"] = np.nan # Create with NaNs if it doesn't exist
    if "UTM Medium" not in df.columns: # Keep this for potential future re-introduction
        # st.info("'UTM Medium' column not found in data.")
        df["UTM Medium"] = np.nan 
    return df

def calculate_proforma_metrics(_processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    if _processed_df is None or _processed_df.empty: return pd.DataFrame()
    if not isinstance(monthly_ad_spend_input, dict): st.warning("ProForma: Ad spend input is not a dictionary."); return pd.DataFrame()
    if "Submission_Month" not in _processed_df.columns: st.error("ProForma: 'Submission_Month' column missing."); return pd.DataFrame()
    processed_df = _processed_df.copy()
    try:
        cohort_summary = processed_df.groupby("Submission_Month").size().reset_index(name="Total Qualified Referrals_Calc")
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
        st.warning("Site column not found or data is empty. Cannot calculate site metrics.")
        return pd.DataFrame()
    
    site_metrics_df = calculate_grouped_performance_metrics(
        _processed_df, ordered_stages, ts_col_map,
        grouping_cols=["Site"], 
        unclassified_label="Unassigned Site"
    )
    # Rename columns for backward compatibility with existing scoring weights and display logic for "Site Performance"
    if "Projection Lag (Days)" in site_metrics_df.columns:
        site_metrics_df.rename(columns={"Projection Lag (Days)": "Site Projection Lag (Days)"}, inplace=True)
    if "Screen Fail % (from ICF)" in site_metrics_df.columns: # Make sure this matches the output from grouped_metrics
        site_metrics_df.rename(columns={"Screen Fail % (from ICF)": "Site Screen Fail %"}, inplace=True)
    
    # The calculate_grouped_performance_metrics will return the grouping col named as per grouping_cols[0] if len is 1
    # So, if grouping_cols was ["Site"], the column should already be "Site".
    # No need to rename "Site_Cleaned" to "Site" here if the generic function handles it as described.
    return site_metrics_df


# NEW: Generic function for calculating performance metrics based on grouping columns
def calculate_grouped_performance_metrics(_processed_df, ordered_stages, ts_col_map, grouping_cols: list, unclassified_label="Unclassified"):
    if _processed_df is None or _processed_df.empty:
        return pd.DataFrame()

    processed_df_copy = _processed_df.copy()
    
    # This will be the name of the column used for groupby and returned for scoring
    actual_grouping_col_name_for_return = "" 
    
    if len(grouping_cols) == 1:
        gc_single = grouping_cols[0]
        if gc_single not in processed_df_copy.columns:
            st.info(f"Grouping column '{gc_single}' not found. Creating it and labeling all as '{unclassified_label}'.")
            processed_df_copy[gc_single] = unclassified_label 
        
        # The column used for grouping will be the original name, after cleaning
        processed_df_copy[gc_single] = processed_df_copy[gc_single].astype(str).str.strip().replace('', unclassified_label).fillna(unclassified_label)
        actual_grouping_col_name_for_return = gc_single 
    
    # SIMPLIFIED: Removed logic for combining multiple grouping_cols as per new requirement
    # elif len(grouping_cols) > 1: 
        # ... (This part is removed as we only support single UTM Source for now) ...
    
    else: # No grouping columns or more than 1 (which is not supported for now by this simplification)
        st.error("Invalid grouping_cols specification for performance metrics calculation. Expecting a single column name.")
        return pd.DataFrame()

    performance_metrics_list = []
    ts_pof_col = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
    ts_psa_col = ts_col_map.get(STAGE_PRE_SCREENING_ACTIVITIES)
    ts_sts_col = ts_col_map.get(STAGE_SENT_TO_SITE)
    ts_appt_col = ts_col_map.get(STAGE_APPOINTMENT_SCHEDULED)
    ts_icf_col = ts_col_map.get(STAGE_SIGNED_ICF)
    ts_sf_col = ts_col_map.get(STAGE_SCREEN_FAILED)

    potential_ts_cols = [ts_pof_col, ts_psa_col, ts_sts_col, ts_appt_col, ts_icf_col, ts_sf_col]
    for col_name_ts in potential_ts_cols:
        if col_name_ts and col_name_ts not in processed_df_copy.columns:
            processed_df_copy[col_name_ts] = pd.NaT
            processed_df_copy[col_name_ts] = pd.to_datetime(processed_df_copy[col_name_ts], errors='coerce')

    projection_segments_for_lag_path = [
        (STAGE_PASSED_ONLINE_FORM, STAGE_PRE_SCREENING_ACTIVITIES),
        (STAGE_PRE_SCREENING_ACTIVITIES, STAGE_SENT_TO_SITE),
        (STAGE_SENT_TO_SITE, STAGE_APPOINTMENT_SCHEDULED),
        (STAGE_APPOINTMENT_SCHEDULED, STAGE_SIGNED_ICF)
    ]
    site_contact_attempt_statuses = ["Site Contact Attempt 1"] 
    post_sts_progress_stages = [STAGE_APPOINTMENT_SCHEDULED, STAGE_SIGNED_ICF, "Enrolled", STAGE_SCREEN_FAILED]

    try:
        grouped_data = processed_df_copy.groupby(actual_grouping_col_name_for_return) # Group by the cleaned original column name
        for group_name_val, group_df in grouped_data:
            metrics = {actual_grouping_col_name_for_return: group_name_val} 
            
            count_pof = group_df[ts_pof_col].notna().sum() if ts_pof_col and ts_pof_col in group_df else 0
            count_psa = group_df[ts_psa_col].notna().sum() if ts_psa_col and ts_psa_col in group_df else 0
            count_sts = group_df[ts_sts_col].notna().sum() if ts_sts_col and ts_sts_col in group_df else 0
            count_appt = group_df[ts_appt_col].notna().sum() if ts_appt_col and ts_appt_col in group_df else 0
            count_icf = group_df[ts_icf_col].notna().sum() if ts_icf_col and ts_icf_col in group_df else 0
            count_sf = group_df[ts_sf_col].notna().sum() if ts_sf_col and ts_sf_col in group_df else 0

            metrics['Total Qualified'] = count_pof; metrics['PSA Count'] = count_psa
            metrics['StS Count'] = count_sts; metrics['Appt Count'] = count_appt; metrics['ICF Count'] = count_icf
            metrics['POF -> PSA %'] = (count_psa / count_pof) if count_pof > 0 else 0.0
            metrics['PSA -> StS %'] = (count_sts / count_psa) if count_psa > 0 else 0.0
            metrics['StS -> Appt %'] = (count_appt / count_sts) if count_sts > 0 else 0.0
            metrics['Appt -> ICF %'] = (count_icf / count_appt) if count_appt > 0 else 0.0
            metrics['Qual -> ICF %'] = (count_icf / count_pof) if count_pof > 0 else 0.0

            group_total_projection_lag = 0.0; valid_lag_segments_group = 0
            for seg_from_name, seg_to_name in projection_segments_for_lag_path:
                ts_seg_from = ts_col_map.get(seg_from_name); ts_seg_to = ts_col_map.get(seg_to_name)
                if ts_seg_from and ts_seg_to and ts_seg_from in group_df.columns and ts_seg_to in group_df.columns:
                    segment_lag = calculate_avg_lag_generic(group_df, ts_seg_from, ts_seg_to)
                    if pd.notna(segment_lag): group_total_projection_lag += segment_lag; valid_lag_segments_group +=1
                    else: group_total_projection_lag = np.nan; break
                else: group_total_projection_lag = np.nan; break
            if valid_lag_segments_group < len(projection_segments_for_lag_path): group_total_projection_lag = np.nan
            
            if actual_grouping_col_name_for_return == "Site": # If it's for sites
                 metrics['Site Projection Lag (Days)'] = group_total_projection_lag
                 metrics['Site Screen Fail %'] = (count_sf / count_icf) if count_icf > 0 else 0.0
                 ttc_times_group = []; funnel_movement_steps_group = []
                 parsed_status_col_name = f"Parsed_Lead_Status_History"; parsed_stage_col_name = f"Parsed_Lead_Stage_History"
                 sent_to_site_group_df = group_df.dropna(subset=[ts_sts_col]) if ts_sts_col and ts_sts_col in group_df else pd.DataFrame()
                 if not sent_to_site_group_df.empty and parsed_status_col_name in sent_to_site_group_df.columns and parsed_stage_col_name in sent_to_site_group_df.columns:
                     for idx, row in sent_to_site_group_df.iterrows():
                         ts_sent = row[ts_sts_col]; first_contact_ts = pd.NaT
                         history_list_status = row.get(parsed_status_col_name, [])
                         if history_list_status:
                             for status_name, event_dt in history_list_status:
                                 if status_name in site_contact_attempt_statuses and pd.notna(event_dt) and pd.notna(ts_sent) and event_dt >= ts_sent:
                                     first_contact_ts = event_dt; break
                         if pd.notna(first_contact_ts) and pd.notna(ts_sent):
                             time_diff = first_contact_ts - ts_sent
                             if time_diff >= pd.Timedelta(0): ttc_times_group.append(time_diff.total_seconds() / (60*60*24))
                         stages_reached_post_sts = set()
                         history_list_stage = row.get(parsed_stage_col_name, [])
                         if history_list_stage and pd.notna(ts_sent):
                              for stage_name_hist, event_dt_hist in history_list_stage:
                                  if stage_name_hist in post_sts_progress_stages and pd.notna(event_dt_hist) and event_dt_hist > ts_sent:
                                      stages_reached_post_sts.add(stage_name_hist)
                         funnel_movement_steps_group.append(len(stages_reached_post_sts))
                 metrics['Avg TTC (Days)'] = np.mean(ttc_times_group) if ttc_times_group else np.nan
                 metrics['Avg Funnel Movement Steps'] = np.mean(funnel_movement_steps_group) if funnel_movement_steps_group else 0.0
            else: # For other groupings like Ad Performance
                 metrics['Projection Lag (Days)'] = group_total_projection_lag
                 metrics['Screen Fail % (from ICF)'] = (count_sf / count_icf) if count_icf > 0 else 0.0
                 metrics['Avg TTC (Days)'] = np.nan 
                 metrics['Avg Funnel Movement Steps'] = 0.0

            metrics['Lag Qual -> ICF (Days)'] = calculate_avg_lag_generic(group_df, ts_pof_col, ts_icf_col) if ts_pof_col and ts_icf_col else np.nan
            performance_metrics_list.append(metrics)

        performance_df_final = pd.DataFrame(performance_metrics_list)
        return performance_df_final
    except Exception as e:
        st.error(f"Error calculating grouped performance metrics for '{actual_grouping_col_name_for_return}': {e}"); st.exception(e)
        return pd.DataFrame()


def score_performance_groups(_performance_metrics_df, weights, group_col_name): 
    if _performance_metrics_df is None or _performance_metrics_df.empty: return pd.DataFrame()
    try:
        performance_metrics_df = _performance_metrics_df.copy()
        
        if group_col_name not in performance_metrics_df.columns:
             st.error(f"Scoring: Grouping column '{group_col_name}' missing. Available: {performance_metrics_df.columns.tolist()}");
             return _performance_metrics_df 

        performance_metrics_df[group_col_name] = performance_metrics_df[group_col_name].astype(str).fillna("Unknown Group")
        if performance_metrics_df[group_col_name].duplicated().any():
            performance_metrics_df = performance_metrics_df.drop_duplicates(subset=[group_col_name], keep='first')

        performance_metrics_df_indexed = performance_metrics_df.set_index(group_col_name)

        metrics_to_scale = list(weights.keys())
        lower_is_better = ["Avg TTC (Days)", "Screen Fail % (from ICF)", "Site Screen Fail %", 
                           "Lag Qual -> ICF (Days)", "Projection Lag (Days)", "Site Projection Lag (Days)"]

        scaled_metrics_data = performance_metrics_df_indexed.reindex(columns=metrics_to_scale).copy()
        for col in metrics_to_scale:
            if col not in scaled_metrics_data.columns:
                scaled_metrics_data[col] = 0 if col not in lower_is_better else np.nan
            if col in lower_is_better:
                max_val = scaled_metrics_data[col].max(skipna=True);
                std_val = scaled_metrics_data[col].std(skipna=True)
                fill_val = (max_val + std_val) if pd.notna(max_val) and max_val > 0 and pd.notna(std_val) and std_val > 0 else (max_val * 1.5 if pd.notna(max_val) and max_val > 0 else 999)
                scaled_metrics_data[col] = scaled_metrics_data[col].fillna(fill_val)
            else:
                scaled_metrics_data[col] = scaled_metrics_data[col].fillna(0)

        scaled_metrics_display = pd.DataFrame(index=scaled_metrics_data.index)
        if not scaled_metrics_data.empty:
            for col in metrics_to_scale:
                 if col in scaled_metrics_data.columns:
                     min_val = scaled_metrics_data[col].min(); max_val = scaled_metrics_data[col].max()
                     if min_val == max_val : scaled_metrics_display[col] = 0.5
                     elif pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val) != 0 :
                         scaler = MinMaxScaler();
                         scaled_values = scaler.fit_transform(scaled_metrics_data[[col]]);
                         scaled_metrics_display[col] = scaled_values.flatten()
                     else: scaled_metrics_display[col] = 0.5
                 else: scaled_metrics_display[col] = 0.5

            for col in lower_is_better:
                if col in scaled_metrics_display.columns: scaled_metrics_display[col] = 1 - scaled_metrics_display[col]

        performance_metrics_df_indexed['Score_Raw'] = 0.0; total_weight_applied = 0.0
        for metric, weight_value in weights.items():
             if metric in scaled_metrics_display.columns: 
                 current_scaled_metric_series = scaled_metrics_display.get(metric)
                 if current_scaled_metric_series is not None:
                     performance_metrics_df_indexed['Score_Raw'] += current_scaled_metric_series.fillna(0.5) * weight_value
                 else: performance_metrics_df_indexed['Score_Raw'] += 0.5 * weight_value
                 total_weight_applied += abs(weight_value)

        if total_weight_applied > 0: performance_metrics_df_indexed['Score'] = (performance_metrics_df_indexed['Score_Raw'] / total_weight_applied) * 100
        else: performance_metrics_df_indexed['Score'] = 0.0

        performance_metrics_df_indexed['Score'] = performance_metrics_df_indexed['Score'].fillna(0.0)

        if len(performance_metrics_df_indexed) > 1:
            performance_metrics_df_indexed['Score_Rank_Percentile'] = performance_metrics_df_indexed['Score'].rank(pct=True)
            bins = [0, 0.10, 0.25, 0.60, 0.85, 1.0]; labels = ['F', 'D', 'C', 'B', 'A']
            try: performance_metrics_df_indexed['Grade'] = pd.qcut(performance_metrics_df_indexed['Score_Rank_Percentile'], q=bins, labels=labels, duplicates='drop')
            except ValueError:
                 st.warning(f"Using fixed score ranges for grading '{performance_metrics_df_indexed.index.name}' (percentile method failed).")
                 def assign_grade_fallback(score_value):
                     if pd.isna(score_value): return 'N/A'
                     score_value = round(score_value)
                     if score_value >= 90: return 'A' 
                     elif score_value >= 80: return 'B' 
                     elif score_value >= 70: return 'C' 
                     elif score_value >= 60: return 'D' 
                     else: return 'F'
                 performance_metrics_df_indexed['Grade'] = performance_metrics_df_indexed['Score'].apply(assign_grade_fallback)
            performance_metrics_df_indexed['Grade'] = performance_metrics_df_indexed['Grade'].astype(str).replace('nan', 'N/A')
        elif len(performance_metrics_df_indexed) == 1:
            def assign_single_group_grade(score_value):
                if pd.isna(score_value): return 'N/A'
                score_value = round(score_value)
                if score_value >= 90: return 'A' 
                elif score_value >= 80: return 'B' 
                elif score_value >= 70: return 'C' 
                elif score_value >= 60: return 'D' 
                else: return 'F'
            performance_metrics_df_indexed['Grade'] = performance_metrics_df_indexed['Score'].apply(assign_single_group_grade)
        else: performance_metrics_df_indexed['Grade'] = None

        final_df_output = performance_metrics_df_indexed.reset_index() # This puts `group_col_name` back as a column
        if 'Score' in final_df_output.columns: final_df_output = final_df_output.sort_values('Score', ascending=False)
        return final_df_output
    except Exception as e:
        st.error(f"Error during Performance Group Scoring: {e}"); st.exception(e)
        if _performance_metrics_df is not None and not _performance_metrics_df.empty:
             if _performance_metrics_df.index.name == group_col_name and group_col_name not in _performance_metrics_df.columns:
                 return _performance_metrics_df.reset_index()
             return _performance_metrics_df
        return pd.DataFrame()

def score_sites(_site_metrics_df, weights):
    # calculate_site_metrics ensures the grouping column is named "Site"
    return score_performance_groups(_site_metrics_df, weights, group_col_name="Site")

# app.py (Continuation of Part 1 - which includes corrected calculate_grouped_performance_metrics and score_performance_groups)

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
            else: actual_col_from = reached_stage_cols_map_hist.get(stage_from_name)
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
                        elif denominator_val > 0: rate_for_this_month_calc = numerator_val / denominator_val
                        adjusted_monthly_rates_list.append(rate_for_this_month_calc)
                    else: substitutions_made_log.append(f"Mth {month_period_loop.strftime('%Y-%m')} for '{rate_key}': Excluded (too recent, maturity: {maturity_days_for_this_rate}d).")
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
    required_keys = ['horizon', 'spend_dict', 'cpqr_dict', 'final_conv_rates', 'goal_icf',
                     'site_performance_data', 'inter_stage_lags', 'icf_variation_percentage']
    if not isinstance(projection_inputs, dict) or not all(k in projection_inputs for k in required_keys):
        missing_keys = [k for k in required_keys if k not in projection_inputs]
        st.warning(f"Proj: Missing inputs. Need: {missing_keys}. Got: {list(projection_inputs.keys())}")
        return default_return_tuple

    processed_df = _processed_df.copy(); horizon = projection_inputs['horizon']
    future_spend_dict = projection_inputs['spend_dict']; assumed_cpqr_dict = projection_inputs['cpqr_dict']
    final_projection_conv_rates = projection_inputs['final_conv_rates']; goal_total_icfs = projection_inputs['goal_icf']
    site_performance_data = projection_inputs['site_performance_data']
    inter_stage_lags = projection_inputs.get('inter_stage_lags', {}); icf_variation_percent = projection_inputs.get('icf_variation_percentage', 0)
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
                if pd.notna(lag_value): calculated_sum_of_lags += lag_value; valid_segments_count += 1
                else: all_segments_found_and_valid = False; break
            # else: all_segments_found_and_valid = False; break
        if all_segments_found_and_valid and valid_segments_count == len(projection_segments_for_lag_path):
            avg_actual_lag_days_for_display = calculated_sum_of_lags
            lag_calculation_method_message = "Using summed inter-stage lags for POF->ICF projection path."
        else:
            all_segments_found_and_valid = False
            lag_calculation_method_message = "Summed inter-stage lag for POF->ICF path failed (missing segments or lags). "
    else: all_segments_found_and_valid = False; lag_calculation_method_message = "Inter-stage lags not available. "

    if not all_segments_found_and_valid or pd.isna(avg_actual_lag_days_for_display):
        start_stage_for_overall_lag = ordered_stages[0] if ordered_stages and len(ordered_stages) > 0 else None
        end_stage_for_overall_lag = STAGE_SIGNED_ICF
        overall_lag_calc_val = np.nan
        if start_stage_for_overall_lag and ts_col_map.get(start_stage_for_overall_lag) and ts_col_map.get(end_stage_for_overall_lag):
            ts_col_start_overall = ts_col_map[start_stage_for_overall_lag]; ts_col_end_overall = ts_col_map[end_stage_for_overall_lag]
            if ts_col_start_overall in processed_df.columns and ts_col_end_overall in processed_df.columns:
                overall_lag_calc_val = calculate_avg_lag_generic(processed_df, ts_col_start_overall, ts_col_end_overall)
        if pd.notna(overall_lag_calc_val):
            avg_actual_lag_days_for_display = overall_lag_calc_val
            lag_calculation_method_message += f"Used historical overall lag ({start_stage_for_overall_lag} -> {end_stage_for_overall_lag})."
        else: avg_actual_lag_days_for_display = 30.0; lag_calculation_method_message += "Used default lag (30 days)."
    if pd.isna(avg_actual_lag_days_for_display): avg_actual_lag_days_for_display = 30.0; lag_calculation_method_message = "Critical Lag Error: All methods failed. Used default 30 days."

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
        icf_proj_col_name_base = "" ; current_stage_count_col = 'Forecasted_PSQ'
        for stage_from, stage_to in projection_segments_for_lag_path:
            conv_rate_key = f"{stage_from} -> {stage_to}"; conv_rate = final_projection_conv_rates.get(conv_rate_key, 0.0)
            proj_col_to_name = f"Projected_{stage_to.replace(' ', '_').replace('(', '').replace(')', '')}"
            if current_stage_count_col in projection_cohorts.columns:
                proj_counts_for_to_stage = (projection_cohorts[current_stage_count_col] * conv_rate)
                projection_cohorts[proj_col_to_name] = proj_counts_for_to_stage.round(0).fillna(0).astype(int)
                current_stage_count_col = proj_col_to_name
            else: projection_cohorts[proj_col_to_name] = 0; current_stage_count_col = proj_col_to_name
            if stage_to == STAGE_SIGNED_ICF: icf_proj_col_name_base = proj_col_to_name; break

        projection_results = pd.DataFrame(index=future_months); projection_results['Projected_ICF_Landed'] = 0.0
        if not icf_proj_col_name_base or icf_proj_col_name_base not in projection_cohorts.columns:
            st.error(f"Critical Error: Projected ICF column ('{icf_proj_col_name_base}') not found after funnel progression.");
            return default_return_tuple[0], avg_actual_lag_days_for_display, "Error", "Error", pd.DataFrame(), "ICF Proj Col Missing after funnel calc"

        icf_proj_col_low = f"{icf_proj_col_name_base}_low"; icf_proj_col_high = f"{icf_proj_col_name_base}_high"
        projection_cohorts[icf_proj_col_low] = (projection_cohorts[icf_proj_col_name_base] * (1 - variation_factor)).round(0).astype(int).clip(lower=0)
        projection_cohorts[icf_proj_col_high] = (projection_cohorts[icf_proj_col_name_base] * (1 + variation_factor)).round(0).astype(int).clip(lower=0)
        projection_cohorts['Projected_CPICF_Cohort_Mean'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_name_base].replace(0, np.nan)).round(2)
        projection_cohorts['Projected_CPICF_Cohort_Low'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_high].replace(0, np.nan)).round(2)
        projection_cohorts['Projected_CPICF_Cohort_High'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_low].replace(0, np.nan)).round(2)

        overall_current_lag_days_to_use = avg_actual_lag_days_for_display; days_in_avg_month = 30.4375
        for cohort_start_month_iter, row_data in projection_cohorts.iterrows():
            icfs_from_this_cohort_val = row_data[icf_proj_col_name_base]
            if icfs_from_this_cohort_val == 0: continue
            full_lag_months_val = int(np.floor(overall_current_lag_days_to_use / days_in_avg_month))
            remaining_lag_days_comp_val = overall_current_lag_days_to_use - (full_lag_months_val * days_in_avg_month)
            fraction_for_next_month_val = remaining_lag_days_comp_val / days_in_avg_month; fraction_for_current_offset_month_val = 1.0 - fraction_for_next_month_val
            icfs_month_1_val = icfs_from_this_cohort_val * fraction_for_current_offset_month_val
            icfs_month_2_val = icfs_from_this_cohort_val * fraction_for_next_month_val
            landing_month_1_period_val = cohort_start_month_iter + full_lag_months_val; landing_month_2_period_val = cohort_start_month_iter + full_lag_months_val + 1
            if landing_month_1_period_val in projection_results.index: projection_results.loc[landing_month_1_period_val, 'Projected_ICF_Landed'] += icfs_month_1_val
            if landing_month_2_period_val in projection_results.index: projection_results.loc[landing_month_2_period_val, 'Projected_ICF_Landed'] += icfs_month_2_val
        projection_results['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed'].round(0).fillna(0).astype(int)
        projection_results['Cumulative_ICF_Landed'] = projection_results['Projected_ICF_Landed'].cumsum()

        lpi_month_series_val = projection_results[projection_results['Cumulative_ICF_Landed'] >= goal_total_icfs]
        if not lpi_month_series_val.empty:
            lpi_month_period_val = lpi_month_series_val.index[0]; icfs_in_lpi_month_val = projection_results.loc[lpi_month_period_val, 'Projected_ICF_Landed']
            cumulative_before_lpi_direct_val = projection_results['Cumulative_ICF_Landed'].shift(1).fillna(0).loc[lpi_month_period_val]
            icfs_needed_in_lpi_month_val = goal_total_icfs - cumulative_before_lpi_direct_val
            if icfs_in_lpi_month_val > 0:
                fraction_of_lpi_month_val = max(0,min(1, icfs_needed_in_lpi_month_val / icfs_in_lpi_month_val))
                lpi_day_offset_val = int(np.ceil(fraction_of_lpi_month_val * days_in_avg_month)); lpi_day_offset_val = max(1, lpi_day_offset_val)
                lpi_date_val_calc = lpi_month_period_val.start_time + pd.Timedelta(days=lpi_day_offset_val -1); lpi_date_str = lpi_date_val_calc.strftime('%Y-%m-%d')
            elif icfs_needed_in_lpi_month_val <= 0:
                 lpi_date_str = (lpi_month_period_val.start_time - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            else: lpi_date_str = lpi_month_period_val.start_time.strftime('%Y-%m-%d')

        projection_cohorts['Cumulative_Projected_ICF_Generated'] = projection_cohorts[icf_proj_col_name_base].cumsum()
        ads_off_s_granular = projection_cohorts[projection_cohorts['Cumulative_Projected_ICF_Generated'] >= (goal_total_icfs - 0.5 + 1e-9)]
        if not ads_off_s_granular.empty:
            ads_off_month_period_granular = ads_off_s_granular.index[0]
            icfs_gen_by_ads_off_month_granular = projection_cohorts.loc[ads_off_month_period_granular, icf_proj_col_name_base]
            cum_gen_before_ads_off_month_granular = projection_cohorts.loc[ads_off_month_period_granular, 'Cumulative_Projected_ICF_Generated'] - icfs_gen_by_ads_off_month_granular
            icfs_needed_from_ads_off_month_granular = goal_total_icfs - cum_gen_before_ads_off_month_granular
            if icfs_needed_from_ads_off_month_granular <= 0:
                prev_m_ads_off_granular = ads_off_month_period_granular - 1
                if prev_m_ads_off_granular in projection_cohorts.index and cum_gen_before_ads_off_month_granular >= (goal_total_icfs - 0.5 + 1e-9):
                     ads_off_date_str = prev_m_ads_off_granular.end_time.strftime('%Y-%m-%d')
                else: ads_off_date_str = ads_off_month_period_granular.start_time.strftime('%Y-%m-%d')
            elif icfs_gen_by_ads_off_month_granular > 1e-9:
                fraction_needed_granular = max(0, min(1, icfs_needed_from_ads_off_month_granular / icfs_gen_by_ads_off_month_granular))
                day_offset_granular = int(np.ceil(fraction_needed_granular * days_in_avg_month)); day_offset_granular = max(1,day_offset_granular)
                ads_off_date_str = (ads_off_month_period_granular.start_time + pd.Timedelta(days=day_offset_granular -1)).strftime('%Y-%m-%d')
            else: ads_off_date_str = ads_off_month_period_granular.start_time.strftime('%Y-%m-%d')

        display_df_out = pd.DataFrame(index=future_months)
        display_df_out['Forecasted_Ad_Spend'] = projection_cohorts['Forecasted_Ad_Spend']
        display_df_out['Forecasted_Qual_Referrals'] = projection_cohorts['Forecasted_PSQ']
        display_df_out['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed']
        cpicf_display_mean_val = pd.Series(index=future_months, dtype=float); cpicf_display_low_val = pd.Series(index=future_months, dtype=float); cpicf_display_high_val = pd.Series(index=future_months, dtype=float)
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
                historical_site_pof_proportions = pd.Series(dtype=float)
                pof_ts_col_for_prop = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
                if pof_ts_col_for_prop and pof_ts_col_for_prop in _processed_df.columns:
                    valid_pof_df = _processed_df[_processed_df[pof_ts_col_for_prop].notna()]
                    if not valid_pof_df.empty:
                         historical_site_pof_proportions = valid_pof_df['Site'].value_counts(normalize=True)

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

                    site_pof_allocations_this_cohort = {}
                    if not historical_site_pof_proportions.empty:
                        for site_name_iter_s_proj in all_sites_proj:
                            site_prop_s_proj = historical_site_pof_proportions.get(site_name_iter_s_proj, 0)
                            site_pof_allocations_this_cohort[site_name_iter_s_proj] = total_psq_this_cohort_s_proj * site_prop_s_proj
                    elif len(all_sites_proj) > 0:
                        equal_share = total_psq_this_cohort_s_proj / len(all_sites_proj)
                        for site_name_iter_s_proj in all_sites_proj:
                            site_pof_allocations_this_cohort[site_name_iter_s_proj] = equal_share

                    current_sum_pof_float = sum(site_pof_allocations_this_cohort.values())
                    if abs(current_sum_pof_float - total_psq_this_cohort_s_proj) > 1e-6 and current_sum_pof_float > 0:
                        rescale_factor = total_psq_this_cohort_s_proj / current_sum_pof_float
                        for site_n in site_pof_allocations_this_cohort: site_pof_allocations_this_cohort[site_n] *= rescale_factor

                    rounded_site_pof_allocations = {site: round(val) for site, val in site_pof_allocations_this_cohort.items()}
                    diff_after_round = total_psq_this_cohort_s_proj - sum(rounded_site_pof_allocations.values())

                    if diff_after_round != 0 and site_pof_allocations_this_cohort:
                        site_to_adjust = max(site_pof_allocations_this_cohort, key=lambda s: site_pof_allocations_this_cohort[s] - math.floor(site_pof_allocations_this_cohort[s]))
                        rounded_site_pof_allocations[site_to_adjust] += diff_after_round

                    for site_name_iter_s_proj in all_sites_proj:
                        site_proj_pof_cohort_s_proj = rounded_site_pof_allocations.get(site_name_iter_s_proj, 0)
                        month_str_cohort_start_s_proj = cohort_start_month_s_proj.strftime('%Y-%m')
                        site_data_collector_proj[site_name_iter_s_proj][(month_str_cohort_start_s_proj, 'Projected Qual. Referrals (POF)')] += site_proj_pof_cohort_s_proj

                        site_perf_row_s_proj = site_performance_data[site_performance_data['Site'] == site_name_iter_s_proj]
                        current_site_proj_count_s_proj = float(site_proj_pof_cohort_s_proj)

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
                                if pd.notna(site_specific_rate_val_s) and site_specific_rate_val_s >= 0:
                                    rate_to_use_s = site_specific_rate_val_s

                            current_site_proj_count_s_proj *= rate_to_use_s
                            if stage_to_seg_s == STAGE_SIGNED_ICF: break

                        site_proj_icfs_generated_this_cohort_s = current_site_proj_count_s_proj

                        lag_to_use_for_site_s = overall_current_lag_days_to_use
                        if not site_perf_row_s_proj.empty and 'Site Projection Lag (Days)' in site_perf_row_s_proj.columns:
                            site_specific_lag_val_s = site_perf_row_s_proj['Site Projection Lag (Days)'].iloc[0]
                            if pd.notna(site_specific_lag_val_s) and site_specific_lag_val_s >=0: lag_to_use_for_site_s = site_specific_lag_val_s

                        if site_proj_icfs_generated_this_cohort_s > 0:
                            full_lag_m_site_s = int(np.floor(lag_to_use_for_site_s / days_in_avg_month))
                            remain_lag_days_comp_site_s = lag_to_use_for_site_s - (full_lag_m_site_s * days_in_avg_month)
                            frac_next_m_site_s = remain_lag_days_comp_site_s / days_in_avg_month; frac_curr_m_site_s = 1.0 - frac_next_m_site_s

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
                        icf_landed_col_tuple = (m_period_fmt_s_proj, 'Projected ICFs Landed')
                        if icf_landed_col_tuple in site_level_projections_df_final.columns:
                            site_level_projections_df_final[icf_landed_col_tuple] = site_level_projections_df_final[icf_landed_col_tuple].round(0).astype(int)

                        pof_col_tuple = (m_period_fmt_s_proj, 'Projected Qual. Referrals (POF)')
                        if pof_col_tuple in site_level_projections_df_final.columns:
                            site_level_projections_df_final[pof_col_tuple] = site_level_projections_df_final[pof_col_tuple].astype(int)

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


# --- MODIFIED: AI Forecast Core Function with Site Activation/Deactivation ---
def calculate_ai_forecast_core(
    goal_lpi_date_dt_orig: datetime, goal_icf_number_orig: int, estimated_cpql_user: float,
    icf_variation_percent: float,
    processed_df: pd.DataFrame, ordered_stages: list, ts_col_map: dict,
    effective_projection_conv_rates: dict,
    avg_overall_lag_days: float,
    site_metrics_df: pd.DataFrame, projection_horizon_months: int,
    site_caps_input: dict,
    # NEW: Site Activation/Deactivation Schedule
    site_activity_schedule: dict, # Expected format: {'Site A': {'activation_period': pd.Period, 'deactivation_period': pd.Period}, ...}
    site_scoring_weights_for_ai: dict,
    cpql_inflation_factor_pct: float,
    ql_vol_increase_threshold_pct: float,
    run_mode: str = "primary",
    ai_monthly_ql_capacity_multiplier: float = 3.0,
    ai_lag_method: str = "average",
    ai_lag_p25_days: float = None,
    ai_lag_p50_days: float = None,
    ai_lag_p75_days: float = None
):
    default_return_ai = pd.DataFrame(), pd.DataFrame(), "N/A", "Not Calculated", True, 0
    days_in_avg_m = 30.4375

    if not all([processed_df is not None, not processed_df.empty, ordered_stages, ts_col_map,
                effective_projection_conv_rates, site_metrics_df is not None]): # site_metrics_df can be empty
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Missing critical base data for AI Forecast.", True, 0
    if goal_icf_number_orig <= 0: return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Goal ICF number must be positive.", True, 0
    if estimated_cpql_user <= 0: return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Estimated CPQL must be positive.", True, 0
    if cpql_inflation_factor_pct < 0 or ql_vol_increase_threshold_pct < 0:
         return default_return_ai[0], default_return_ai[1], default_return_ai[2], "CPQL inflation parameters cannot be negative.", True, 0
    if ai_monthly_ql_capacity_multiplier <=0: ai_monthly_ql_capacity_multiplier = 1.0

    effective_lag_for_planning_approx = avg_overall_lag_days
    if ai_lag_method == "percentiles":
        if not all(pd.notna(lag_val) and lag_val >= 0 for lag_val in [ai_lag_p25_days, ai_lag_p50_days, ai_lag_p75_days]):
            return default_return_ai[0], default_return_ai[1], default_return_ai[2], "P25/P50/P75 lag days must be valid non-negative numbers.", True, 0
        if not (ai_lag_p25_days <= ai_lag_p50_days <= ai_lag_p75_days):
            return default_return_ai[0], default_return_ai[1], default_return_ai[2], "P25 lag must be <= P50, and P50 <= P75.", True, 0
        effective_lag_for_planning_approx = ai_lag_p50_days
    elif pd.isna(avg_overall_lag_days) or avg_overall_lag_days < 0:
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Average POF to ICF lag is invalid.", True, 0

    ts_pof_col_for_prop = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
    main_funnel_path_segments = [
        f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}",
        f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}",
        f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}",
        f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"
    ]
    overall_pof_to_icf_rate = 1.0
    for segment in main_funnel_path_segments:
        rate = effective_projection_conv_rates.get(segment)
        if rate is None or rate < 0:
            return default_return_ai[0], default_return_ai[1], default_return_ai[2], f"Conversion rate for segment '{segment}' invalid.", True, 0
        overall_pof_to_icf_rate *= rate
    if overall_pof_to_icf_rate <= 1e-9:
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Overall POF to ICF conversion rate is zero.", True, 0

    last_hist_month = processed_df["Submission_Month"].max() if "Submission_Month" in processed_df and not processed_df["Submission_Month"].empty else pd.Period(datetime.now(), freq='M') - 1
    proj_start_month_period = last_hist_month + 1
    current_goal_icf_number = goal_icf_number_orig
    if run_mode == "primary":
        current_goal_lpi_month_period = pd.Period(goal_lpi_date_dt_orig, freq='M')
    elif run_mode == "best_case_extended_lpi":
        current_goal_lpi_month_period = proj_start_month_period + projection_horizon_months -1 # Max LPI
    else: current_goal_lpi_month_period = pd.Period(goal_lpi_date_dt_orig, freq='M')

    avg_lag_months_approx = int(round(effective_lag_for_planning_approx / days_in_avg_m))
    max_possible_proj_end_month_overall = proj_start_month_period + projection_horizon_months -1
    calc_horizon_end_month_for_display = min(max_possible_proj_end_month_overall, current_goal_lpi_month_period + avg_lag_months_approx + 3)
    calc_horizon_end_month_for_display = max(calc_horizon_end_month_for_display, proj_start_month_period)
    projection_calc_months = pd.period_range(start=proj_start_month_period, end=calc_horizon_end_month_for_display, freq='M')

    if projection_calc_months.empty :
        first_possible_landing_month_check = proj_start_month_period + avg_lag_months_approx
        if current_goal_lpi_month_period < first_possible_landing_month_check:
             feasibility_details_early = f"Goal LPI ({current_goal_lpi_month_period.strftime('%Y-%m')}) is too soon. Minimum landing month: {first_possible_landing_month_check.strftime('%Y-%m')} (Lag: {effective_lag_for_planning_approx:.1f} days)."
             return default_return_ai[0], default_return_ai[1], default_return_ai[2], feasibility_details_early, True, 0
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Projection calculation window is invalid.", True, 0

    ai_gen_df = pd.DataFrame(index=projection_calc_months)
    ai_gen_df['Required_QLs_POF_Initial'] = 0.0; ai_gen_df['Required_QLs_POF_Final'] = 0.0
    ai_gen_df['Unallocatable_QLs'] = 0.0; ai_gen_df['Generated_ICF_Mean'] = 0.0
    ai_gen_df['Adjusted_CPQL_For_Month'] = estimated_cpql_user; ai_gen_df['Implied_Ad_Spend'] = 0.0
    icfs_still_to_assign_globally = float(current_goal_icf_number)
    latest_permissible_generation_month = current_goal_lpi_month_period - avg_lag_months_approx
    earliest_permissible_generation_month = proj_start_month_period
    valid_generation_months_for_planning = pd.period_range(
        start=max(earliest_permissible_generation_month, projection_calc_months.min()),
        end=min(latest_permissible_generation_month, projection_calc_months.max())
    )

    baseline_monthly_ql_volume = 1.0
    if ts_pof_col_for_prop and ts_pof_col_for_prop in processed_df.columns and not processed_df.empty and 'Submission_Month' in processed_df.columns:
        valid_pof_df_baseline = processed_df[processed_df[ts_pof_col_for_prop].notna()]
        if not valid_pof_df_baseline.empty and 'Submission_Month' in valid_pof_df_baseline:
            num_unique_hist_months = valid_pof_df_baseline['Submission_Month'].nunique()
            if num_unique_hist_months > 0:
                months_for_baseline_calc = min(num_unique_hist_months, 6)
                recent_hist_months = valid_pof_df_baseline['Submission_Month'].drop_duplicates().nlargest(months_for_baseline_calc)
                baseline_data_for_avg = valid_pof_df_baseline[valid_pof_df_baseline['Submission_Month'].isin(recent_hist_months)]
                if not baseline_data_for_avg.empty:
                    total_pof_baseline_period = baseline_data_for_avg.shape[0]
                    calculated_baseline = total_pof_baseline_period / months_for_baseline_calc
                    if calculated_baseline > 0: baseline_monthly_ql_volume = calculated_baseline
    monthly_ql_capacity_target_heuristic = baseline_monthly_ql_volume * ai_monthly_ql_capacity_multiplier
    if monthly_ql_capacity_target_heuristic < 1: monthly_ql_capacity_target_heuristic = 50

    site_level_monthly_qlof = {}
    all_defined_sites = site_metrics_df['Site'].unique() if not site_metrics_df.empty and 'Site' in site_metrics_df else np.array([])

    hist_site_pof_prop_overall = pd.Series(dtype=float)
    if ts_pof_col_for_prop and ts_pof_col_for_prop in processed_df.columns and 'Site' in processed_df.columns:
        valid_pof_data_for_prop = processed_df[processed_df[ts_pof_col_for_prop].notna()]
        if not valid_pof_data_for_prop.empty and 'Site' in valid_pof_data_for_prop:
             hist_site_pof_prop_overall = valid_pof_data_for_prop['Site'].value_counts(normalize=True)

    site_redistribution_scores_overall = {}
    if not site_metrics_df.empty and 'Site' in site_metrics_df.columns and 'Qual -> ICF %' in site_metrics_df.columns:
        for _, site_row_score in site_metrics_df.iterrows():
            score_val = site_row_score.get('Qual -> ICF %', 0.0)
            score_val = 0.0 if pd.isna(score_val) or score_val < 0 else score_val
            site_redistribution_scores_overall[site_row_score['Site']] = score_val if score_val > 1e-6 else 1e-6

    if not valid_generation_months_for_planning.empty:
        for gen_month in valid_generation_months_for_planning:
            if icfs_still_to_assign_globally <= 1e-9: break

            active_sites_this_gen_month = []
            if all_defined_sites.size > 0:
                for site_name_iter in all_defined_sites:
                    is_active_for_month = True
                    if site_activity_schedule and site_name_iter in site_activity_schedule:
                        activity_info = site_activity_schedule[site_name_iter]
                        activation_pd = activity_info.get('activation_period')
                        deactivation_pd = activity_info.get('deactivation_period')
                        if activation_pd and gen_month < activation_pd: is_active_for_month = False
                        if deactivation_pd and gen_month > deactivation_pd: is_active_for_month = False
                    if is_active_for_month: active_sites_this_gen_month.append(site_name_iter)
            
            hist_site_pof_prop_active = hist_site_pof_prop_overall[hist_site_pof_prop_overall.index.isin(active_sites_this_gen_month)]
            if not hist_site_pof_prop_active.empty and hist_site_pof_prop_active.sum() > 1e-9 :
                 hist_site_pof_prop_active = hist_site_pof_prop_active / hist_site_pof_prop_active.sum()
            else: hist_site_pof_prop_active = pd.Series(dtype=float)
            site_redist_scores_active = {s: score for s, score in site_redistribution_scores_overall.items() if s in active_sites_this_gen_month}

            qls_theoretically_needed_for_remaining_float = (icfs_still_to_assign_globally / overall_pof_to_icf_rate) if overall_pof_to_icf_rate > 1e-9 else float('inf')
            ql_target_potential_this_month_heuristic_limited = min(qls_theoretically_needed_for_remaining_float, monthly_ql_capacity_target_heuristic)
            if icfs_still_to_assign_globally > 1e-9 and \
               qls_theoretically_needed_for_remaining_float < monthly_ql_capacity_target_heuristic and \
               qls_theoretically_needed_for_remaining_float > 0:
                current_month_initial_ql_target = math.ceil(qls_theoretically_needed_for_remaining_float)
            elif ql_target_potential_this_month_heuristic_limited > 0 and ql_target_potential_this_month_heuristic_limited < 1.0:
                current_month_initial_ql_target = math.ceil(ql_target_potential_this_month_heuristic_limited)
            else: current_month_initial_ql_target = round(max(0, ql_target_potential_this_month_heuristic_limited))

            ai_gen_df.loc[gen_month, 'Required_QLs_POF_Initial'] = current_month_initial_ql_target
            site_ql_allocations_month_specific = {site: 0 for site in active_sites_this_gen_month}
            unallocatable_this_month = 0

            if current_month_initial_ql_target > 0 and active_sites_this_gen_month:
                temp_site_allocations_float = {}
                if not hist_site_pof_prop_active.empty:
                    for site_n_dist, prop_dist in hist_site_pof_prop_active.items():
                        temp_site_allocations_float[site_n_dist] = current_month_initial_ql_target * prop_dist
                else:
                    ql_per_site_fallback_float = current_month_initial_ql_target / len(active_sites_this_gen_month)
                    for site_n_dist in active_sites_this_gen_month: temp_site_allocations_float[site_n_dist] = ql_per_site_fallback_float
                for site_n_round, ql_float_round in temp_site_allocations_float.items():
                    site_ql_allocations_month_specific[site_n_round] = round(ql_float_round)
                current_sum_ql_after_initial_round = sum(site_ql_allocations_month_specific.values())
                diff_ql_rounding_adj = current_month_initial_ql_target - current_sum_ql_after_initial_round
                if diff_ql_rounding_adj != 0 and active_sites_this_gen_month:
                    target_site_for_diff_adj = active_sites_this_gen_month[0]
                    if temp_site_allocations_float:
                         target_site_for_diff_adj = max(temp_site_allocations_float, key=temp_site_allocations_float.get, default=active_sites_this_gen_month[0])
                    elif site_redist_scores_active:
                         best_site_cand_adj = max(site_redist_scores_active, key=site_redist_scores_active.get, default=None)
                         if best_site_cand_adj: target_site_for_diff_adj = best_site_cand_adj
                    site_ql_allocations_month_specific[target_site_for_diff_adj] += diff_ql_rounding_adj
                max_iterations_site_cap_loop = 10
                for iteration_cap_loop in range(max_iterations_site_cap_loop):
                    excess_ql_pool_iter_val = 0; newly_capped_this_iter_val = False
                    for site_n_iter_cap, allocated_qls_iter_cap in list(site_ql_allocations_month_specific.items()):
                        if site_n_iter_cap not in active_sites_this_gen_month: continue
                        site_cap_val_iter = site_caps_input.get(site_n_iter_cap, float('inf'))
                        if allocated_qls_iter_cap > site_cap_val_iter:
                            diff_iter_cap = allocated_qls_iter_cap - site_cap_val_iter
                            excess_ql_pool_iter_val += diff_iter_cap
                            site_ql_allocations_month_specific[site_n_iter_cap] = site_cap_val_iter
                            newly_capped_this_iter_val = True
                    if excess_ql_pool_iter_val < 1: break
                    candidate_sites_for_rd_list = {s: score for s, score in site_redist_scores_active.items() if s in site_ql_allocations_month_specific and site_ql_allocations_month_specific[s] < site_caps_input.get(s, float('inf'))}
                    if not candidate_sites_for_rd_list: unallocatable_this_month += round(excess_ql_pool_iter_val); break
                    total_score_candidates_rd = sum(candidate_sites_for_rd_list.values())
                    if total_score_candidates_rd <= 1e-9: unallocatable_this_month += round(excess_ql_pool_iter_val); break
                    temp_excess_after_rd_iter = excess_ql_pool_iter_val
                    sorted_candidates_for_rd_list = sorted(candidate_sites_for_rd_list.items(), key=lambda item_rd: item_rd[1], reverse=True)
                    for site_rd_val, score_rd_val in sorted_candidates_for_rd_list:
                        if temp_excess_after_rd_iter < 1: break
                        share_of_excess_raw_val = (score_rd_val / total_score_candidates_rd) * excess_ql_pool_iter_val
                        capacity_to_take_val = site_caps_input.get(site_rd_val, float('inf')) - site_ql_allocations_month_specific[site_rd_val]
                        actual_add_to_site_val = min(share_of_excess_raw_val, capacity_to_take_val, temp_excess_after_rd_iter)
                        actual_add_rounded_val = round(actual_add_to_site_val)
                        site_ql_allocations_month_specific[site_rd_val] += actual_add_rounded_val
                        temp_excess_after_rd_iter -= actual_add_rounded_val
                    excess_ql_pool_iter_val = max(0, temp_excess_after_rd_iter)
                    if excess_ql_pool_iter_val < 1 or not newly_capped_this_iter_val:
                        if excess_ql_pool_iter_val >= 1: unallocatable_this_month += round(excess_ql_pool_iter_val)
                        break
                    if iteration_cap_loop == max_iterations_site_cap_loop - 1 and excess_ql_pool_iter_val >=1 :
                        unallocatable_this_month += round(excess_ql_pool_iter_val)
            elif current_month_initial_ql_target > 0 and not active_sites_this_gen_month:
                 unallocatable_this_month = current_month_initial_ql_target

            sum_actually_allocated_qls_final = sum(site_ql_allocations_month_specific.values())
            ai_gen_df.loc[gen_month, 'Required_QLs_POF_Final'] = sum_actually_allocated_qls_final
            ai_gen_df.loc[gen_month, 'Unallocatable_QLs'] = unallocatable_this_month
            site_level_monthly_qlof[gen_month] = site_ql_allocations_month_specific.copy()
            icfs_generated_this_month_float = sum_actually_allocated_qls_final * overall_pof_to_icf_rate
            ai_gen_df.loc[gen_month, 'Generated_ICF_Mean'] = icfs_generated_this_month_float
            icfs_still_to_assign_globally -= icfs_generated_this_month_float
    else:
        if run_mode == "primary" and 'st' in globals() and hasattr(st, 'sidebar'):
            st.sidebar.error("Primary run: No valid generation months for planning.")
        if icfs_still_to_assign_globally > 1e-9:
            first_possible_landing_month_check_no_gen = proj_start_month_period + avg_lag_months_approx
            feasibility_details_no_gen = f"Goal LPI ({current_goal_lpi_month_period.strftime('%Y-%m')}) combined with lag ({effective_lag_for_planning_approx:.1f} days) results in no valid QL generation months. Min. landing: {first_possible_landing_month_check_no_gen.strftime('%Y-%m')}."
            ai_results_df_empty = pd.DataFrame(index=projection_calc_months); ai_results_df_empty['Projected_ICF_Landed'] = 0; ai_results_df_empty['Cumulative_ICF_Landed'] = 0; ai_results_df_empty['Target_QLs_POF'] = 0; ai_results_df_empty['Implied_Ad_Spend'] = 0
            return ai_results_df_empty, pd.DataFrame(), "N/A", feasibility_details_no_gen, True, 0

    total_generated_icfs_float = ai_gen_df['Generated_ICF_Mean'].sum()
    generation_goal_met_or_exceeded_in_float = (total_generated_icfs_float >= (current_goal_icf_number - 0.01))
    total_unallocated_qls_run = 0
    for gen_month_spend_idx_val in ai_gen_df.index:
        final_qls_for_cpql_calc_val = round(ai_gen_df.loc[gen_month_spend_idx_val, 'Required_QLs_POF_Final'])
        current_cpql_for_month_val = estimated_cpql_user
        if ql_vol_increase_threshold_pct > 0 and cpql_inflation_factor_pct > 0 and baseline_monthly_ql_volume > 0:
            if final_qls_for_cpql_calc_val > baseline_monthly_ql_volume:
                ql_increase_pct_val_calc = (final_qls_for_cpql_calc_val - baseline_monthly_ql_volume) / baseline_monthly_ql_volume
                threshold_units_crossed_val_calc = ql_increase_pct_val_calc / (ql_vol_increase_threshold_pct / 100.0)
                if threshold_units_crossed_val_calc > 0:
                    inflation_multiplier_val_calc = 1 + (threshold_units_crossed_val_calc * (cpql_inflation_factor_pct / 100.0))
                    current_cpql_for_month_val = estimated_cpql_user * inflation_multiplier_val_calc
        ai_gen_df.loc[gen_month_spend_idx_val, 'Adjusted_CPQL_For_Month'] = current_cpql_for_month_val
        ai_gen_df.loc[gen_month_spend_idx_val, 'Implied_Ad_Spend'] = final_qls_for_cpql_calc_val * current_cpql_for_month_val
        total_unallocated_qls_run += ai_gen_df.loc[gen_month_spend_idx_val, 'Unallocatable_QLs']

    variation_f_val = icf_variation_percent / 100.0
    ai_gen_df['Generated_ICF_Low'] = (ai_gen_df['Generated_ICF_Mean'] * (1 - variation_f_val)).round(0).astype(int).clip(lower=0)
    ai_gen_df['Generated_ICF_High'] = (ai_gen_df['Generated_ICF_Mean'] * (1 + variation_f_val)).round(0).astype(int).clip(lower=0)
    ai_gen_df['Projected_CPICF_Mean'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_Mean'].replace(0, np.nan)).round(2)
    ai_gen_df['Projected_CPICF_Low'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_High'].replace(0, np.nan)).round(2)
    ai_gen_df['Projected_CPICF_High'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_Low'].replace(0, np.nan)).round(2)

    ai_results_df = pd.DataFrame(index=projection_calc_months); ai_results_df['Projected_ICF_Landed'] = 0.0
    for cohort_g_month_land_idx_val in ai_gen_df.index:
        icfs_gen_this_cohort_land_val = ai_gen_df.loc[cohort_g_month_land_idx_val, 'Generated_ICF_Mean']
        if icfs_gen_this_cohort_land_val <= 0: continue
        if ai_lag_method == "percentiles":
            lags_to_use = [ai_lag_p25_days, ai_lag_p50_days, ai_lag_p75_days]
            proportions = [0.25, 0.50, 0.25]
            for i, lag_days_for_share in enumerate(lags_to_use):
                icf_share = icfs_gen_this_cohort_land_val * proportions[i]
                if icf_share <= 0: continue
                full_lag_mths_share = int(np.floor(lag_days_for_share / days_in_avg_m))
                rem_lag_days_share = lag_days_for_share - (full_lag_mths_share * days_in_avg_m)
                frac_next_share = rem_lag_days_share / days_in_avg_m; frac_curr_share = 1.0 - frac_next_share
                land_m1_share = cohort_g_month_land_idx_val + full_lag_mths_share
                land_m2_share = land_m1_share + 1
                if land_m1_share in ai_results_df.index: ai_results_df.loc[land_m1_share, 'Projected_ICF_Landed'] += icf_share * frac_curr_share
                if land_m2_share in ai_results_df.index: ai_results_df.loc[land_m2_share, 'Projected_ICF_Landed'] += icf_share * frac_next_share
        else:
            full_lag_mths_land_val = int(np.floor(avg_overall_lag_days / days_in_avg_m))
            rem_lag_days_land_val = avg_overall_lag_days - (full_lag_mths_land_val * days_in_avg_m)
            frac_next_land_val = rem_lag_days_land_val / days_in_avg_m; frac_curr_land_val = 1.0 - frac_next_land_val
            land_m1_val = cohort_g_month_land_idx_val + full_lag_mths_land_val
            land_m2_val = land_m1_val + 1
            if land_m1_val in ai_results_df.index: ai_results_df.loc[land_m1_val, 'Projected_ICF_Landed'] += icfs_gen_this_cohort_land_val * frac_curr_land_val
            if land_m2_val in ai_results_df.index: ai_results_df.loc[land_m2_val, 'Projected_ICF_Landed'] += icfs_gen_this_cohort_land_val * frac_next_land_val
    ai_results_df['Projected_ICF_Landed'] = ai_results_df['Projected_ICF_Landed'].round(0).astype(int)
    ai_results_df['Cumulative_ICF_Landed'] = ai_results_df['Projected_ICF_Landed'].cumsum()
    ai_results_df['Target_QLs_POF'] = ai_gen_df['Required_QLs_POF_Initial'].reindex(ai_results_df.index).fillna(0).round(0).astype(int)
    ai_results_df['Implied_Ad_Spend'] = ai_gen_df['Implied_Ad_Spend'].reindex(ai_results_df.index).fillna(0)

    cpicf_m_res = pd.Series(index=ai_results_df.index, dtype=float); cpicf_l_res = pd.Series(index=ai_results_df.index, dtype=float); cpicf_h_res = pd.Series(index=ai_results_df.index, dtype=float)
    for g_m_cpicf_idx_val in ai_gen_df.index:
        if ai_gen_df.loc[g_m_cpicf_idx_val, 'Generated_ICF_Mean'] > 0:
            display_land_m_cpicf_val = g_m_cpicf_idx_val + avg_lag_months_approx
            if display_land_m_cpicf_val in cpicf_m_res.index and pd.isna(cpicf_m_res.loc[display_land_m_cpicf_val]):
                cpicf_m_res.loc[display_land_m_cpicf_val] = ai_gen_df.loc[g_m_cpicf_idx_val, 'Projected_CPICF_Mean']
                cpicf_l_res.loc[display_land_m_cpicf_val] = ai_gen_df.loc[g_m_cpicf_idx_val, 'Projected_CPICF_Low']
                cpicf_h_res.loc[display_land_m_cpicf_val] = ai_gen_df.loc[g_m_cpicf_idx_val, 'Projected_CPICF_High']
    ai_results_df['Projected_CPICF_Cohort_Source_Mean'] = cpicf_m_res
    ai_results_df['Projected_CPICF_Cohort_Source_Low'] = cpicf_l_res
    ai_results_df['Projected_CPICF_Cohort_Source_High'] = cpicf_h_res

    ai_gen_df['Cumulative_Generated_ICF_Final'] = ai_gen_df['Generated_ICF_Mean'].cumsum()
    ads_off_s_val = ai_gen_df[ai_gen_df['Cumulative_Generated_ICF_Final'] >= (current_goal_icf_number - 1e-9)]
    ads_off_date_str_calc_val = "Goal Not Met by End of Projection"
    if not ads_off_s_val.empty:
        ads_off_month_period_val_calc = ads_off_s_val.index[0]
        icfs_generated_by_ads_off_month_cohort_val = ai_gen_df.loc[ads_off_month_period_val_calc, 'Generated_ICF_Mean']
        cumulative_generated_before_ads_off_month_cohort_val = ai_gen_df.loc[ads_off_month_period_val_calc, 'Cumulative_Generated_ICF_Final'] - icfs_generated_by_ads_off_month_cohort_val
        icfs_needed_from_ads_off_month_cohort_val = current_goal_icf_number - cumulative_generated_before_ads_off_month_cohort_val
        if icfs_needed_from_ads_off_month_cohort_val <= 0:
            prev_month_period_ads_off_val = ads_off_month_period_val_calc - 1
            if prev_month_period_ads_off_val in ai_gen_df.index and \
               cumulative_generated_before_ads_off_month_cohort_val >= (current_goal_icf_number - 1e-9) :
                 ads_off_date_str_calc_val = prev_month_period_ads_off_val.end_time.strftime('%Y-%m-%d')
            else: ads_off_date_str_calc_val = ads_off_month_period_val_calc.start_time.strftime('%Y-%m-%d')
        elif icfs_generated_by_ads_off_month_cohort_val > 1e-9:
            fraction_of_ads_off_month_needed_val = max(0, min(1, icfs_needed_from_ads_off_month_cohort_val / icfs_generated_by_ads_off_month_cohort_val))
            ads_off_day_offset_val = int(np.ceil(fraction_of_ads_off_month_needed_val * days_in_avg_m)); ads_off_day_offset_val = max(1, ads_off_day_offset_val)
            ads_off_date_str_calc_val = (ads_off_month_period_val_calc.start_time + pd.Timedelta(days=ads_off_day_offset_val - 1)).strftime('%Y-%m-%d')
        else: ads_off_date_str_calc_val = ads_off_month_period_val_calc.start_time.strftime('%Y-%m-%d')

    ai_site_proj_df = pd.DataFrame()
    if all_defined_sites.size > 0:
        site_data_coll_ai_final_val = {site: {} for site in all_defined_sites}
        for site_n_final_init_val in all_defined_sites:
            for month_p_final_init_val in projection_calc_months:
                month_str_final_init_val = month_p_final_init_val.strftime('%Y-%m')
                site_data_coll_ai_final_val[site_n_final_init_val][(month_str_final_init_val, 'Projected QLs (POF)')] = 0
                site_data_coll_ai_final_val[site_n_final_init_val][(month_str_final_init_val, 'Projected ICFs Landed')] = 0.0
        for gen_month_site_idx_val in ai_gen_df.index:
            qlof_for_month_val = site_level_monthly_qlof.get(gen_month_site_idx_val, {})
            active_sites_for_this_gen_month_for_icf_calc = [] 
            for site_name_check_icf in all_defined_sites:
                is_active_for_month_icf = True
                if site_activity_schedule and site_name_check_icf in site_activity_schedule:
                    activity_info_icf = site_activity_schedule[site_name_check_icf]
                    act_pd_icf = activity_info_icf.get('activation_period')
                    deact_pd_icf = activity_info_icf.get('deactivation_period')
                    if act_pd_icf and gen_month_site_idx_val < act_pd_icf: is_active_for_month_icf = False
                    if deact_pd_icf and gen_month_site_idx_val > deact_pd_icf: is_active_for_month_icf = False
                if is_active_for_month_icf: active_sites_for_this_gen_month_for_icf_calc.append(site_name_check_icf)

            for site_n_final_val in active_sites_for_this_gen_month_for_icf_calc:
                qls_for_site_this_gen_month_val = round(qlof_for_month_val.get(site_n_final_val, 0))
                site_data_coll_ai_final_val[site_n_final_val][(gen_month_site_idx_val.strftime('%Y-%m'), 'Projected QLs (POF)')] = qls_for_site_this_gen_month_val
                site_perf_r_final_val = site_metrics_df[site_metrics_df['Site'] == site_n_final_val] if not site_metrics_df.empty else pd.DataFrame()
                site_pof_icf_rate_final_val = overall_pof_to_icf_rate
                if not site_perf_r_final_val.empty and 'Qual -> ICF %' in site_perf_r_final_val.columns:
                    rate_val_site = site_perf_r_final_val['Qual -> ICF %'].iloc[0]
                    if pd.notna(rate_val_site) and rate_val_site >= 0: site_pof_icf_rate_final_val = rate_val_site
                site_gen_icfs_this_gen_month_float = qls_for_site_this_gen_month_val * site_pof_icf_rate_final_val
                if site_gen_icfs_this_gen_month_float > 0:
                    lag_days_to_use_for_site_smear = [avg_overall_lag_days]
                    proportions_for_site_smear = [1.0]
                    if ai_lag_method == "percentiles":
                        lag_days_to_use_for_site_smear = [ai_lag_p25_days, ai_lag_p50_days, ai_lag_p75_days]
                        proportions_for_site_smear = [0.25, 0.50, 0.25]
                    for i_site_smear, lag_d_site_smear in enumerate(lag_days_to_use_for_site_smear):
                        icf_s_share_smear = site_gen_icfs_this_gen_month_float * proportions_for_site_smear[i_site_smear]
                        if icf_s_share_smear <=0: continue
                        s_f_lag_m = int(np.floor(lag_d_site_smear / days_in_avg_m))
                        s_r_lag_d = lag_d_site_smear - (s_f_lag_m * days_in_avg_m)
                        s_f_next = s_r_lag_d / days_in_avg_m; s_f_curr = 1.0 - s_f_next
                        s_l_m1 = gen_month_site_idx_val + s_f_lag_m
                        s_l_m2 = s_l_m1 + 1
                        if s_l_m1 in projection_calc_months:
                            k_l_m1_s = (s_l_m1.strftime('%Y-%m'), 'Projected ICFs Landed')
                            site_data_coll_ai_final_val[site_n_final_val][k_l_m1_s] = site_data_coll_ai_final_val[site_n_final_val].get(k_l_m1_s, 0.0) + (icf_s_share_smear * s_f_curr)
                        if s_l_m2 in projection_calc_months:
                            k_l_m2_s = (s_l_m2.strftime('%Y-%m'), 'Projected ICFs Landed')
                            site_data_coll_ai_final_val[site_n_final_val][k_l_m2_s] = site_data_coll_ai_final_val[site_n_final_val].get(k_l_m2_s, 0.0) + (icf_s_share_smear * s_f_next)
        if site_data_coll_ai_final_val:
            ai_site_proj_df = pd.DataFrame.from_dict(site_data_coll_ai_final_val, orient='index')
            if not ai_site_proj_df.empty:
                ai_site_proj_df.columns = pd.MultiIndex.from_tuples(ai_site_proj_df.columns, names=['Month', 'Metric'])
                ai_site_proj_df = ai_site_proj_df.sort_index(axis=1, level=[0,1])
                for m_fmt_site_final_val in projection_calc_months.strftime('%Y-%m'):
                    landed_col_tuple_site = (m_fmt_site_final_val, 'Projected ICFs Landed')
                    if landed_col_tuple_site in ai_site_proj_df.columns:
                         ai_site_proj_df[landed_col_tuple_site] = ai_site_proj_df[landed_col_tuple_site].round(0).astype(int)
                ai_site_proj_df = ai_site_proj_df.fillna(0)
                if not ai_site_proj_df.empty:
                    numeric_cols_site_ai_final_val = [c for c in ai_site_proj_df.columns if pd.api.types.is_numeric_dtype(ai_site_proj_df[c])]
                    if numeric_cols_site_ai_final_val:
                        total_r_ai_final_val = ai_site_proj_df[numeric_cols_site_ai_final_val].sum(axis=0)
                        total_df_ai_final_val = pd.DataFrame(total_r_ai_final_val).T; total_df_ai_final_val.index = ["Grand Total"]
                        ai_site_proj_df = pd.concat([ai_site_proj_df, total_df_ai_final_val])

    final_achieved_icfs_landed_run = ai_results_df['Cumulative_ICF_Landed'].max() if 'Cumulative_ICF_Landed' in ai_results_df and not ai_results_df.empty else 0
    goal_met_on_time_this_run = False
    actual_lpi_month_achieved_this_run = current_goal_lpi_month_period
    if not ai_results_df.empty and 'Cumulative_ICF_Landed' in ai_results_df:
        met_goal_series_val = ai_results_df[ai_results_df['Cumulative_ICF_Landed'] >= (current_goal_icf_number - 1e-9)]
        if not met_goal_series_val.empty:
            actual_lpi_month_achieved_this_run = met_goal_series_val.index.min()
            if actual_lpi_month_achieved_this_run <= current_goal_lpi_month_period :
                 goal_met_on_time_this_run = True

    significant_icfs_still_to_assign_from_gen_planning = icfs_still_to_assign_globally > 0.1
    is_unfeasible_this_run = not goal_met_on_time_this_run or \
                             total_unallocated_qls_run > 0 or \
                             significant_icfs_still_to_assign_from_gen_planning
    feasibility_prefix = "AI Projection: "
    detailed_outcome_message = ""
    effective_lpi_for_run_msg = current_goal_lpi_month_period
    effective_icf_goal_for_run_msg = current_goal_icf_number
    if run_mode == "primary":
        goal_desc_for_msg = f"Original goals ({goal_icf_number_orig} ICFs by {goal_lpi_date_dt_orig.strftime('%Y-%m-%d')})"
    else:
        goal_desc_for_msg = f"Best Case Scenario (LPI extended to {current_goal_lpi_month_period.strftime('%Y-%m')} for original {goal_icf_number_orig} ICFs goal)"
    landed_percentage_of_effective_goal = 0
    if effective_icf_goal_for_run_msg > 0: landed_percentage_of_effective_goal = final_achieved_icfs_landed_run / effective_icf_goal_for_run_msg
    near_miss_threshold_pct = 0.95
    if not is_unfeasible_this_run:
        detailed_outcome_message = f"{goal_desc_for_msg} appear ACHIEVABLE."
        if final_achieved_icfs_landed_run > effective_icf_goal_for_run_msg: detailed_outcome_message += f" (Projected to exceed goal, landing {final_achieved_icfs_landed_run:.0f} ICFs)."
        elif run_mode == "best_case_extended_lpi": detailed_outcome_message = f"{goal_desc_for_msg}: Target of {effective_icf_goal_for_run_msg} ICFs ACHIEVED by {actual_lpi_month_achieved_this_run.strftime('%Y-%m')}."
    else:
        base_unfeasible_msg_for_else = f"{goal_desc_for_msg} "
        constraint_msgs_list = []
        hard_constraints_hit_flag = False
        if total_unallocated_qls_run > 0: constraint_msgs_list.append(f"{total_unallocated_qls_run:.0f} QLs unallocatable (site caps/activity)."); hard_constraints_hit_flag = True
        if significant_icfs_still_to_assign_from_gen_planning: constraint_msgs_list.append(f"{icfs_still_to_assign_globally:.1f} ICFs (target: {effective_icf_goal_for_run_msg}) could not be fully planned in generation phase."); hard_constraints_hit_flag = True
        if hard_constraints_hit_flag: detailed_outcome_message = base_unfeasible_msg_for_else + "appear UNFEASIBLE due to planning constraints: " + " ".join(constraint_msgs_list)
        elif not goal_met_on_time_this_run:
            achieved_by_date_str_msg = actual_lpi_month_achieved_this_run.strftime('%Y-%m') if final_achieved_icfs_landed_run > 0 and pd.notna(actual_lpi_month_achieved_this_run) else 'end of projection'
            target_lpi_str_msg = effective_lpi_for_run_msg.strftime('%Y-%m')
            if landed_percentage_of_effective_goal >= near_miss_threshold_pct and actual_lpi_month_achieved_this_run <= effective_lpi_for_run_msg:
                detailed_outcome_message = base_unfeasible_msg_for_else + f"LANDING GOAL NEAR MISS. Projected {final_achieved_icfs_landed_run:.0f} ICFs ({landed_percentage_of_effective_goal*100:.1f}%) by {achieved_by_date_str_msg} (Target LPI: {target_lpi_str_msg})."
                if generation_goal_met_or_exceeded_in_float and final_achieved_icfs_landed_run < effective_icf_goal_for_run_msg: detailed_outcome_message += " (Note: Goal met in total ICFs generated; landed sum slightly short due to monthly rounding)."
            else:
                if actual_lpi_month_achieved_this_run > effective_lpi_for_run_msg and final_achieved_icfs_landed_run >= effective_icf_goal_for_run_msg : detailed_outcome_message = base_unfeasible_msg_for_else + f"LPI NOT MET. Goal of {effective_icf_goal_for_run_msg} ICFs achieved by {achieved_by_date_str_msg}, which is after target LPI of {target_lpi_str_msg}."
                else: detailed_outcome_message = base_unfeasible_msg_for_else + f"LANDING GOAL SHORTFALL/LPI MISSED. Projected {final_achieved_icfs_landed_run:.0f} ICFs ({landed_percentage_of_effective_goal*100:.1f}%) by {achieved_by_date_str_msg} (Target: {effective_icf_goal_for_run_msg} ICFs by {target_lpi_str_msg})."
        else:
            detailed_outcome_message = base_unfeasible_msg_for_else + "appear UNFEASIBLE for other reasons."
            if total_unallocated_qls_run > 0 and not hard_constraints_hit_flag: detailed_outcome_message += f" Minor QL unallocation: {total_unallocated_qls_run:.0f}."
            if icfs_still_to_assign_globally > 1e-5 and not significant_icfs_still_to_assign_from_gen_planning and not hard_constraints_hit_flag: detailed_outcome_message += f" Small remainder of {icfs_still_to_assign_globally:.2f} ICFs from generation."
    feasibility_msg_final_display = feasibility_prefix + detailed_outcome_message.strip()

    display_end_month_final_val = projection_calc_months[-1] if not projection_calc_months.empty else proj_start_month_period
    if not ai_results_df.empty and 'Cumulative_ICF_Landed' in ai_results_df:
        met_goal_series_trim_val = ai_results_df[ai_results_df['Cumulative_ICF_Landed'] >= (current_goal_icf_number - 1e-9)]
        if not met_goal_series_trim_val.empty:
            lpi_achieved_month_for_trim_val_calc = met_goal_series_trim_val.index.min()
            try:
                candidate_end_month_val_ts_calc = lpi_achieved_month_for_trim_val_calc.to_timestamp() + pd.offsets.MonthEnd(3)
                candidate_end_month_val_calc = candidate_end_month_val_ts_calc.to_period('M')
                display_end_month_final_val = min(projection_calc_months[-1], max(candidate_end_month_val_calc, current_goal_lpi_month_period + 3))
            except Exception: pass
    ai_results_df_final_display_val = pd.DataFrame()
    if not ai_results_df.empty and proj_start_month_period <= display_end_month_final_val:
        try: ai_results_df_final_display_val = ai_results_df.loc[proj_start_month_period:display_end_month_final_val].copy()
        except Exception: ai_results_df_final_display_val = ai_results_df.copy()
    if 'st' in globals() and hasattr(st, 'session_state'):
        if run_mode == "primary": st.session_state.ai_gen_df_debug_primary = ai_gen_df.copy(); st.session_state.ai_results_df_debug_primary = ai_results_df.copy()
        elif run_mode == "best_case_extended_lpi": st.session_state.ai_gen_df_debug_best_case = ai_gen_df.copy(); st.session_state.ai_results_df_debug_best_case = ai_results_df.copy()
    return ai_results_df_final_display_val, ai_site_proj_df, ads_off_date_str_calc_val, feasibility_msg_final_display, is_unfeasible_this_run, final_achieved_icfs_landed_run
# --- END OF AI FORECAST CORE FUNCTION ---


# --- FINAL, VERIFIED CORE FUNCTION FOR FUNNEL ANALYSIS ---
@st.cache_data
def calculate_pipeline_projection(
    _processed_df, ordered_stages, ts_col_map, inter_stage_lags,
    conversion_rates, lag_assumption_model
):
    """
    Calculates the future ICFs and Enrollments expected from the current in-flight pipeline.
    This version makes lag calculations robust to handle missing (NaN) data.
    """
    default_return = pd.DataFrame(columns=['Projected_ICF_Landed', 'Cumulative_ICF_Landed', 'Projected_Enrollments_Landed', 'Cumulative_Enrollments_Landed'])
    if _processed_df is None or _processed_df.empty:
        return default_return

    # --- 1. Define TRUE Terminal Stages ---
    true_terminal_stages = [s for s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST] if s in ts_col_map]
    true_terminal_ts_cols = [ts_col_map.get(s) for s in true_terminal_stages]

    # --- 2. Filter for In-Flight Leads ---
    in_flight_df = _processed_df.copy()
    for ts_col in true_terminal_ts_cols:
        if ts_col in in_flight_df.columns:
            in_flight_df = in_flight_df[in_flight_df[ts_col].isna()]
    if in_flight_df.empty:
        st.info("Funnel Analysis: No leads are currently in-flight.")
        return default_return

    # --- 3. Determine Current Stage ---
    def get_current_stage(row, ordered_stages, ts_col_map):
        last_stage, last_ts = None, pd.NaT
        for stage in ordered_stages:
            if stage in true_terminal_stages: continue
            ts_col = ts_col_map.get(stage)
            if ts_col and ts_col in row and pd.notna(row[ts_col]):
                if pd.isna(last_ts) or row[ts_col] > last_ts:
                    last_ts, last_stage = row[ts_col], stage
        return last_stage, last_ts
    in_flight_df[['current_stage', 'current_stage_ts']] = in_flight_df.apply(
        lambda row: get_current_stage(row, ordered_stages, ts_col_map), axis=1, result_type='expand'
    )
    in_flight_df.dropna(subset=['current_stage'], inplace=True)
    
    # --- 4. Create the Master Pool of ALL In-Flight ICFs ---
    all_icfs_to_project = []
    icf_target_stage = STAGE_SIGNED_ICF
    ts_icf_col = ts_col_map.get(icf_target_stage)

    # Source 1: Already signed ICFs (lag to become an ICF is 0)
    already_icf_in_flight = in_flight_df[in_flight_df[ts_icf_col].notna()].copy()
    for _, row in already_icf_in_flight.iterrows():
        all_icfs_to_project.append({'prob': 1.0, 'lag_to_icf': 0.0, 'start_date': row[ts_icf_col]})

    # Source 2: Leads before ICF
    leads_before_icf = in_flight_df[in_flight_df[ts_icf_col].isna()].copy()
    for _, row in leads_before_icf.iterrows():
        prob_to_icf, path_found = 1.0, False
        lags_to_icf_list = []
        
        start_index = ordered_stages.index(row['current_stage'])
        for i in range(start_index, len(ordered_stages) - 1):
            from_stage, to_stage = ordered_stages[i], ordered_stages[i+1]
            rate_key = f"{from_stage} -> {to_stage}"
            prob_to_icf *= conversion_rates.get(rate_key, 0.0)
            # Add lag to a list for safe summing
            lags_to_icf_list.append(inter_stage_lags.get(rate_key, 0.0))
            if to_stage == icf_target_stage:
                path_found = True
                break
        
        if path_found and prob_to_icf > 0:
            # BUG FIX: Safely sum the lags, treating any missing values (NaN) as 0.
            total_lag_to_icf = np.nansum(lags_to_icf_list)
            all_icfs_to_project.append({'prob': prob_to_icf, 'lag_to_icf': total_lag_to_icf, 'start_date': row['current_stage_ts']})

    # --- 5. Project Enrollments from the Master Pool ---
    projected_enrollments = []
    icf_to_enroll_rate = conversion_rates.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0.0)
    icf_to_enroll_lag = inter_stage_lags.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0.0)
    # BUG FIX: Ensure the final lag step is also not NaN.
    if pd.isna(icf_to_enroll_lag):
        icf_to_enroll_lag = 0.0

    for icf in all_icfs_to_project:
        if pd.isna(icf['start_date']): continue
        # Calculate when the ICF is expected to land
        icf_landing_date = icf['start_date'] + pd.to_timedelta(icf['lag_to_icf'], unit='D')
        
        projected_enrollments.append({
            'prob': icf['prob'] * icf_to_enroll_rate,
            'lag': icf_to_enroll_lag, # This is the additional lag from ICF to Enrollment
            'start_date': icf_landing_date # The clock for the enrollment lag starts when the ICF lands
        })

    # --- 6. Aggregate and Smear Projections ---
    days_in_avg_month = 30.4375
    proj_start_month = pd.Period(datetime.now(), 'M')
    max_lag_enroll = max([p['lag'] for p in projected_enrollments if pd.notna(p['lag'])], default=0)
    max_lag_icf = max([p['lag_to_icf'] for p in all_icfs_to_project if pd.notna(p['lag_to_icf'])], default=0)
    proj_horizon = int(np.ceil((max_lag_icf + max_lag_enroll) / days_in_avg_month)) + 3
    proj_horizon = max(proj_horizon, 6)
    
    future_months = pd.period_range(start=proj_start_month, periods=proj_horizon, freq='M')
    results_df = pd.DataFrame(0.0, index=future_months, columns=['Projected_ICF_Landed', 'Projected_Enrollments_Landed'])

    def smear_projection(df, projections_list, lag_col_name, target_col):
        for proj in projections_list:
            if proj['prob'] <= 0 or pd.isna(proj['start_date']) or pd.isna(proj[lag_col_name]): continue
            landing_date = proj['start_date'] + pd.to_timedelta(proj[lag_col_name], unit='D')
            landing_period = pd.Period(landing_date, 'M')
            if landing_period in df.index:
                df.loc[landing_period, target_col] += proj['prob']
        return df

    # Smear using the correct lag column name for each projection type
    results_df = smear_projection(results_df, all_icfs_to_project, 'lag_to_icf', 'Projected_ICF_Landed')
    results_df = smear_projection(results_df, projected_enrollments, 'lag', 'Projected_Enrollments_Landed')

    # Final formatting
    results_df['Projected_ICF_Landed'] = results_df['Projected_ICF_Landed'].round(0).astype(int)
    results_df['Cumulative_ICF_Landed'] = results_df['Projected_ICF_Landed'].cumsum()
    results_df['Projected_Enrollments_Landed'] = results_df['Projected_Enrollments_Landed'].round(0).astype(int)
    results_df['Cumulative_Enrollments_Landed'] = results_df['Projected_Enrollments_Landed'].cumsum()

    return results_df

# --- NEW: UI TAB FOR FUNNEL ANALYSIS ---
def render_funnel_analysis_tab():
    st.header("🔬 Funnel Analysis (Based on Current Pipeline)")
    st.info("""
    This forecast shows the expected outcomes (**ICFs & Enrollments**) from the leads **already in your funnel**.
    It answers the question: "If we stopped all new recruitment activities today, what results would we still see and when?"
    """)

    st.markdown("---")
    st.subheader("Funnel Analysis Assumptions")

    # Conversion Rate Assumption
    rate_options_display = {
        "Manual Input Below": "Manual", "Overall Historical Average": "Overall", 
        "1-Month Rolling Avg.": "1-Month", "3-Month Rolling Avg.": "3-Month", "6-Month Rolling Avg.": "6-Month"
    }
    selected_rate_method_label = st.radio(
        "Base Funnel Conversion Rates On:", options=list(rate_options_display.keys()), index=2, 
        key="fa_rate_method_radio", horizontal=True
    )
    
    fa_rate_method_internal = "Manual Input Below"
    fa_rolling_window = 0
    if selected_rate_method_label == "Overall Historical Average":
        fa_rate_method_internal = "Rolling Historical Average"; fa_rolling_window = 999
    elif "Rolling" in selected_rate_method_label:
        fa_rate_method_internal = "Rolling Historical Average"
        fa_rolling_window = int(selected_rate_method_label.split('-')[0])

    # Manual rate inputs
    manual_rates_fa = {}
    st.write("Define Manual Conversion Rates for Funnel Analysis:")
    fa_cols_rate = st.columns(3)
    with fa_cols_rate[0]:
        manual_rates_fa[f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}"] = st.slider("FA: POF -> PreScreen %", 0.0, 100.0, 95.0, key='fa_cr_qps', step=0.1, format="%.1f%%") / 100.0
        manual_rates_fa[f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}"] = st.slider("FA: PreScreen -> StS %", 0.0, 100.0, 20.0, key='fa_cr_pssts', step=0.1, format="%.1f%%") / 100.0
    with fa_cols_rate[1]:
        manual_rates_fa[f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}"] = st.slider("FA: StS -> Appt %", 0.0, 100.0, 45.0, key='fa_cr_sa', step=0.1, format="%.1f%%") / 100.0
        manual_rates_fa[f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"] = st.slider("FA: Appt -> ICF %", 0.0, 100.0, 55.0, key='fa_cr_ai', step=0.1, format="%.1f%%") / 100.0
    with fa_cols_rate[2]:
        manual_rates_fa[f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}"] = st.slider("FA: ICF -> Enrolled %", 0.0, 100.0, 85.0, key='fa_cr_ie', step=0.1, format="%.1f%%") / 100.0

    st.markdown("---")

    # --- Calculation & Display ---
    if st.button("🔬 Analyze Current Funnel", key="run_funnel_analysis"):
        
        # Ensure the new rate is included for rolling avg calculation
        manual_rates_with_enroll = manual_rates_fa.copy()
        if f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}" not in manual_rates_with_enroll:
             manual_rates_with_enroll[f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}"] = 0.85 # Default if not in UI somehow

        effective_rates_fa, rates_method_desc_fa = determine_effective_projection_rates(
            st.session_state.referral_data_processed, st.session_state.ordered_stages,
            st.session_state.ts_col_map, fa_rate_method_internal,
            fa_rolling_window, manual_rates_with_enroll, # Pass the full dictionary
            st.session_state.get('inter_stage_lags', {}), sidebar_display_area=None
        )

        st.session_state.funnel_analysis_results = calculate_pipeline_projection(
            st.session_state.referral_data_processed,
            st.session_state.ordered_stages,
            st.session_state.ts_col_map,
            st.session_state.get('inter_stage_lags', {}),
            effective_rates_fa,
            None # Lag model parameter is for future enhancement
        )
        st.session_state.funnel_analysis_rates_desc = rates_method_desc_fa

    if 'funnel_analysis_results' in st.session_state:
        results_df = st.session_state.funnel_analysis_results
        rates_desc = st.session_state.get('funnel_analysis_rates_desc', "N/A")
        st.caption(f"**Projection Using: {rates_desc} Conversion Rates**")

        if not results_df.empty:
            total_icfs = results_df['Projected_ICF_Landed'].sum()
            total_enrolls = results_df['Projected_Enrollments_Landed'].sum()

            fa_col1_res, fa_col2_res = st.columns(2)
            fa_col1_res.metric("Total Expected ICFs from Pipeline", f"{total_icfs:,.0f}")
            fa_col2_res.metric("Total Expected Enrollments from Pipeline", f"{total_enrolls:,.0f}")

            st.subheader("Projected Monthly Landings from Current Pipeline")
            display_df = results_df.copy()
            display_df.index = display_df.index.strftime('%Y-%m')
            
            display_df = display_df[['Projected_ICF_Landed', 'Projected_Enrollments_Landed']]
            st.dataframe(display_df.style.format("{:,.0f}"))
            
            st.subheader("Cumulative Projections Over Time")
            chart_df = results_df[['Cumulative_ICF_Landed', 'Cumulative_Enrollments_Landed']].copy()
            if isinstance(chart_df.index, pd.PeriodIndex):
                chart_df.index = chart_df.index.to_timestamp()
            st.line_chart(chart_df)

            try:
                csv_fa = results_df.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Funnel Analysis Data", data=csv_fa,
                    file_name='funnel_analysis_projection.csv', mime='text/csv',
                    key='dl_funnel_analysis'
                )
            except Exception as e:
                st.warning(f"Funnel Analysis download error: {e}")

        else:
            # This message is now handled inside the calculation function for better context
            pass
            
# --- Streamlit UI ---
# ... (Session state initializations remain the same) ...
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
ai_cpql_inflation_factor_sidebar = 0.0
ai_ql_volume_threshold_sidebar = 10.0
ai_monthly_ql_capacity_multiplier_sidebar_val = 3.0

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"], key="referral_uploader_main")
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (TSV)", type=["tsv"], key="funnel_uploader_main")
    st.divider()
    with st.expander("Historical Ad Spend"):
        st.info("Enter **historical** ad spend for past months. Add rows as needed.")
        edited_historical_spend_df = st.data_editor(
            st.session_state.historical_spend_df, num_rows="dynamic", key="hist_spend_editor_main", use_container_width=True,
            column_config={ "Month (YYYY-MM)": st.column_config.TextColumn(help="Enter month in YYYY-MM format",required=True), "Historical Spend": st.column_config.NumberColumn(help="Enter ad spend amount", min_value=0.0, format="$%.2f", required=True),})
        temp_ad_spend_input_dict = {}
        valid_hist_spend_entries = True
        for index, row in edited_historical_spend_df.iterrows():
            month_str = str(row['Month (YYYY-MM)']).strip(); spend_val = row['Historical Spend']
            if month_str and pd.notna(spend_val) and spend_val >=0 :
                try: month_period = pd.Period(month_str, freq='M'); temp_ad_spend_input_dict[month_period] = float(spend_val)
                except ValueError: st.sidebar.error(f"Invalid month format for hist. spend: '{month_str}'. Use YYYY-MM."); valid_hist_spend_entries = False; break
                except Exception as e: st.sidebar.error(f"Error processing hist. spend row ({month_str}, {spend_val}): {e}"); valid_hist_spend_entries = False; break
            elif not month_str and pd.notna(spend_val): st.sidebar.warning(f"Row {index+1} in historical spend has amount but no month. Ignoring.");
        if valid_hist_spend_entries: ad_spend_input_dict = temp_ad_spend_input_dict; st.session_state.historical_spend_df = edited_historical_spend_df
    st.divider()
    with st.expander("Performance Scoring Weights"): 
        weights_input_local = {}
        weights_input_local["Qual -> ICF %"] = st.slider("Qual (POF) -> ICF %", 0, 100, 20, key='w_qicf_v2')
        weights_input_local["Avg TTC (Days)"] = st.slider("Avg Time to Contact", 0, 100, 25, key='w_ttc_v2', help="Note: This metric is more relevant for Site Performance.")
        weights_input_local["Avg Funnel Movement Steps"] = st.slider("Avg Funnel Movement Steps", 0, 100, 5, key='w_fms_v2', help="Note: This metric is more relevant for Site Performance.")
        
        # Ensuring correct weight keys match output of calculate_grouped_performance_metrics
        weights_input_local["Screen Fail % (from ICF)"] = st.slider("Screen Fail % (from ICF)", 0, 100, 5, key='w_sfr_v3_generic', help="Screen Failures as % of ICFs. Used for Ad Performance.")
        weights_input_local["Site Screen Fail %"] = st.slider("Site Screen Fail %", 0, 100, 5, key='w_sfr_v2_site', help="Used for Site Performance.") # Keep specific for site if desired
        
        weights_input_local["StS -> Appt %"] = st.slider("StS -> Appt Sched %", 0, 100, 30, key='w_sa_site_score_v2')
        weights_input_local["Appt -> ICF %"] = st.slider("Appt Sched -> ICF %", 0, 100, 15, key='w_ai_site_score_v2')
        weights_input_local["Lag Qual -> ICF (Days)"] = st.slider("Lag Qual (POF) -> ICF (Days)", 0, 100, 0, key='w_lagqicf_v2')

        weights_input_local["Projection Lag (Days)"] = st.slider("Overall Projection Lag (Days)", 0, 100, 0, key='w_projlag_v3', help="Sum of avg lags for key funnel segments; for Ad Performance.")
        weights_input_local["Site Projection Lag (Days)"] = st.slider("Site Projection Lag (Days)", 0, 100, 0, key='w_siteprojlag_v2', help="Used for Site Performance.")

        total_weight_input_local = sum(abs(w) for w in weights_input_local.values())
        if total_weight_input_local > 0: weights_normalized = {k: v / total_weight_input_local for k, v in weights_input_local.items()}
        else: weights_normalized = {k: 0 for k in weights_input_local}
        st.caption(f"Weights normalized. Lower is better for TTC, Screen Fail %, Lags.")
    st.divider()
    with st.expander("Projection Assumptions (Main Projections Tab)", expanded=False):
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
        if current_spend_df_rows != proj_horizon_editor_val or 'proj_spend_df_cache' not in st.session_state or \
           (not st.session_state.proj_spend_df_cache.empty and pd.Period(st.session_state.proj_spend_df_cache['Month'].iloc[0], freq='M') != future_months_ui_for_editor[0]):
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
        if current_spend_df_rows != proj_horizon_editor_val or 'proj_cpqr_df_cache' not in st.session_state or \
            (not st.session_state.proj_cpqr_df_cache.empty and pd.Period(st.session_state.proj_cpqr_df_cache['Month'].iloc[0], freq='M') != future_months_ui_for_editor[0]):
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
            rolling_window_months_sidebar = st.selectbox("Select Rolling Window (Months):", [1, 3, 6, 999], index=1, format_func=lambda x: "Overall Average" if x==999 else f"{x}-Month", key='rolling_window_v2_sel')
            if st.session_state.data_processed_successfully and st.session_state.referral_data_processed is not None and \
               st.session_state.ordered_stages is not None and st.session_state.ts_col_map is not None:
                # Prepare a manual rates dict that includes all potential stages for rolling calc
                manual_rates_for_rolling_calc = manual_proj_conv_rates_sidebar.copy()
                manual_rates_for_rolling_calc[f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}"] = 0.85 # Add a default
                determine_effective_projection_rates(
                    st.session_state.referral_data_processed, st.session_state.ordered_stages,
                    st.session_state.ts_col_map, rate_assumption_method_sidebar,
                    rolling_window_months_sidebar, manual_rates_for_rolling_calc,
                    st.session_state.get('inter_stage_lags', {}), sidebar_display_area=st.sidebar)
            else: st.sidebar.caption("Upload data to view calculated rolling rates.")
        else: rolling_window_months_sidebar = 0; st.sidebar.caption("Using manually input rates above for Projections tab.")
    st.divider()
    with st.expander("Shared Projection Sensitivity & AI Settings", expanded=True):
        proj_icf_variation_percent_sidebar = st.slider(
            "Projected ICF Variation (+/- %)", min_value=0, max_value=50, value=proj_icf_variation_percent_sidebar, step=1,
            key='proj_icf_variation_widget_v3',
            help="Applies to ICFs generated by cohorts for CPICF range on 'Projections' & 'AI Forecast' tabs."
        )
        st.subheader("Diminishing Returns on CPQL (AI Forecast)")
        ai_cpql_inflation_factor_sidebar = st.slider(
            "CPQL Inflation Factor (%)", 0.0, 25.0, 5.0, 0.5, key="ai_cpql_inflation_v2",
            help="For each threshold unit of QL volume increase, CPQL goes up by this percent."
        )
        ai_ql_volume_threshold_sidebar = st.slider(
            "QL Volume Increase Threshold (%) for Inflation", 1.0, 50.0, 10.0, 1.0, key="ai_ql_vol_thresh_v2",
            help="Inflation applies for every X% increase in monthly QLs over historical baseline."
        )
        ai_monthly_ql_capacity_multiplier_sidebar_val = st.slider(
            "AI: Monthly QL Capacity Multiplier",
            min_value=1.0, max_value=30.0, value=ai_monthly_ql_capacity_multiplier_sidebar_val, step=0.5,
            key="ai_monthly_ql_cap_mult_v1",
            help="Multiplier for baseline QLs a single month will attempt to handle during AI forecast planning. Higher values allow more QLs per month if needed by the timeline."
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
    
    # UPDATED TABS
    tab_titles = ["📅 Monthly ProForma", "🏆 Site Performance", "📢 Ad Performance", "📈 Projections", "🔬 Funnel Analysis", "🤖 AI Forecast"]
    tabs = st.tabs(tab_titles)

    tab_proforma = tabs[0]
    tab_site_perf = tabs[1]
    tab_ad_perf = tabs[2]
    tab_projections = tabs[3]
    tab_funnel_analysis = tabs[4] # NEW TAB
    tab_ai_forecast = tabs[5]


    with tab_proforma: 
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

    with tab_site_perf: 
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
                    if '%' in col_fmt and final_ranked_display_tab2[col_fmt].dtype == 'float':
                        final_ranked_display_tab2[col_fmt] = final_ranked_display_tab2[col_fmt].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '-')
                    elif ('Lag' in col_fmt or 'TTC' in col_fmt or 'Steps' in col_fmt) and final_ranked_display_tab2[col_fmt].dtype == 'float':
                        final_ranked_display_tab2[col_fmt] = final_ranked_display_tab2[col_fmt].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
                    elif ('Count' in col_fmt or 'Qualified' in col_fmt) and pd.api.types.is_numeric_dtype(final_ranked_display_tab2[col_fmt]):
                         final_ranked_display_tab2[col_fmt] = final_ranked_display_tab2[col_fmt].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x==x else '-')
                st.dataframe(final_ranked_display_tab2.style.format(na_rep='-'))
                try: csv_tab2 = final_ranked_display_tab2.to_csv(index=False).encode('utf-8'); st.download_button(label="Download Site Ranking", data=csv_tab2, file_name='site_ranking.csv', mime='text/csv', key='dl_sites_v2')
                except Exception as e_dl2: st.warning(f"Site ranking download error: {e_dl2}")
            else: st.warning("Site ranking display table is empty after filtering columns.")
        elif site_metrics_calculated_data.empty: st.warning("Site metrics have not been calculated (e.g. no 'Site' column in data or error during calculation). Site performance cannot be shown.")
        else: st.warning("Site performance cannot be ranked until data is loaded and weights are set.")

    # --- NEW AD PERFORMANCE TAB ---
    with tab_ad_perf:
        st.header("Ad Channel Performance")
        st.caption("Performance metrics grouped by UTM parameters, scored using the same weights as Site Performance.")

        if not st.session_state.data_processed_successfully or referral_data_processed is None or referral_data_processed.empty:
            st.warning("Please upload and process referral data first.")
        elif "UTM Source" not in referral_data_processed.columns:
            st.warning("UTM Source column not found in the uploaded data. Ad Performance cannot be calculated.")
        else:
            df_for_ad_perf = referral_data_processed.copy()
            st.subheader("Performance by UTM Source")
            utm_source_metrics_df = calculate_grouped_performance_metrics(
                df_for_ad_perf, ordered_stages, ts_col_map,
                grouping_cols=["UTM Source"], 
                unclassified_label="Unclassified Source"
            )

            if not utm_source_metrics_df.empty:
                ranked_utm_source_df = score_performance_groups(
                    utm_source_metrics_df, weights_normalized,
                    group_col_name="UTM Source" 
                )
                
                display_cols_ad = ['UTM Source', 'Score', 'Grade', 'Total Qualified', 'PSA Count', 'StS Count', 'Appt Count', 'ICF Count',
                                   'Qual -> ICF %', 'POF -> PSA %', 'PSA -> StS %', 'StS -> Appt %', 'Appt -> ICF %',
                                   'Lag Qual -> ICF (Days)', 'Projection Lag (Days)', 'Screen Fail % (from ICF)',
                                   'Avg TTC (Days)', 'Avg Funnel Movement Steps'] 
                
                display_cols_ad_exist = [col for col in display_cols_ad if col in ranked_utm_source_df.columns]
                final_ad_display = ranked_utm_source_df[display_cols_ad_exist].copy()

                if not final_ad_display.empty:
                    if 'Score' in final_ad_display.columns: final_ad_display['Score'] = final_ad_display['Score'].round(1)
                    for col_fmt in final_ad_display.columns:
                        if '%' in col_fmt and final_ad_display[col_fmt].dtype == 'float':
                            final_ad_display[col_fmt] = final_ad_display[col_fmt].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else '-')
                        elif ('Lag' in col_fmt or 'TTC' in col_fmt or 'Steps' in col_fmt) and final_ad_display[col_fmt].dtype == 'float':
                            final_ad_display[col_fmt] = final_ad_display[col_fmt].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
                        elif ('Count' in col_fmt or 'Qualified' in col_fmt) and pd.api.types.is_numeric_dtype(final_ad_display[col_fmt]):
                            final_ad_display[col_fmt] = final_ad_display[col_fmt].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x==x else '-')
                    st.dataframe(final_ad_display.style.format(na_rep='-'))
                    try:
                        csv_ad_source = final_ad_display.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download UTM Source Performance", data=csv_ad_source, file_name='utm_source_performance.csv', mime='text/csv', key='dl_ad_source_perf_v2')
                    except Exception as e_dl_ads: st.warning(f"UTM Source performance download error: {e_dl_ads}")
                else: st.info("No data to display for UTM Source performance after processing.")
            else: st.info("Could not calculate performance metrics for UTM Source.")


    with tab_projections: 
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
                
                # Use the full manual rates dict to ensure all keys are checked
                manual_rates_with_enroll_tab3 = manual_proj_conv_rates_sidebar.copy()
                manual_rates_with_enroll_tab3[f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}"] = 0.85 # Default
                
                if inter_stage_lags_data:
                    for r_key_disp_tab3 in manual_rates_with_enroll_tab3.keys():
                        avg_lag_disp_tab3 = inter_stage_lags_data.get(r_key_disp_tab3)
                        if pd.notna(avg_lag_disp_tab3) and avg_lag_disp_tab3 > 0:
                            calc_mat_disp_tab3 = round(display_maturity_lag_multiplier_tab3 * avg_lag_disp_tab3)
                            temp_maturity_periods_display_tab3[r_key_disp_tab3] = max(calc_mat_disp_tab3, display_min_effective_maturity_tab3)
                        else: temp_maturity_periods_display_tab3[r_key_disp_tab3] = display_default_maturity_tab3
                for key_lag, val_lag in inter_stage_lags_data.items():
                    maturity_p_disp_tab3 = temp_maturity_periods_display_tab3.get(key_lag, "N/A (Not a projection rate segment)")
                    lag_df_list_tab3.append({'Stage Transition': key_lag, 'Avg Lag (Days)': f"{val_lag:.1f}" if pd.notna(val_lag) else "N/A", 'Implied Maturity Used (Days)': f"{maturity_p_disp_tab3}" if isinstance(maturity_p_disp_tab3, (int, float)) else maturity_p_disp_tab3})
                if lag_df_list_tab3: st.table(pd.DataFrame(lag_df_list_tab3))
                else: st.caption("No inter-stage lags calculated or available to determine maturity periods.")
            else: st.caption("Inter-stage lags have not been calculated (required for rolling rate maturity).")
        st.markdown("---")
        projection_inputs_tab3 = {
            'horizon': proj_horizon_sidebar, 'spend_dict': proj_spend_dict_sidebar,
            'cpqr_dict': proj_cpqr_dict_sidebar, 'final_conv_rates': effective_proj_rates_tab3,
            'goal_icf': goal_icf_count_sidebar, 'site_performance_data': site_metrics_calculated_data,
            'inter_stage_lags': inter_stage_lags_data, 'icf_variation_percentage': proj_icf_variation_percent_sidebar
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
                    elif 'Qual_Referrals' in col_name_fmt_tab3 or 'ICF_Landed' in col_name_fmt_tab3 : results_display_filtered_tab3[col_name_fmt_tab3] = results_display_filtered_tab3[col_name_fmt_tab3].apply(lambda x: f"{int(x):,}" if pd.notna(x) and isinstance(x, (int,float)) and x==x else (x if isinstance(x,str) else '-'))
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
            df_for_editor_tab3 = site_level_proj_tab3.copy()
            if df_for_editor_tab3.index.name == 'Site': df_for_editor_tab3.reset_index(inplace=True)
            grand_total_row_tab3 = None
            if 'Site' in df_for_editor_tab3.columns and "Grand Total" in df_for_editor_tab3['Site'].values:
                grand_total_row_tab3 = df_for_editor_tab3[df_for_editor_tab3['Site'] == "Grand Total"].copy()
                df_for_editor_tab3 = df_for_editor_tab3[df_for_editor_tab3['Site'] != "Grand Total"].copy()
            elif "Grand Total" in df_for_editor_tab3.index:
                grand_total_row_tab3 = df_for_editor_tab3.loc[["Grand Total"]].copy()
                df_for_editor_tab3 = df_for_editor_tab3.drop(index="Grand Total", errors='ignore')
            if isinstance(df_for_editor_tab3.columns, pd.MultiIndex):
                df_for_editor_tab3.columns = [f"{col[1]} ({col[0]})" for col in df_for_editor_tab3.columns]
                if grand_total_row_tab3 is not None and isinstance(grand_total_row_tab3.columns, pd.MultiIndex):
                     grand_total_row_tab3.columns = [f"{col[1]} ({col[0]})" for col in grand_total_row_tab3.columns]
            if 'Site' in df_for_editor_tab3.columns:
                cols_ordered_tab3 = ['Site'] + [col for col in df_for_editor_tab3.columns if col != 'Site']
                df_for_editor_tab3 = df_for_editor_tab3[cols_ordered_tab3]
            st.caption("Edit projected QLs or ICFs per site per month. Note: Totals below are based on initial calculation.")
            edited_site_level_df_tab3 = st.data_editor(df_for_editor_tab3, use_container_width=True, key="site_level_editor_v3_tab3", num_rows="dynamic")
            if grand_total_row_tab3 is not None and not grand_total_row_tab3.empty:
                st.caption("Totals (based on initial calculation, not live edits from above table):")
                grand_total_row_tab3_display = grand_total_row_tab3.copy()
                if 'Site' not in grand_total_row_tab3_display.columns:
                    if grand_total_row_tab3_display.index.name == 'Site' or "Grand Total" in grand_total_row_tab3_display.index:
                         grand_total_row_tab3_display = grand_total_row_tab3_display.reset_index()
                         if grand_total_row_tab3_display.columns[0] == 'index':
                             grand_total_row_tab3_display.rename(columns={'index':'Site'}, inplace=True)
                if 'Site' in grand_total_row_tab3_display.columns:
                    gt_cols_ordered_tab3 = ['Site'] + [col for col in grand_total_row_tab3_display.columns if col != 'Site']
                    grand_total_row_tab3_display = grand_total_row_tab3_display[gt_cols_ordered_tab3]
                numeric_cols_gt_tab3 = grand_total_row_tab3_display.select_dtypes(include=np.number).columns
                formatters_gt_tab3 = {col: "{:,.0f}" for col in numeric_cols_gt_tab3}
                st.dataframe(grand_total_row_tab3_display.style.format(formatters_gt_tab3, na_rep='0'), use_container_width=True)
            try:
                csv_site_proj_tab3 = edited_site_level_df_tab3.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Edited Site Projections (Tab 3)", data=csv_site_proj_tab3, file_name='edited_site_level_projections_tab3.csv', mime='text/csv', key='dl_edited_site_proj_tab3_v2')
            except Exception as e_dl_site3: st.warning(f"Site projection (Tab 3) download error: {e_dl_site3}")
        else: st.info("Site-level projection data (Projections Tab) is not available or is empty.")

    # --- NEW: FUNNEL ANALYSIS TAB ---
    with tab_funnel_analysis:
        render_funnel_analysis_tab()

    with tab_ai_forecast:
        st.header("AI Forecast (Goal-Based)")
        st.info("""
        Define your recruitment goals. The tool will estimate a monthly plan using a 'frontloading' strategy
        (prioritizing earlier months for activity, up to a monthly capacity) to meet your LPI.
        - **Conversion Rates:** Choose how historical conversion rates are applied.
        - **Lag Assumption:** Choose between using an overall average POF->ICF lag or a P25/P50/P75 day distribution for ICF landing.
        - **Site Activity:** Define activation/deactivation dates for sites to influence QL allocation.
        - **CPQL:** Your estimated Cost Per Qualified Lead (e.g., for "Passed Online Form").
        - **Site Caps:** Optionally, set monthly QL (POF) limits per site.
        - **CPQL Inflation & Monthly QL Multiplier:** Optionally, model increasing CPQL with higher volume and control monthly QL planning aggressiveness (set in sidebar).
        - **ICF Variation:** Applies a +/- percentage to generated ICFs for CPICF sensitivity (set in sidebar).
        """)
        # CONTINUATION FROM HERE...
        ai_cols_goals = st.columns(3)
        with ai_cols_goals[0]:
            ai_goal_lpi_date = st.date_input("Target LPI Date", value=datetime.now() + pd.DateOffset(months=12), min_value=datetime.now() + pd.DateOffset(months=1), key="ai_lpi_date_v3")
        with ai_cols_goals[1]:
            ai_goal_icf_num = st.number_input("Target Total ICFs", min_value=1, value=100, step=10, key="ai_icf_num_v3")
        with ai_cols_goals[2]:
            ai_cpql_estimate = st.number_input("Base Estimated CPQL (POF)", min_value=1.0, value=75.0, step=5.0, format="%.2f", key="ai_cpql_v3", help="Your average cost for a 'Passed Online Form' lead. This may be adjusted by inflation settings.")

        st.markdown("---"); st.subheader("AI Forecast Assumptions")

        ai_lag_assumption_method = st.radio(
            "ICF Landing Lag Assumption:",
            ("Use Overall Average POF->ICF Lag", "Use P25/P50/P75 Day Lag Distribution"),
            key="ai_lag_method_radio_v1", horizontal=True,
            help="Determines how generated ICFs are distributed to landing months."
        )
        ai_lag_p25_days_input, ai_lag_p50_days_input, ai_lag_p75_days_input = None, None, None
        if ai_lag_assumption_method == "Use P25/P50/P75 Day Lag Distribution":
            lag_cols = st.columns(3)
            with lag_cols[0]: ai_lag_p25_days_input = st.number_input("P25 Lag (Days)", min_value=0, value=20, step=1, key="ai_lag_p25_v1")
            with lag_cols[1]: ai_lag_p50_days_input = st.number_input("P50 (Median) Lag (Days)", min_value=0, value=30, step=1, key="ai_lag_p50_v1")
            with lag_cols[2]: ai_lag_p75_days_input = st.number_input("P75 Lag (Days)", min_value=0, value=45, step=1, key="ai_lag_p75_v1")
            if ai_lag_p25_days_input is not None and ai_lag_p50_days_input is not None and ai_lag_p75_days_input is not None: 
                if not (ai_lag_p25_days_input <= ai_lag_p50_days_input <= ai_lag_p75_days_input):
                    st.warning("P25 lag should be <= P50, and P50 should be <= P75 for logical distribution.")

        rate_options_ai_display = {"Manual Input Below": "Manual Input Below", "Overall Historical Average": "Overall Historical", "1-Month Rolling Avg.": "1-Month Rolling", "3-Month Rolling Avg.": "3-Month Rolling", "6-Month Rolling Avg.": "6-Month Rolling"}
        selected_rate_method_label_ai_tab = st.radio("Base AI Forecast Conversion Rates On:", options=list(rate_options_ai_display.keys()), index=4, key="ai_rate_method_radio_v3", horizontal=True)
        ai_rate_assumption_method_internal_val = "Manual Input Below"; ai_rolling_window_months_internal_val = 0
        if selected_rate_method_label_ai_tab == "Overall Historical Average": ai_rate_assumption_method_internal_val = "Rolling Historical Average"; ai_rolling_window_months_internal_val = 999
        elif "Rolling" in selected_rate_method_label_ai_tab:
            ai_rate_assumption_method_internal_val = "Rolling Historical Average"
            try: ai_rolling_window_months_internal_val = int(selected_rate_method_label_ai_tab.split('-')[0])
            except: ai_rolling_window_months_internal_val = 3
        ai_manual_conv_rates_tab_input_val = {}
        if selected_rate_method_label_ai_tab == "Manual Input Below":
            st.write("Define Manual Conversion Rates for AI Forecast:")
            ai_cols_rate_man = st.columns(2)
            with ai_cols_rate_man[0]:
                 ai_manual_conv_rates_tab_input_val[f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}"] = st.slider("AI: POF -> PreScreen %", 0.0, 100.0, 90.0, step=0.1, format="%.1f%%", key='ai_cr_qps_v5') / 100.0
                 ai_manual_conv_rates_tab_input_val[f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}"] = st.slider("AI: PreScreen -> StS %", 0.0, 100.0, 25.0, step=0.1, format="%.1f%%", key='ai_cr_pssts_v5') / 100.0
            with ai_cols_rate_man[1]:
                 ai_manual_conv_rates_tab_input_val[f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}"] = st.slider("AI: StS -> Appt %", 0.0, 100.0, 50.0, step=0.1, format="%.1f%%", key='ai_cr_sa_v5') / 100.0
                 ai_manual_conv_rates_tab_input_val[f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"] = st.slider("AI: Appt -> ICF %", 0.0, 100.0, 60.0, step=0.1, format="%.1f%%", key='ai_cr_ai_v5') / 100.0

        st.markdown("---")

        st.subheader("Site Activation/Deactivation Dates (Optional)")
        st.caption("Define when sites are active for QL allocation. Leave dates blank if active for the entire projection. Dates must be in YYYY-MM format.")
        default_site_activity_schedule_input = {} 
        site_activity_editor_data_list = []

        if not site_metrics_calculated_data.empty and 'Site' in site_metrics_calculated_data.columns:
            for site_name_act_iter in site_metrics_calculated_data['Site'].unique():
                site_activity_editor_data_list.append({
                    "Site": site_name_act_iter,
                    "Activation Date (YYYY-MM)": None, 
                    "Deactivation Date (YYYY-MM)": None 
                })

            if site_activity_editor_data_list:
                # Initialize session state for the data editor if it doesn't exist
                if 'edited_site_activity_df' not in st.session_state:
                    st.session_state.edited_site_activity_df = pd.DataFrame(site_activity_editor_data_list)
                
                edited_site_activity_df_ui = st.data_editor(
                    st.session_state.edited_site_activity_df, # Use session state here
                    key="ai_site_activity_editor_v1", 
                    use_container_width=True,
                    column_config={
                        "Site": st.column_config.TextColumn(disabled=True),
                        "Activation Date (YYYY-MM)": st.column_config.TextColumn(help="YYYY-MM or leave blank if active from start."),
                        "Deactivation Date (YYYY-MM)": st.column_config.TextColumn(help="YYYY-MM or leave blank if active till end.")
                    },
                    num_rows="dynamic" 
                )
                # Update session state if the dataframe has been changed by the user
                if not edited_site_activity_df_ui.equals(st.session_state.edited_site_activity_df):
                    st.session_state.edited_site_activity_df = edited_site_activity_df_ui


                if edited_site_activity_df_ui is not None:
                    for _, row_act_val in edited_site_activity_df_ui.iterrows():
                        site_name_from_editor = row_act_val["Site"]
                        act_date_str = str(row_act_val["Activation Date (YYYY-MM)"]).strip() if pd.notna(row_act_val["Activation Date (YYYY-MM)"]) else ""
                        deact_date_str = str(row_act_val["Deactivation Date (YYYY-MM)"]).strip() if pd.notna(row_act_val["Deactivation Date (YYYY-MM)"]) else ""
                        act_period, deact_period = None, None
                        try:
                            if act_date_str: act_period = pd.Period(act_date_str, freq='M')
                        except ValueError: st.warning(f"Invalid activation date format for site {site_name_from_editor}: '{act_date_str}'. Ignoring.")
                        try:
                            if deact_date_str: deact_period = pd.Period(deact_date_str, freq='M')
                        except ValueError: st.warning(f"Invalid deactivation date format for site {site_name_from_editor}: '{deact_date_str}'. Ignoring.")

                        if act_period and deact_period and act_period > deact_period:
                            st.warning(f"Activation date for site {site_name_from_editor} is after deactivation date. Please correct.")
                        else:
                            default_site_activity_schedule_input[site_name_from_editor] = {
                                'activation_period': act_period,
                                'deactivation_period': deact_period
                            }
            else:
                st.caption("No site data (from 'Site Metrics') available to define activity dates.")
        else:
            st.caption("Site metrics not calculated. Upload data and ensure 'Site' column exists to define activity dates.")
        
        st.markdown("---"); st.subheader("Site-Specific Monthly QL (POF) Caps (Optional)")
        default_site_caps_ai_input_val = {}
        if not site_metrics_calculated_data.empty and 'Site' in site_metrics_calculated_data.columns and referral_data_processed is not None:
            site_cap_editor_data_list = []
            avg_monthly_ql_per_site_val = {}
            ts_pof_col_for_avg_calc_val = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
            if ts_pof_col_for_avg_calc_val and ts_pof_col_for_avg_calc_val in referral_data_processed.columns and 'Site' in referral_data_processed.columns:
                temp_df_for_avg_val = referral_data_processed.copy()
                if not pd.api.types.is_period_dtype(temp_df_for_avg_val['Submission_Month']):
                    try: temp_df_for_avg_val['Submission_Month'] = temp_df_for_avg_val['Submission_Month'].dt.to_period('M')
                    except AttributeError:
                        try: temp_df_for_avg_val['Submission_Month'] = pd.Series(temp_df_for_avg_val['Submission_Month']).apply(lambda x: pd.Period(x, freq='M'))
                        except: pass
                if pd.api.types.is_period_dtype(temp_df_for_avg_val['Submission_Month']):
                    site_monthly_counts_for_avg_val = temp_df_for_avg_val.dropna(subset=[ts_pof_col_for_avg_calc_val]).groupby(['Site', 'Submission_Month']).size().reset_index(name='MonthlyPOFCount')
                    if not site_monthly_counts_for_avg_val.empty:
                        avg_monthly_ql_per_site_val = site_monthly_counts_for_avg_val.groupby('Site')['MonthlyPOFCount'].mean().round(0).astype(int).to_dict()
            for site_name_cap_iter_val in site_metrics_calculated_data['Site'].unique():
                site_cap_editor_data_list.append({ "Site": site_name_cap_iter_val, "Historical Avg. Monthly POF": avg_monthly_ql_per_site_val.get(site_name_cap_iter_val, 0), "Monthly POF Cap": np.nan })
            
            if site_cap_editor_data_list:
                st.caption("Set a maximum number of 'Passed Online Form' (POF) leads a site can handle per month. Leave blank for no cap.")
                # Initialize session state for the site caps editor
                if 'edited_site_caps_df_ai_val' not in st.session_state:
                    st.session_state.edited_site_caps_df_ai_val = pd.DataFrame(site_cap_editor_data_list)

                edited_site_caps_df_ai_val_ui = st.data_editor( 
                    st.session_state.edited_site_caps_df_ai_val, # Use session state
                    key="ai_site_caps_editor_v3", 
                    use_container_width=True,
                    column_config={ 
                        "Site": st.column_config.TextColumn(disabled=True), 
                        "Historical Avg. Monthly POF": st.column_config.NumberColumn(format="%d", disabled=True), 
                        "Monthly POF Cap": st.column_config.NumberColumn(min_value=0, format="%d", step=1)
                    },
                    num_rows="dynamic" 
                )
                if not edited_site_caps_df_ai_val_ui.equals(st.session_state.edited_site_caps_df_ai_val):
                     st.session_state.edited_site_caps_df_ai_val = edited_site_caps_df_ai_val_ui


                if edited_site_caps_df_ai_val_ui is not None:
                    for _, row_cap_ai_val in edited_site_caps_df_ai_val_ui.iterrows():
                        if pd.notna(row_cap_ai_val["Monthly POF Cap"]) and row_cap_ai_val["Monthly POF Cap"] >= 0:
                            default_site_caps_ai_input_val[row_cap_ai_val["Site"]] = int(row_cap_ai_val["Monthly POF Cap"])
            else: st.caption("No site data available to set caps.")
        else: st.caption("Site performance data or referral data not available for setting caps.")
        st.markdown("---")

        if st.button("🚀 Generate AI Forecast", key="run_ai_forecast_v5"):
            st.session_state.ai_forecast_results = None
            for key_debug in ['ai_gen_df_debug_primary', 'ai_results_df_debug_primary', 'ai_gen_df_debug_best_case', 'ai_results_df_debug_best_case']:
                if key_debug in st.session_state: del st.session_state[key_debug]

            if selected_rate_method_label_ai_tab == "Manual Input Below":
                ai_effective_rates = ai_manual_conv_rates_tab_input_val
                ai_rates_method_desc = "Manual Input for AI Forecast"
            else:
                # Add all possible rates to the manual dict so rolling calc can find them
                manual_rates_for_ai_rolling = manual_proj_conv_rates_sidebar.copy()
                manual_rates_for_ai_rolling.update(ai_manual_conv_rates_tab_input_val)
                manual_rates_for_ai_rolling[f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}"] = 0.85

                ai_effective_rates, ai_rates_method_desc = determine_effective_projection_rates(
                    referral_data_processed, ordered_stages, ts_col_map, ai_rate_assumption_method_internal_val,
                    ai_rolling_window_months_internal_val, manual_rates_for_ai_rolling,
                    inter_stage_lags_data, sidebar_display_area=None )
                if "Error" in ai_rates_method_desc or "Failed" in ai_rates_method_desc or "No History" in ai_rates_method_desc or not ai_effective_rates or all(v == 0 for v in ai_effective_rates.values()):
                    st.warning(f"Could not determine reliable '{selected_rate_method_label_ai_tab}' rates for AI Forecast ({ai_rates_method_desc}). Using manual rates from Projections Tab sidebar as fallback.")
                    ai_effective_rates = manual_proj_conv_rates_sidebar
                    ai_rates_method_desc = f"Manual (Fallback from Projections Tab Sidebar due to issue with '{selected_rate_method_label_ai_tab}')"

            pof_ts_col_ai_val = ts_col_map.get(STAGE_PASSED_ONLINE_FORM); icf_ts_col_ai_val = ts_col_map.get(STAGE_SIGNED_ICF)
            avg_pof_icf_lag_ai_val = np.nan
            if pof_ts_col_ai_val and icf_ts_col_ai_val and \
               pof_ts_col_ai_val in referral_data_processed.columns and icf_ts_col_ai_val in referral_data_processed.columns:
                avg_pof_icf_lag_ai_val = calculate_avg_lag_generic(referral_data_processed, pof_ts_col_ai_val, icf_ts_col_ai_val)

            current_ai_lag_method = "average"
            current_p25, current_p50, current_p75 = None, None, None
            lag_source_message_ai_val = ""
            if ai_lag_assumption_method == "Use P25/P50/P75 Day Lag Distribution":
                if ai_lag_p25_days_input is not None and ai_lag_p50_days_input is not None and ai_lag_p75_days_input is not None and \
                   ai_lag_p25_days_input <= ai_lag_p50_days_input <= ai_lag_p75_days_input:
                    current_ai_lag_method = "percentiles"
                    current_p25 = float(ai_lag_p25_days_input); current_p50 = float(ai_lag_p50_days_input); current_p75 = float(ai_lag_p75_days_input)
                    lag_source_message_ai_val = f"P25/P50/P75 lags: {current_p25:.1f}/{current_p50:.1f}/{current_p75:.1f} days."
                else:
                    st.warning("Invalid P25/P50/P75 inputs or order. Defaulting to Overall Average POF->ICF Lag.")
                    if pd.notna(avg_pof_icf_lag_ai_val): lag_source_message_ai_val = f"Using fallback historical POF-ICF lag: {avg_pof_icf_lag_ai_val:.1f} days."
                    else: avg_pof_icf_lag_ai_val = 30.0; lag_source_message_ai_val = f"Using fallback default lag: {avg_pof_icf_lag_ai_val:.1f} days."
            else:
                if pd.notna(avg_pof_icf_lag_ai_val): lag_source_message_ai_val = f"Using historical POF-ICF lag: {avg_pof_icf_lag_ai_val:.1f} days."
                else: avg_pof_icf_lag_ai_val = 30.0; lag_source_message_ai_val = f"Using default POF-ICF lag: {avg_pof_icf_lag_ai_val:.1f} days (historical calculation failed)."

            if (current_ai_lag_method == "average" and pd.isna(avg_pof_icf_lag_ai_val)) or \
               (current_ai_lag_method == "percentiles" and not all([current_p25 is not None, current_p50 is not None, current_p75 is not None])):
                 st.error("Cannot run AI Forecast: Lag parameters are not correctly set.")
            elif not ai_effective_rates or all(r == 0 for r in ai_effective_rates.values()):
                st.error("Cannot run AI Forecast: Effective conversion rates are zero or unavailable.")
            else:
                st.write(f"AI Forecast using: {lag_source_message_ai_val} Conversion rates based on: {ai_rates_method_desc}")
                run_mode_for_call_primary_val = "primary"
                ai_results_df_run1, ai_site_df_run1, ai_ads_off_run1, ai_message_run1, ai_unfeasible_run1, ai_actual_icfs_run1 = calculate_ai_forecast_core(
                    goal_lpi_date_dt_orig=ai_goal_lpi_date, goal_icf_number_orig=ai_goal_icf_num,
                    estimated_cpql_user=ai_cpql_estimate, icf_variation_percent=proj_icf_variation_percent_sidebar,
                    processed_df=referral_data_processed, ordered_stages=ordered_stages, ts_col_map=ts_col_map,
                    effective_projection_conv_rates=ai_effective_rates,
                    avg_overall_lag_days=avg_pof_icf_lag_ai_val,
                    site_metrics_df=site_metrics_calculated_data, projection_horizon_months=proj_horizon_sidebar,
                    site_caps_input=default_site_caps_ai_input_val,
                    site_activity_schedule=default_site_activity_schedule_input, 
                    site_scoring_weights_for_ai=weights_normalized,
                    cpql_inflation_factor_pct=ai_cpql_inflation_factor_sidebar, ql_vol_increase_threshold_pct=ai_ql_volume_threshold_sidebar,
                    run_mode=run_mode_for_call_primary_val,
                    ai_monthly_ql_capacity_multiplier=ai_monthly_ql_capacity_multiplier_sidebar_val,
                    ai_lag_method=current_ai_lag_method,
                    ai_lag_p25_days=current_p25,
                    ai_lag_p50_days=current_p50,
                    ai_lag_p75_days=current_p75
                )

                st.session_state.ai_forecast_results = {
                    'df': ai_results_df_run1, 'site_df': ai_site_df_run1, 'ads_off': ai_ads_off_run1,
                    'message': ai_message_run1, 'unfeasible': ai_unfeasible_run1, 'actual_icfs': ai_actual_icfs_run1,
                    'lpi_goal': ai_goal_lpi_date, 'icf_goal': ai_goal_icf_num, 'run_mode_displayed': run_mode_for_call_primary_val
                }

                if ai_unfeasible_run1:
                    st.info(f"Initial AI forecast: {ai_message_run1}. Attempting best-case scenario by extending LPI to max projection horizon.")
                    run_mode_for_call_best_case_val = "best_case_extended_lpi"
                    ai_results_df_run2, ai_site_df_run2, ai_ads_off_run2, ai_message_run2, ai_unfeasible_run2, ai_actual_icfs_run2 = calculate_ai_forecast_core(
                        goal_lpi_date_dt_orig=ai_goal_lpi_date,
                        goal_icf_number_orig=ai_goal_icf_num,
                        estimated_cpql_user=ai_cpql_estimate, icf_variation_percent=proj_icf_variation_percent_sidebar,
                        processed_df=referral_data_processed, ordered_stages=ordered_stages, ts_col_map=ts_col_map,
                        effective_projection_conv_rates=ai_effective_rates,
                        avg_overall_lag_days=avg_pof_icf_lag_ai_val,
                        site_metrics_df=site_metrics_calculated_data, projection_horizon_months=proj_horizon_sidebar,
                        site_caps_input=default_site_caps_ai_input_val,
                        site_activity_schedule=default_site_activity_schedule_input, # NEW
                        site_scoring_weights_for_ai=weights_normalized,
                        cpql_inflation_factor_pct=ai_cpql_inflation_factor_sidebar, ql_vol_increase_threshold_pct=ai_ql_volume_threshold_sidebar,
                        run_mode=run_mode_for_call_best_case_val,
                        ai_monthly_ql_capacity_multiplier=ai_monthly_ql_capacity_multiplier_sidebar_val,
                        ai_lag_method=current_ai_lag_method,
                        ai_lag_p25_days=current_p25,
                        ai_lag_p50_days=current_p50,
                        ai_lag_p75_days=current_p75
                    )

                    st.session_state.ai_forecast_results = {
                        'df': ai_results_df_run2, 'site_df': ai_site_df_run2, 'ads_off': ai_ads_off_run2,
                        'message': ai_message_run2, 'unfeasible': ai_unfeasible_run2, 'actual_icfs': ai_actual_icfs_run2,
                        'lpi_goal': ai_goal_lpi_date, 'icf_goal': ai_goal_icf_num, 'run_mode_displayed': run_mode_for_call_best_case_val
                    }

        if st.session_state.get('ai_forecast_results'):
            results_ai_tab = st.session_state.ai_forecast_results
            ai_results_df_display = results_ai_tab['df']
            ai_site_df_display = results_ai_tab['site_df']
            ai_ads_off_display = results_ai_tab['ads_off']
            ai_message_display = results_ai_tab['message']
            ai_unfeasible_display_flag = results_ai_tab['unfeasible']
            ai_actual_icfs_display = results_ai_tab['actual_icfs']
            
            st.markdown("---")
            if not ai_unfeasible_display_flag :
                 st.success(f"Forecast Status: {ai_message_display}")
            elif "NEAR MISS" in ai_message_display:
                st.warning(f"Feasibility Note: {ai_message_display}")
            else:
                st.error(f"Feasibility Note: {ai_message_display}")

            ai_col1_res, ai_col2_res, ai_col3_res = st.columns(3)
            ai_col1_res.metric("Target LPI Date (Original Goal)", results_ai_tab['lpi_goal'].strftime("%Y-%m-%d"))

            goal_display_val_ai = f"{results_ai_tab['icf_goal']:,}"
            if ai_actual_icfs_display < results_ai_tab['icf_goal']:
                 goal_display_val_ai = f"{ai_actual_icfs_display:,.0f} (Original Goal: {results_ai_tab['icf_goal']:,})"
            ai_col2_res.metric("Projected/Goal ICFs", goal_display_val_ai)
            ai_col3_res.metric("Est. Ads Off Date (Generation)", ai_ads_off_display if ai_ads_off_display != "N/A" else "Past LPI/Goal Unmet")

            if ai_results_df_display is not None and not ai_results_df_display.empty:
                st.subheader("AI Forecasted Monthly Performance")
                ai_display_df_res_fmt = ai_results_df_display.copy();
                if isinstance(ai_display_df_res_fmt.index, pd.PeriodIndex): ai_display_df_res_fmt.index = ai_display_df_res_fmt.index.strftime('%Y-%m')
                if 'Target_QLs_POF' in ai_display_df_res_fmt.columns: ai_display_df_res_fmt.rename(columns={'Target_QLs_POF': 'Planned QLs (POF)'}, inplace=True)

                cols_to_show_ai_res_list = ['Planned QLs (POF)', 'Implied_Ad_Spend', 'Projected_ICF_Landed']
                if all(c in ai_display_df_res_fmt.columns for c in ['Projected_CPICF_Cohort_Source_Low', 'Projected_CPICF_Cohort_Source_Mean', 'Projected_CPICF_Cohort_Source_High']):
                    ai_display_df_res_fmt['Projected CPICF (Low-Mean-High)'] = ai_display_df_res_fmt.apply(
                        lambda row_cpicf: (f"${row_cpicf['Projected_CPICF_Cohort_Source_Low']:,.2f} - ${row_cpicf['Projected_CPICF_Cohort_Source_Mean']:,.2f} - ${row_cpicf['Projected_CPICF_Cohort_Source_High']:,.2f}"
                                     if pd.notna(row_cpicf['Projected_CPICF_Cohort_Source_Low']) and pd.notna(row_cpicf['Projected_CPICF_Cohort_Source_Mean']) and pd.notna(row_cpicf['Projected_CPICF_Cohort_Source_High'])
                                     else (f"${row_cpicf['Projected_CPICF_Cohort_Source_Mean']:,.2f} (Range N/A)" if pd.notna(row_cpicf['Projected_CPICF_Cohort_Source_Mean']) else "-"))
                                    if proj_icf_variation_percent_sidebar > 0 else (f"${row_cpicf['Projected_CPICF_Cohort_Source_Mean']:,.2f}" if pd.notna(row_cpicf['Projected_CPICF_Cohort_Source_Mean']) else "-"), axis=1)
                    cols_to_show_ai_res_list.append('Projected CPICF (Low-Mean-High)')
                elif 'Projected_CPICF_Cohort_Source_Mean' in ai_display_df_res_fmt.columns:
                     ai_display_df_res_fmt.rename(columns={'Projected_CPICF_Cohort_Source_Mean':'Projected CPICF (Mean)'}, inplace=True)
                     ai_display_df_res_fmt['Projected CPICF (Mean)'] = ai_display_df_res_fmt['Projected CPICF (Mean)'].apply(lambda x_cpicf: f"${x_cpicf:,.2f}" if pd.notna(x_cpicf) else '-')
                     cols_to_show_ai_res_list.append('Projected CPICF (Mean)')

                ai_display_df_filtered_res_fmt = ai_display_df_res_fmt[[col for col in cols_to_show_ai_res_list if col in ai_display_df_res_fmt.columns]].copy()
                for col_n_ai_res_fmt in ai_display_df_filtered_res_fmt.columns:
                    if 'Ad_Spend' in col_n_ai_res_fmt : ai_display_df_filtered_res_fmt[col_n_ai_res_fmt] = ai_display_df_filtered_res_fmt[col_n_ai_res_fmt].apply(lambda x_fmt: f"${x_fmt:,.2f}" if isinstance(x_fmt, (int,float)) and pd.notna(x_fmt) else (x_fmt if isinstance(x_fmt,str) else '-'))
                    elif 'Planned QLs (POF)' in col_n_ai_res_fmt or 'ICF_Landed' in col_n_ai_res_fmt: ai_display_df_filtered_res_fmt[col_n_ai_res_fmt] = ai_display_df_filtered_res_fmt[col_n_ai_res_fmt].apply(lambda x_fmt: f"{int(x_fmt):,}" if pd.notna(x_fmt) and isinstance(x_fmt,(int,float)) and x_fmt==x_fmt else (x_fmt if isinstance(x_fmt,str) else '-'))
                st.dataframe(ai_display_df_filtered_res_fmt.style.format(na_rep='-'))
                
                # The incomplete line is now corrected and joined with the subsequent code.
                if proj_icf_variation_percent_sidebar > 0 and 'Projected CPICF (Low-Mean-High)' in ai_display_df_filtered_res_fmt.columns:
                    st.caption(f"Note: CPICF range based on +/- {proj_icf_variation_percent_sidebar}% ICF variation (set in sidebar).")

                if 'Projected_ICF_Landed' in ai_results_df_display.columns:
                    st.subheader("Projected ICFs Landed Over Time (AI Forecast)")
                    ai_chart_data_res_val = ai_results_df_display[['Projected_ICF_Landed']].copy()
                    if isinstance(ai_chart_data_res_val.index, pd.PeriodIndex): ai_chart_data_res_val.index = ai_chart_data_res_val.index.to_timestamp()
                    ai_chart_data_res_val['Projected_ICF_Landed'] = pd.to_numeric(ai_chart_data_res_val['Projected_ICF_Landed'], errors='coerce').fillna(0)
                    st.line_chart(ai_chart_data_res_val)
                    
            elif ai_message_display and "Not Calculated" not in ai_message_display and "Missing critical base data" not in ai_message_display : 
                st.info(f"AI Forecast calculation did not produce a monthly performance table. Status: {ai_message_display}")

            if ai_site_df_display is not None and not ai_site_df_display.empty:
                st.subheader("AI Forecasted Site-Level Performance")
                ai_site_df_displayable_res_fmt = ai_site_df_display.copy()
                if ai_site_df_displayable_res_fmt.index.name != 'Site' and 'Site' in ai_site_df_displayable_res_fmt.columns: ai_site_df_displayable_res_fmt.set_index('Site', inplace=True)
                elif ai_site_df_displayable_res_fmt.index.name != 'Site' and 'Site' not in ai_site_df_displayable_res_fmt.columns and "Grand Total" in ai_site_df_displayable_res_fmt.index : ai_site_df_displayable_res_fmt.index.name = 'Site'

                formatted_site_df_ai_res_val = ai_site_df_displayable_res_fmt.copy()
                if isinstance(formatted_site_df_ai_res_val.columns, pd.MultiIndex):
                    new_cols_site_ai = []
                    for col_tuple_site_ai in formatted_site_df_ai_res_val.columns:
                        if isinstance(col_tuple_site_ai, tuple) and len(col_tuple_site_ai) == 2: new_cols_site_ai.append(f"{str(col_tuple_site_ai[1])} ({str(col_tuple_site_ai[0])})")
                        else: new_cols_site_ai.append(str(col_tuple_site_ai))
                    formatted_site_df_ai_res_val.columns = new_cols_site_ai

                for col_site_ai_name_res_fmt in formatted_site_df_ai_res_val.columns:
                    if 'Projected QLs (POF)' in col_site_ai_name_res_fmt or 'Projected ICFs Landed' in col_site_ai_name_res_fmt:
                        try: formatted_site_df_ai_res_val[col_site_ai_name_res_fmt] = formatted_site_df_ai_res_val[col_site_ai_name_res_fmt].apply(lambda x_fmt_site: f"{int(x_fmt_site):,}" if pd.notna(x_fmt_site) and isinstance(x_fmt_site, (int, float)) and x_fmt_site==x_fmt_site else ("-" if pd.isna(x_fmt_site) else str(x_fmt_site)))
                        except ValueError: formatted_site_df_ai_res_val[col_site_ai_name_res_fmt] = formatted_site_df_ai_res_val[col_site_ai_name_res_fmt].astype(str)
                st.dataframe(formatted_site_df_ai_res_val.style.format(na_rep='-'))
            elif "Missing critical base data" not in ai_message_display and "Not Calculated" not in ai_message_display :
                st.info("Site-level AI forecast not available or sites not defined/active.")
        else:
            st.caption("Click the button above to generate the AI forecast based on your goals.")

elif not uploaded_referral_file or not uploaded_funnel_def_file:
    st.info("👋 Welcome! Please upload both the Referral Data (CSV) and Funnel Definition (TSV) files using the sidebar to begin.")
               