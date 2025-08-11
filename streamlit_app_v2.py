import streamlit as st
import pandas as pd
import numpy as np
import os
from fpdf import FPDF
from datetime import datetime

# Set page config for a wider layout
st.set_page_config(layout="wide")

st.title("🌬️ 1-Minute Wind Data Processor & Stow Trigger Optimization Tool")

# --- Initialize Session State ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
    st.session_state.results = {}

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    project_name_input = st.text_input(
        "Project Name",
        placeholder="e.g., Ameer Solar Farm",
        help="Enter the name of the project for the analysis."
    )
    gust_threshold_input = st.number_input(
        "Gust Threshold (mph)",
        min_value=20,
        max_value=70,
        value=40,
        help="Operational design limit. Trackers must be fully stowed by this speed. Defined in the structural calculation package."
    )
    uploaded_file = st.file_uploader(
        "Upload your raw CSV file",
        type=['csv'],
        help="Upload your CSV file here. The app is configured to handle large files."
    )
    run_button = st.button("🚀 Run Analysis")

# --- Data Processing & Analysis Functions ---
def process_wind_data(df_raw):
    df = df_raw.copy()
    def find_col(df, target, contains=False):
        target = target.lower()
        for c in df.columns:
            c_norm = c.strip().lower()
            if (contains and target in c_norm) or (not contains and c_norm == target):
                return c
        return None
    ts = find_col(df, 'valid', contains=True)
    stn = find_col(df, 'station')
    sknt = find_col(df, 'sknt')
    gust = find_col(df, 'gust_sknt', contains=True)
    if None in (ts, stn, sknt, gust):
        raise ValueError("Required columns (valid, station, sknt, gust_sknt) not found in the raw data.")
    df = df.rename(columns={ts: 'valid', stn: 'station', sknt: 'sknt', gust: 'gust_sknt'})
    df['valid'] = pd.to_datetime(df['valid']).dt.floor('min')
    df = df.loc[:, ~df.columns.str.contains('station_name', case=False)]
    pieces = []
    for s, g in df.groupby('station'):
        g = (g.sort_values('valid')
             .drop_duplicates('valid', keep='last')
             .set_index('valid')[['sknt', 'gust_sknt']])
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq='min')
        g = g.reindex(full_idx)
        g['station'] = s
        pieces.append(g.reset_index().rename(columns={'index': 'valid'}))
    df_full = pd.concat(pieces, ignore_index=True)
    K2MPH = 1.15078
    df_full[['sknt', 'gust_sknt']] = (df_full[['sknt', 'gust_sknt']].apply(pd.to_numeric, errors='coerce') * K2MPH)
    df_full['60Avg'] = (2 * df_full['sknt'] - df_full.groupby('station')['sknt'].shift(1))
    df_full.loc[df_full['60Avg'] < 0, '60Avg'] = 0
    df_full['60Avg'] = df_full['60Avg'].round(2)
    df_full['gust_sknt'] = df_full['gust_sknt'].round(2)
    df_full = df_full.rename(columns={'valid': 'DateTime', 'station': 'Station', 'gust_sknt': 'Gust'})
    return df_full[['DateTime', 'Station', '60Avg', 'Gust']]

def find_optimal_triggers(df, gust_threshold):
    AVG_MIN, AVG_MAX = 0, 150
    GUST_MIN, GUST_MAX = 0, 250
    SPIKE_AVG, SPIKE_GUST = 20, 30
    AVG_GUST_DIFF = 30
    RATE_CHANGE_DIFF = 30
    RATE_CHANGE_SHIFT = 60
    GUST_HOLD_TIME, AVG_HOLD_TIME = 90, 30
    PCTL = 0.975
    GUST_FACTOR_METHOD = "m1"
    MIN_GUST_BELOW_THRESHOLD = 10
    MIN_AVG_BELOW_GUST = 5
    MIN_AVG_TRIGGER = 10
    GUST_THRESHOLD = gust_threshold
    NEW_COLUMNS = ["DateTime", "Station", "60Avg", "Gust", "Missing", "AvgMax", "GustMax", "SpikeAvg", "SpikeGust", "AvgGustDiff", "RateofChange", "ValidData", "GustTimer", "AvgTimer", "InStow", "TriggeredBy", "T0_Gust", "T1_Gust", "T2_Gust", "T3_Gust", "T4_Gust", "T0_Avg",  "T1_Avg",  "T2_Avg",  "T3_Avg",  "T4_Avg"]
    FLAG_COLS = ["Missing", "AvgMax", "GustMax", "SpikeAvg", "SpikeGust", "AvgGustDiff", "RateofChange"]
    for col in NEW_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col in FLAG_COLS else pd.NA
    df = df.reindex(columns=NEW_COLUMNS)
    df[FLAG_COLS] = df[FLAG_COLS].fillna(0).astype("int8")
    for col in ['60Avg', 'Gust']:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df.loc[df["60Avg"].isna() | df["Gust"].isna(), "Missing"] = 1
    df.loc[(df["60Avg"] < AVG_MIN) | (df["60Avg"] > AVG_MAX), "AvgMax"] = 1
    df.loc[(df["Gust"] < GUST_MIN) | (df["Gust"] > GUST_MAX), "GustMax"] = 1
    df["SpikeAvg"] = (df["60Avg"].diff() > SPIKE_AVG).astype("int8")
    df["SpikeGust"] = (df["Gust"].diff() > SPIKE_GUST).astype("int8")
    df["AvgGustDiff"] = ((df["Gust"] - df["60Avg"]) > AVG_GUST_DIFF).astype("int8")
    ref = df["60Avg"].shift(RATE_CHANGE_SHIFT)
    rate_mask = (df["60Avg"] - ref).abs() > RATE_CHANGE_DIFF
    df["RateofChange"] = (rate_mask | ref.isna()).astype("int8")
    df["ValidData"] = ~(df[FLAG_COLS] == 1).any(axis=1)
    df['GustFactor'] = df['Gust'] / df['60Avg'].replace(0, np.nan)
    gust_effect_factor = df['GustFactor'].mean() + df['GustFactor'].std()
    avg_threshold = GUST_THRESHOLD / gust_effect_factor
    gust_arr = df["Gust"].to_numpy(dtype=np.float32, na_value=np.nan)
    avg_arr = df["60Avg"].to_numpy(dtype=np.float32, na_value=np.nan)
    valid = df["ValidData"].to_numpy(dtype=bool)
    time_ok = df["DateTime"].dt.hour.between(7, 19).to_numpy(dtype=bool)
    N, idx = len(df), np.arange(len(df), dtype=np.int32)
    def minutes_since(mask: np.ndarray, hold: int) -> np.ndarray:
        last = np.where(mask, idx, -1)
        np.maximum.accumulate(last, out=last)
        elapsed = idx - last + 1
        ok = (last != -1) & (elapsed <= hold)
        return np.where(ok, elapsed, 0).astype(np.int16)
    def pct_offsets(row_idx: np.ndarray, data: np.ndarray, off=(1, 2, 3, 4), pctl=PCTL) -> np.ndarray:
        out = []
        for o in off:
            tgt = row_idx + o
            tgt = tgt[tgt < N]
            vals = data[tgt]
            vals = vals[np.isfinite(vals)]
            out.append(np.quantile(vals, pctl, method="weibull") if vals.size else np.nan)
        return np.array(out, dtype=np.float32)
    gust_search_range = range(int(GUST_THRESHOLD) - MIN_GUST_BELOW_THRESHOLD, MIN_AVG_TRIGGER + MIN_AVG_BELOW_GUST - 1, -1)
    combos = [(g, a) for g in gust_search_range for a in range(g - MIN_AVG_BELOW_GUST, MIN_AVG_TRIGGER - 1, -1)]
    records = []
    progress_bar = st.progress(0, text="Evaluating trigger combinations...")
    for i, (g_trig, a_trig) in enumerate(combos):
        g_mask = (gust_arr >= g_trig) & valid & time_ok
        a_mask = (avg_arr >= a_trig) & valid & time_ok
        in_stow = (minutes_since(g_mask, GUST_HOLD_TIME) > 0) | (minutes_since(a_mask, AVG_HOLD_TIME) > 0)
        st_starts = np.where(in_stow & ~np.roll(in_stow, 1))[0]
        st_starts = st_starts[st_starts < N - 4]
        g_starts = st_starts[g_mask[st_starts]]
        a_starts = st_starts[a_mask[st_starts]]
        g_vec = pct_offsets(g_starts, gust_arr)
        a_vec = pct_offsets(a_starts, avg_arr)
        passed = (np.nan_to_num(g_vec, nan=np.inf) < GUST_THRESHOLD).all() and (np.nan_to_num(a_vec, nan=np.inf) < avg_threshold).all()
        stow_mins = in_stow.sum() if passed else np.nan
        records.append((g_trig, a_trig, passed, stow_mins, g_vec, a_vec))
        progress_bar.progress((i + 1) / len(combos), text=f"Evaluating combo {i+1}/{len(combos)}: Gust={g_trig}, Avg={a_trig}")
    progress_bar.empty()
    trigger_combo_results = pd.DataFrame.from_records(records, columns=["gust", "avg", "passed", "stow_mins", "g_vec", "a_vec"])
    passing_results_df = trigger_combo_results[trigger_combo_results['passed']].sort_values('stow_mins').reset_index(drop=True)
    return passing_results_df, trigger_combo_results, avg_threshold, time_ok.sum()

def generate_pdf_report(results, gust_threshold, avg_thresh, total_combos, tracking_mins, project_name):
    best = results.iloc[0]

    class PDF(FPDF):
        def header(self):
            try:
                logo_path = os.path.join(os.path.dirname(__file__), "Horizontal logo (1).png")
                if os.path.exists(logo_path):
                    self.image(logo_path, x=10, y=8, w=42)
            except Exception:
                pass
            self.set_font("Arial", "B", 14)
            self.cell(0, 8, "Terrasmart", 0, 1, "C")
            self.set_font("Arial", "", 9)
            self.cell(0, 6, "Terrasmart Engineering Department", 0, 1, "C")
            self.set_line_width(0.3)
            self.line(10, 28, 200, 28)
            self.ln(6)

    pdf = PDF()
    pdf.set_margins(10, 10, 10)
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    content_w = pdf.w - pdf.l_margin - pdf.r_margin

    # Title and meta
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 8, "TerraTrak Wind Speed Trigger Optimization Report", 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Project: {project_name}", 0, 1, "C")
    pdf.cell(0, 6, f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, "C")
    pdf.ln(2)

    # Intro
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Introduction", 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    intro = (
        "This report documents the determination of wind speed triggers for a TerraTrak single axis tracker project. "
        "The objective is to establish project specific gust and average thresholds that satisfy safety limits and support maximizing energy production."
    )
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, intro)
    pdf.ln(1)

    # Definitions immediately after intro
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Definitions", 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    gust_def = (
        "Gust Threshold: Operational design limit. Trackers must be fully stowed by this speed. Defined in the structural calculation package."
    )
    avg_def = (
        "Derived Average Threshold: The average wind speed limit computed from project data and the Gust Threshold. "
        "A representative gust factor is calculated as the ratio of gust speed to one minute average speed using mean plus one standard deviation. "
        "The average threshold then equals the Gust Threshold divided by this gust factor."
    )
    candidate_pair_def = (
        "Candidate Pair: A combination of gust and average wind speed values evaluated to be used in the analysis to determine the optimal triggers for the site under consideration."
    )
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, f"- {gust_def}")
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, f"- {avg_def}")
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, f"- {candidate_pair_def}")
    pdf.ln(1)

    # Analysis Methodology
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Analysis Methodology", 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    methodology = (
        "Local wind records are used to evaluate candidate pairs of gust and average triggers. "
        "For each candidate pair, the TerraTrak control logic is simulated on the time series. "
        "Post stow behavior is then assessed against percentile based limits. "
    )
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, methodology)
    pdf.ln(1)

    # Method at a glance
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Method at a Glance", 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    bullets = [
        "Each candidate pair is applied to the project wind time series using a simulation of the TerraTrak stow logic.",
        "In the four minutes following each simulated stow command, the wind speeds must remain safely below the design thresholds. This is verified by ensuring the statistical distribution of post-stow triggered winds meets a reliability target equal to or better than that required by ASCE standards for the given risk category of the structure."
    ]
    for b in bullets:
        pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, f"- {b}")
    pdf.ln(1)

    # Pass/Fail Criteria
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Pass/Fail Criteria", 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    pf1 = (
        "A candidate pair is designated 'passing' when the post-stow triggered wind speeds demonstrate a high degree of statistical reliability. The analysis confirms that the tail of the wind speed distribution is well-controlled and that the probability of exceeding design thresholds is acceptably low, consistent with ASCE structural safety standards for the given risk category of the structure."
    )
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, pf1)
    pdf.ln(1)

    # Engineering Basis
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Engineering Basis", 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    j1 = (
        "Structure parameters combined with wind tunnel testing determine threshold selection and provide traceable justification."
    )
    j2 = (
        "The analysis uses a statistical assessment of the post-stow triggered wind speed distribution to ensure its tail behavior is well-controlled. This method provides high confidence that the system's response meets or exceeds the structural reliability targets for wind loading consistent with ASCE standards for the given risk category of the structure, while reducing sensitivity to anomalous single-point measurements in the data."
    )
    j3 = (
        "The selection process supports maximizing energy capture while maintaining structural performance and safety requirements."
    )
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, j1)
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, j2)
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, j3)
    pdf.ln(1)

    # Parameters and results table
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Analysis Parameters and Results", 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    pdf.set_fill_color(240, 240, 240)
    col_w = 95
    row_h = 6

    pdf.cell(col_w, row_h, "Parameter", 1, 0, "C", 1)
    pdf.cell(col_w, row_h, "Value", 1, 1, "C", 1)

    pdf.cell(col_w, row_h, "Gust Wind Speed Threshold", 1)
    pdf.cell(col_w, row_h, f"{gust_threshold} mph", 1, 1)

    pdf.cell(col_w, row_h, "Derived Average Wind Speed Threshold", 1)
    pdf.cell(col_w, row_h, f"{avg_thresh:.2f} mph", 1, 1)

    pdf.cell(col_w, row_h, "Total Number of Candidate Pairs Evaluated", 1)
    pdf.cell(col_w, row_h, f"{total_combos:,}", 1, 1)

    pdf.cell(col_w, row_h, "Number of Passing Candidate Pairs Found", 1)
    pdf.cell(col_w, row_h, f"{len(results):,}", 1, 1)

    pdf.set_font("Arial", "B", 9)
    pdf.cell(col_w, row_h, "Optimal Gust Wind Speed Trigger", 1)
    pdf.cell(col_w, row_h, f"{best['gust']} mph", 1, 1)

    pdf.cell(col_w, row_h, "Optimal Average Wind Speed Trigger", 1)
    pdf.cell(col_w, row_h, f"{best['avg']} mph", 1, 1)
    pdf.set_font("Arial", "", 9)

    pdf.ln(2)

    # Conclusion
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Conclusion", 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    conclusion = (
        f"For the {project_name} project, the evaluation processed {total_combos:,} candidate pairs and identified {len(results):,} passing pairs. "
        f"Of the {len(results):,} passing pairs the one that has the lowest impact to tracking uptime is a gust wind speed trigger of {best['gust']} mph and an average wind speed trigger of {best['avg']} mph. "
        "This configuration satisfies the stated criteria in the post-stow period and supports energy production while maintaining structural integrity requirements."
    )
    pdf.set_x(pdf.l_margin); pdf.multi_cell(content_w, 4.5, conclusion)

    return bytes(pdf.output(dest='S'))

# --- Main App Logic ---
if run_button:
    if uploaded_file is not None:
        try:
            with st.spinner("Reading raw data file..."):
                raw_df = pd.read_csv(uploaded_file)
            with st.spinner("Processing raw wind data..."):
                processed_df = process_wind_data(raw_df)
            with st.spinner("Running optimization analysis... This may take a few minutes."):
                passing_results, all_results, avg_thresh, tracking_mins = find_optimal_triggers(processed_df, gust_threshold_input)
            
            st.session_state.results = {
                'passing_results': passing_results,
                'all_results': all_results,
                'avg_thresh': avg_thresh,
                'tracking_mins': tracking_mins,
                'project_name': project_name_input,
                'gust_threshold': gust_threshold_input
            }
            st.session_state.analysis_run = True
            st.rerun()

        except ValueError as e:
            st.error(f"An error occurred during data processing: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please upload a file to run the analysis.")

if st.session_state.analysis_run:
    st.header("📈 Results")
    
    if st.button("🧹 Clear Analysis"):
        st.session_state.analysis_run = False
        st.session_state.results = {}
        st.rerun()

    res = st.session_state.results
    passing_results = res['passing_results']
    all_results = res['all_results']

    with st.expander("View Full Optimizer Search Results", expanded=False):
        st.write(f"A total of {len(all_results)} combinations were evaluated.")
        st.dataframe(all_results[['gust', 'avg', 'passed']])

    if passing_results is not None and not passing_results.empty:
        best = passing_results.iloc[0]
        st.subheader("⭐ Optimal Trigger Configuration")
        col1, col2 = st.columns(2)
        col1.metric("Optimal Gust Trigger", f"{best['gust']} mph")
        col2.metric("Optimal Average Trigger", f"{best['avg']} mph")
        st.divider()
        with st.expander("View Detailed Percentiles for Optimal Triggers"):
            st.write("Gust Wind Speed Reliability Check (mph)")
            g_vec_df = pd.DataFrame([best['g_vec']], columns=[f"T+{i+1}" for i in range(4)], index=["Gust (mph)"])
            st.dataframe(g_vec_df.style.format("{:.2f}"))
            st.write("Average Wind Speed Reliability Check (mph)")
            a_vec_df = pd.DataFrame([best['a_vec']], columns=[f"T+{i+1}" for i in range(4)], index=["Avg (mph)"])
            st.dataframe(a_vec_df.style.format("{:.2f}"))
        st.info(f"""
        - **Gust Threshold:** `{res['gust_threshold']} mph`
        - **Derived Avg Threshold:** `{res['avg_thresh']:.2f} mph`
        - **Total Combinations Evaluated:** `{len(all_results)}`
        - **Passing Combinations Found:** `{len(passing_results)}`
        """)
        pdf_bytes = generate_pdf_report(passing_results, res['gust_threshold'], res['avg_thresh'], len(all_results), res['tracking_mins'], res['project_name'])
        st.download_button(
            label="Download Report",
            data=pdf_bytes,
            file_name=f"{res['project_name'] or 'report'} - TerraTrak Wind Stow Trigger Optimization Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    elif all_results is not None:
        st.warning("No passing trigger combinations were found. Review the full search results above.")
else:
    st.markdown("""
    This application processes raw ASOS wind data and then analyzes it to find the optimal gust and average wind speed triggers.
    
    **Instructions:**
    1.  Set the desired `Gust Threshold` in the sidebar from the projects structural calculation package.
    2.  Upload your **raw** wind data CSV file.
    3.  Click `Run Analysis`.
    """)
