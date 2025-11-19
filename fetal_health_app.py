# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import altair as alt
import os
import difflib

# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = '../model/fetal_model.pkl'  # change path if your model is elsewhere
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Update MODEL_PATH variable.")
    st.stop()

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Feature order expected by the model (must match training)
FEATURES = [
    'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions',
    'light_decelerations', 'severe_decelerations', 'prolonged_decelerations',
    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability',
    'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
    'histogram_variance', 'histogram_tendency'
]

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fetal Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS (styling) using theme color #8FF1E9
# -----------------------------
st.markdown(
    """
    <style>
    /* Theme primary color: #8FF1E9 (light aqua) */
    :root {
        --theme-aqua: #8FF1E9;
        --theme-aqua-dark: #5FD7CC;
        --theme-aqua-strong: #62c7bd;
        --card-bg: rgba(255,255,255,0.88);
        --muted-text: #334155;
    }

    /* App background: subtle gradient including theme color */
    .stApp {
        background: linear-gradient(120deg, var(--theme-aqua) 0%, #d7fcf9 40%, #ffffff 100%);
        color: #0f1724;
    }

    /* Card style for main content */
    .card {
        background: var(--card-bg);
        border-left: 6px solid var(--theme-aqua);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 20px rgba(20, 40, 80, 0.06);
        margin-bottom: 16px;
    }

    /* Metric tiles */
    .metric-tile {
        border-radius: 10px;
        padding: 12px;
        color: white;
        text-align: center;
        font-weight: 600;
    }

    .metric-green { background: linear-gradient(90deg,#16a34a,#43c85a); }
    .metric-yellow { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
    .metric-red { background: linear-gradient(90deg,#ef4444,#fb7185); }
    /* Aqua themed metric */
    .metric-blue { background: linear-gradient(90deg,var(--theme-aqua),var(--theme-aqua-dark)); }

    /* Sidebar header area (class name may vary by Streamlit version) */
    .css-1d391kg, .stSidebar > div[role="contentinfo"] {
        background: linear-gradient(180deg, rgba(143,241,233,0.35), rgba(143,241,233,0.08));
    }

    /* Button style using theme color */
    .stButton>button {
        background-color: var(--theme-aqua);
        color: #003235;
        border-radius: 8px;
        padding: 8px 12px;
        border: 1px solid var(--theme-aqua-strong);
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: var(--theme-aqua-dark);
    }

    /* Inputs focus/outline accent (attempt; Streamlit classes vary) */
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus, textarea:focus {
        outline: 2px solid rgba(143,241,233,0.5) !important;
        box-shadow: 0 0 0 3px rgba(143,241,233,0.12) !important;
    }

    /* Make main blocks sit above background nicely */
    .block-container, .main {
        position: relative;
        z-index: 1;
    }

    /* Small responsive tweaks */
    @media (max-width: 850px) {
        .metric-tile { font-size: 14px; padding: 10px; }
    }

    /* Info box accent */
    .stAlert {
        border-left: 4px solid var(--theme-aqua);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="card">
        <h1 style="margin:0">🩺 Fetal Health Prediction</h1>
        <p style="color:var(--muted-text);margin-top:6px">
            Predict fetal health condition from CTG parameters. Upload a CSV for batch predictions or provide manual inputs in the sidebar, then click <b>Predict Fetal Health</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar Inputs (CSV uploader moved to top)
# -----------------------------
with st.sidebar:
    st.header("Upload CSV for Batch Prediction")
    csv_file = st.file_uploader("Upload CSV", type=['csv'])
    st.markdown("---")
    st.header("Manual Patient CTG Input")
    st.markdown("<div style='padding:6px 0 4px 0;color:var(--muted-text)'>General CTG</div>", unsafe_allow_html=True)
    baseline_value = st.number_input("Baseline Value (bpm)", value=120.0, step=0.1, format="%.2f")
    accelerations = st.number_input("Accelerations", value=0.02, step=0.001, format="%.5f")
    fetal_movement = st.number_input("Fetal Movement", value=0.0, step=0.001, format="%.5f")
    uterine_contractions = st.number_input("Uterine Contractions", value=0.0, step=0.001, format="%.5f")

    st.markdown("<div style='padding:6px 0 4px 0;color:var(--muted-text)'>Decelerations</div>", unsafe_allow_html=True)
    light_decelerations = st.number_input("Light Decelerations", value=0.0, step=0.001, format="%.5f")
    severe_decelerations = st.number_input("Severe Decelerations", value=0.0, step=0.001, format="%.5f")
    prolonged_decelerations = st.number_input("Prolonged Decelerations", value=0.0, step=0.001, format="%.5f")

    st.markdown("<div style='padding:6px 0 4px 0;color:var(--muted-text)'>Variability</div>", unsafe_allow_html=True)
    abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", value=10.0, step=0.1, format="%.2f")
    mean_value_of_short_term_variability = st.number_input("Mean Short Term Variability", value=2.0, step=0.01, format="%.3f")
    percentage_of_time_with_abnormal_long_term_variability = st.number_input("Percent time abnormal LTV", value=20.0, step=0.1, format="%.2f")
    mean_value_of_long_term_variability = st.number_input("Mean Long Term Variability", value=3.0, step=0.01, format="%.3f")

    st.markdown("<div style='padding:6px 0 4px 0;color:var(--muted-text)'>Histogram Features</div>", unsafe_allow_html=True)
    histogram_width = st.number_input("Histogram Width", value=60.0, step=0.1, format="%.2f")
    histogram_min = st.number_input("Histogram Min", value=50.0, step=0.1, format="%.2f")
    histogram_max = st.number_input("Histogram Max", value=150.0, step=0.1, format="%.2f")
    histogram_number_of_peaks = st.number_input("Histogram Number of Peaks", value=5.0, step=0.1, format="%.2f")
    histogram_number_of_zeroes = st.number_input("Histogram Number of Zeroes", value=2.0, step=0.1, format="%.2f")
    histogram_mode = st.number_input("Histogram Mode", value=120.0, step=0.1, format="%.2f")
    histogram_mean = st.number_input("Histogram Mean", value=130.0, step=0.1, format="%.2f")
    histogram_median = st.number_input("Histogram Median", value=125.0, step=0.1, format="%.2f")
    histogram_variance = st.number_input("Histogram Variance", value=15.0, step=0.1, format="%.2f")
    histogram_tendency = st.number_input("Histogram Tendency", value=0.0, step=0.01, format="%.3f")

# -----------------------------
# CSV batch prediction flow (moved to top of main area so results show first)
# -----------------------------
def normalize_name(s):
    if pd.isna(s):
        return ''
    return str(s).strip().lower().replace(' ', '_')

if 'csv_file' in locals() and csv_file is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("CSV Batch Prediction")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None

    if df is not None:
        st.markdown("### Uploaded CSV")
        st.dataframe(df.head())

        # build normalized mapping of uploaded columns
        orig_cols = list(df.columns)
        norm_to_orig = {normalize_name(c): c for c in orig_cols}
        norm_cols = set(norm_to_orig.keys())

        # aliases for common typos
        aliases = {
            'prolongued_decelerations': 'prolonged_decelerations',
            'prolonged_decelarations': 'prolonged_decelerations',
            # add additional aliases here if needed
        }

        norm_features = [normalize_name(f) for f in FEATURES]

        # Try to auto-map features -> original column names
        auto_map = {}
        unmatched = []
        for nf, feat in zip(norm_features, FEATURES):
            # exact normalized match
            if nf in norm_cols:
                auto_map[feat] = norm_to_orig[nf]
                continue
            # alias match (uploaded has alias)
            if nf in aliases and aliases[nf] in norm_cols:
                auto_map[feat] = norm_to_orig[aliases[nf]]
                continue
            # uploaded column might be an alias (typo)
            for a_k, a_v in aliases.items():
                if a_k in norm_cols and a_v == nf:
                    auto_map[feat] = norm_to_orig[a_k]
                    break
            if feat in auto_map:
                continue
            # fuzzy match against normalized column names
            matches = difflib.get_close_matches(nf, list(norm_cols), n=1, cutoff=0.75)
            if matches:
                auto_map[feat] = norm_to_orig[matches[0]]
            else:
                unmatched.append((feat, nf))

        # If everything auto-mapped, build X
        if len(auto_map) == len(FEATURES):
            st.success("Detected required feature columns.")
            try:
                X = df[[auto_map[f] for f in FEATURES]].astype(float).values
            except Exception as e:
                st.error(f"Failed to convert mapped columns to numeric arrays: {e}")
                X = None
        else:
            # show what matched and what didn't
            st.warning("CSV columns don't exactly match expected feature names. The app attempted to auto-map the following (you can correct below):")
            mapped_table = {f: auto_map.get(f, '---') for f in FEATURES}
            st.write(pd.DataFrame.from_dict(mapped_table, orient='index', columns=['mapped_csv_column']).rename_axis('expected_feature'))
            st.markdown("If the mapping is wrong, please correct using the manual mapping form below, or rename your CSV headers. Known typo fixes and fuzzy matching were attempted.")

            # Manual mapping UI: for each expected feature, ask user to choose a column or enter a constant value
            mapping = {}
            cols = list(df.columns)
            with st.form(key='mapping_form'):
                st.markdown("#### Map CSV columns to model features")
                for feat in FEATURES:
                    pre = auto_map.get(feat, '--constant--')
                    # compute index for preselection if pre is in cols
                    if pre == '--constant--':
                        index = 0
                    else:
                        try:
                            index = cols.index(pre) + 1
                        except ValueError:
                            index = 0
                    choice = st.selectbox(f"Column for '{feat}' (or choose 'constant')", options=['--constant--'] + cols, index=index, key=f"map_{feat}")
                    if choice == '--constant--':
                        val = st.text_input(f"Constant numeric value for {feat}", value="0", key=f"const_{feat}")
                        # safe float conversion
                        try:
                            valf = float(val) if str(val).strip() != '' else 0.0
                        except Exception:
                            valf = 0.0
                        mapping[feat] = {'type': 'const', 'value': valf}
                    else:
                        mapping[feat] = {'type': 'col', 'value': choice}
                submit_map = st.form_submit_button("Apply mapping and predict (CSV)")

            if submit_map:
                built_cols = []
                for feat in FEATURES:
                    m = mapping[feat]
                    if m['type'] == 'col':
                        try:
                            built_cols.append(df[m['value']].astype(float).values)
                        except Exception as e:
                            st.error(f"Failed to convert column {m['value']} to numeric for feature {feat}: {e}")
                            built_cols.append(np.repeat(0.0, len(df)))
                    else:
                        built_cols.append(np.repeat(float(m['value']), len(df)))
                X = np.vstack(built_cols).T
            else:
                X = None

        # If X prepared (auto-mapped or after mapping), run predictions
        if 'X' in locals() and X is not None:
            try:
                preds = model.predict(X)
                preds = np.array(preds, dtype=int)
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
                preds = None

            if preds is not None:
                # optionally get probabilities
                probs = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probs = model.predict_proba(X)
                    except Exception:
                        probs = None

                # Create results dataframe
                res_df = df.copy()
                res_df['predicted_label'] = preds
                res_df['predicted_class'] = res_df['predicted_label'].map({0: 'Normal', 1: 'Suspect', 2: 'Pathological'})

                if probs is not None:
                    # Attach probabilities for each class (attempt to name columns 0,1,2)
                    for i in range(probs.shape[1]):
                        res_df[f'prob_class_{i}'] = probs[:, i]

                st.markdown("### Prediction results")
                st.dataframe(res_df.head(100))

                # Download button for full results
                csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions as CSV", data=csv_bytes, file_name='fetal_predictions.csv', mime='text/csv')

                # Show aggregate counts
                st.markdown("#### Summary")
                st.write(res_df['predicted_class'].value_counts())

                # Show a small chart of the distribution
                chart_df = res_df['predicted_class'].value_counts().reset_index()
                chart_df.columns = ['predicted_class', 'count']
                bar = alt.Chart(chart_df).mark_bar().encode(
                    x='predicted_class:N',
                    y='count:Q'
                ).properties(height=250)
                st.altair_chart(bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Main: Predict Button & Result (manual single-record) — remains below CSV area
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Manual Prediction")

# Manual prediction (single record)
if st.button("Predict Fetal Health"):
    features = np.array([[baseline_value, accelerations, fetal_movement,
                          uterine_contractions, light_decelerations,
                          severe_decelerations, prolonged_decelerations,
                          abnormal_short_term_variability, mean_value_of_short_term_variability,
                          percentage_of_time_with_abnormal_long_term_variability,
                          mean_value_of_long_term_variability,
                          histogram_width, histogram_min, histogram_max,
                          histogram_number_of_peaks, histogram_number_of_zeroes,
                          histogram_mode, histogram_mean, histogram_median,
                          histogram_variance, histogram_tendency]])
    try:
        prediction = int(model.predict(features)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    label, emoji, tile_class = {0: ("Normal", "🟢", "metric-green"),
                               1: ("Suspect", "🟡", "metric-yellow"),
                               2: ("Pathological", "🔴", "metric-red")}.get(prediction, ("Unknown", "⚙️", "metric-blue"))

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px">
            <div style="font-size:20px;padding:10px 14px;border-radius:10px;background:rgba(15,23,36,0.04);border-left:5px solid var(--theme-aqua)">
                <strong>Result</strong>
            </div>
            <div style="font-size:20px;padding:10px 14px;border-radius:10px;background:#fff;border-left:5px solid var(--theme-aqua)">
                <span style="font-size:28px">{emoji}</span>
                &nbsp;&nbsp;<strong style="font-size:18px">Predicted: {label}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-tile metric-blue'><strong>Baseline FHR</strong><div style='font-size:20px'>{baseline_value} bpm</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-tile metric-green'><strong>Accelerations</strong><div style='font-size:20px'>{accelerations}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-tile metric-yellow'><strong>Short Term Var</strong><div style='font-size:20px'>{mean_value_of_short_term_variability}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-tile metric-red'><strong>Long Term Var</strong><div style='font-size:20px'>{mean_value_of_long_term_variability}</div></div>", unsafe_allow_html=True)

    # Charts: example
    with st.expander("View Example Charts"):
        time = list(range(60))
        heart_rate = [baseline_value + float(np.random.normal(0, 3)) for _ in range(60)]
        df_hr = pd.DataFrame({"Time (s)": time, "FHR (bpm)": heart_rate})
        chart = alt.Chart(df_hr).mark_line(point=True).encode(
            x=alt.X("Time (s):Q"),
            y=alt.Y("FHR (bpm):Q")
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

        hist_values = np.random.normal(histogram_mean, np.sqrt(max(0.0001, histogram_variance)), int(max(1, histogram_width)))
        df_hist = pd.DataFrame({"FHR": hist_values})
        hist_chart = alt.Chart(df_hist).mark_bar().encode(
            x=alt.X("FHR:Q", bin=alt.Bin(maxbins=25)),
            y='count()'
        ).properties(height=300)
        st.altair_chart(hist_chart, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Information Panels
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Fetal Health Classification Guide")
st.info("🟢 **Normal:** FHR and variability within normal range; minimal interventions required.")
st.warning("🟡 **Suspect:** Slight abnormalities detected; closer monitoring recommended.")
st.error("🔴 **Pathological:** Significant abnormalities detected; clinical intervention required.")
st.markdown("</div>", unsafe_allow_html=True)
