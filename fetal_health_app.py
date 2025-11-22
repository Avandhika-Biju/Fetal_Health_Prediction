# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import altair as alt
import os
import difflib

# -----------------------------
# Example uploaded file path from conversation history (tooling may transform this to a URL)
UPLOADED_FILE_PATH = "sandbox:/mnt/data/4f0aa5d0-0d7b-49fe-9286-faab01e6a18a.png"
# -----------------------------

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
# Custom CSS (styling)
# -----------------------------
st.markdown(
    """
    <style>
    :root { --theme-aqua: #8FF1E9; --theme-aqua-dark: #5FD7CC; }
    .stApp { background: linear-gradient(120deg, var(--theme-aqua) 0%, #d7fcf9 40%, #ffffff 100%); color: #0f1724; }
    .card { background: rgba(255,255,255,0.88); border-left:6px solid var(--theme-aqua); border-radius:12px; padding:18px; margin-bottom:16px; }
    .metric-tile { border-radius:10px; padding:12px; color:white; text-align:center; font-weight:600; }
    .metric-green { background: linear-gradient(90deg,#16a34a,#43c85a); }
    .metric-yellow { background: linear-gradient(90deg,#f59e0b,#fbbf24); color:#0f1724; }
    .metric-red { background: linear-gradient(90deg,#ef4444,#fb7185); }
    .metric-blue { background: linear-gradient(90deg,var(--theme-aqua),var(--theme-aqua-dark)); color:#003235; }

    /* HTML table style used for preview/results/summary */
    .fetal-table { border-collapse:collapse; width:100%; font-family: "Helvetica", Arial, sans-serif; }
    .fetal-table th, .fetal-table td { border:1px solid rgba(15,23,36,0.06); padding:8px 10px; text-align:left; vertical-align:middle; }
    .fetal-table th { background: rgba(0,0,0,0.03); font-weight:700; }
    .badge-cell { white-space:nowrap; }

    /* Scrollable wrapper used for CSV preview (gives both vertical & horizontal scrollbars) */
    .table-wrapper {
        width: 100%;
        max-height: 380px;  /* adjust vertical viewport height */
        overflow: auto;     /* enable both horizontal + vertical scrolling */
        border-radius: 8px;
        padding: 8px;
        background: rgba(255,255,255,0.95);
        box-shadow: 0 6px 18px rgba(20,40,80,0.04);
    }

    /* make the inner table wide so horizontal scrollbar appears when necessary */
    .fetal-table.minwide { min-width: 900px; }

    @media (max-width: 900px) {
        .fetal-table.minwide { min-width: 700px; }
        .table-wrapper { max-height: 260px; }
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
        <p style="color:#334155;margin-top:6px">Predict fetal health condition from CTG parameters. Upload a CSV for batch predictions or provide manual inputs in the sidebar, then click <b>Predict Fetal Health</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
with st.sidebar:
    st.header("Upload CSV for Batch Prediction")
    csv_file = st.file_uploader("Upload CSV", type=['csv'])
    st.markdown("---")
    st.header("Manual Patient CTG Input")
    st.markdown("<div style='padding:6px 0 4px 0;color:#334155'>General CTG</div>", unsafe_allow_html=True)
    baseline_value = st.number_input("Baseline Value (bpm)", value=120.0, step=0.1, format="%.2f")
    accelerations = st.number_input("Accelerations", value=0.02, step=0.001, format="%.5f")
    fetal_movement = st.number_input("Fetal Movement", value=0.0, step=0.001, format="%.5f")
    uterine_contractions = st.number_input("Uterine Contractions", value=0.0, step=0.001, format="%.5f")

    st.markdown("<div style='padding:6px 0 4px 0;color:#334155'>Decelerations</div>", unsafe_allow_html=True)
    light_decelerations = st.number_input("Light Decelerations", value=0.0, step=0.001, format="%.5f")
    severe_decelerations = st.number_input("Severe Decelerations", value=0.0, step=0.001, format="%.5f")
    prolonged_decelerations = st.number_input("Prolonged Decelerations", value=0.0, step=0.001, format="%.5f")

    st.markdown("<div style='padding:6px 0 4px 0;color:#334155'>Variability</div>", unsafe_allow_html=True)
    abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", value=10.0, step=0.1, format="%.2f")
    mean_value_of_short_term_variability = st.number_input("Mean Short Term Variability", value=2.0, step=0.01, format="%.3f")
    percentage_of_time_with_abnormal_long_term_variability = st.number_input("Percent time abnormal LTV", value=20.0, step=0.1, format="%.2f")
    mean_value_of_long_term_variability = st.number_input("Mean Long Term Variability", value=3.0, step=0.01, format="%.3f")

    st.markdown("<div style='padding:6px 0 4px 0;color:#334155'>Histogram Features</div>", unsafe_allow_html=True)
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
# Helpers
# -----------------------------
def normalize_name(s):
    if pd.isna(s):
        return ''
    return str(s).strip().lower().replace(' ', '_')

def prediction_badge_html(label):
    """Return HTML badge for a prediction label (Normal/Suspect/Pathological)."""
    if pd.isna(label):
        return ""
    label = str(label)
    if label.lower() == 'normal':
        return "<span class='badge-cell' style='display:inline-block;padding:6px 10px;border-radius:10px;background:linear-gradient(90deg,#16a34a,#43c85a);color:#ffffff;font-weight:700;'>🟢 Normal</span>"
    if label.lower() == 'suspect':
        return "<span class='badge-cell' style='display:inline-block;padding:6px 10px;border-radius:10px;background:linear-gradient(90deg,#f59e0b,#fbbf24);color:#0f1724;font-weight:700;'>🟡 Suspect</span>"
    if label.lower() == 'pathological':
        return "<span class='badge-cell' style='display:inline-block;padding:6px 10px;border-radius:10px;background:linear-gradient(90deg,#ef4444,#fb7185);color:#ffffff;font-weight:700;'>🔴 Pathological</span>"
    return f"<span class='badge-cell' style='display:inline-block;padding:6px 10px;border-radius:10px;background:rgba(15,23,36,0.06);color:#0f1724;font-weight:700;'>{label}</span>"

# -----------------------------
# CSV batch prediction flow
# -----------------------------
if 'csv_file' in locals() and csv_file is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("CSV Batch Prediction")

    # Read full CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None

    if df is not None:
        # --- Show full Uploaded CSV in a scrollable container so user can view all rows/cols ---
        st.markdown("### Uploaded CSV")
        # convert full dataframe to HTML and wrap with scrollable div
        # using class 'minwide' so very wide tables will create horizontal scrollbar
        html_full = df.to_html(classes="fetal-table minwide", index=False, escape=True)
        wrapped = f"<div class='table-wrapper'>{html_full}</div>"
        st.write(wrapped, unsafe_allow_html=True)

        # ---------- mapping & prediction pipeline (auto-map + manual mapping) ----------
        orig_cols = list(df.columns)
        norm_to_orig = {normalize_name(c): c for c in orig_cols}
        norm_cols = set(norm_to_orig.keys())

        aliases = {
            'prolongued_decelerations': 'prolonged_decelerations',
            'prolonged_decelarations': 'prolonged_decelerations',
        }

        norm_features = [normalize_name(f) for f in FEATURES]
        auto_map = {}
        unmatched = []
        for nf, feat in zip(norm_features, FEATURES):
            if nf in norm_cols:
                auto_map[feat] = norm_to_orig[nf]
                continue
            if nf in aliases and aliases[nf] in norm_cols:
                auto_map[feat] = norm_to_orig[aliases[nf]]
                continue
            for a_k, a_v in aliases.items():
                if a_k in norm_cols and a_v == nf:
                    auto_map[feat] = norm_to_orig[a_k]
                    break
            if feat in auto_map:
                continue
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
            st.warning("CSV columns don't exactly match expected feature names. The app attempted to auto-map the following (you can correct below):")
            mapped_table = {f: auto_map.get(f, '---') for f in FEATURES}
            st.write(pd.DataFrame.from_dict(mapped_table, orient='index', columns=['mapped_csv_column']).rename_axis('expected_feature'))
            st.markdown("If the mapping is wrong, correct using the manual mapping form below or rename your CSV headers.")

            mapping = {}
            cols = list(df.columns)
            with st.form(key='mapping_form'):
                st.markdown("#### Map CSV columns to model features")
                for feat in FEATURES:
                    pre = auto_map.get(feat, '--constant--')
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
                probs = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probs = model.predict_proba(X)
                    except Exception:
                        probs = None

                # Build results dataframe
                res_df = df.copy()
                res_df['predicted_label'] = preds
                res_df['predicted_class'] = res_df['predicted_label'].map({0: 'Normal', 1: 'Suspect', 2: 'Pathological'})

                if probs is not None:
                    for i in range(probs.shape[1]):
                        res_df[f'prob_class_{i}'] = probs[:, i]

                # Build feature->column mapping (auto_map or manual mapping fallback)
                feature_to_col = {}
                if 'auto_map' in locals() and isinstance(auto_map, dict) and len(auto_map) > 0:
                    for feat, colname in auto_map.items():
                        feature_to_col[feat] = colname
                if 'mapping' in locals() and isinstance(mapping, dict):
                    for feat, m in mapping.items():
                        if m['type'] == 'col':
                            feature_to_col[feat] = m['value']
                def try_find_column(feat_name):
                    nf = normalize_name(feat_name)
                    for c in res_df.columns:
                        if normalize_name(c) == nf:
                            return c
                    return None
                for feat in FEATURES:
                    if feat not in feature_to_col:
                        found = try_find_column(feat)
                        if found:
                            feature_to_col[feat] = found

                # Choose a small set of features to display (customize here)
                display_features = [
                    'baseline_value',
                    'mean_value_of_short_term_variability',
                    'histogram_mean',
                    'percentage_of_time_with_abnormal_long_term_variability'
                ]

                display_cols = []
                for f in display_features:
                    col = feature_to_col.get(f)
                    if col and col in res_df.columns:
                        display_cols.append(col)

                if 'predicted_label' not in res_df.columns:
                    res_df['predicted_label'] = preds

                # Compute predicted_prob if probabilities available
                prob_cols = [c for c in res_df.columns if c.startswith('prob_class_')]
                if prob_cols:
                    try:
                        if 'predicted_prob' not in res_df.columns:
                            def get_pred_prob(row):
                                lab = int(row['predicted_label'])
                                colname = f'prob_class_{lab}'
                                return row[colname] if colname in row.index else np.nan
                            res_df['predicted_prob'] = res_df.apply(get_pred_prob, axis=1)
                    except Exception:
                        if prob_cols:
                            res_df['predicted_prob'] = res_df[prob_cols[0]]
                else:
                    res_df['predicted_prob'] = np.nan

                # Build final display columns (short)
                display_cols = display_cols + ['predicted_class', 'predicted_prob']
                display_cols = [c for i, c in enumerate(display_cols) if c and c in res_df.columns and c not in display_cols[:i]]

                st.markdown("### Prediction results")
                if len(display_cols) == 0:
                    st.warning("Could not identify columns to display. The app will show the full result table instead.")
                    st.dataframe(res_df.head(100), use_container_width=True)
                else:
                    display_df = res_df[display_cols].head(100).copy()

                    # create badge column + formatted prob
                    display_df['Prediction_Badge'] = display_df['predicted_class'].apply(prediction_badge_html)
                    def fmt_prob(x):
                        try:
                            if np.isnan(x):
                                return ""
                            return f"{float(x):.4f}"
                        except Exception:
                            return str(x)
                    display_df['Pred_Prob'] = display_df['predicted_prob'].apply(fmt_prob)

                    # prepare pretty column names and ordering
                    rev_map = {v: k for k, v in feature_to_col.items()}
                    col_rename = {}
                    ordered_cols = []
                    for c in display_df.columns:
                        if c in rev_map:
                            pretty = rev_map[c].replace('_', ' ').title()
                            col_rename[c] = pretty
                            ordered_cols.append(pretty)
                        elif c == 'Prediction_Badge':
                            col_rename[c] = 'Prediction'
                            ordered_cols.append('Prediction')
                        elif c == 'Pred_Prob':
                            col_rename[c] = 'Pred_Prob'
                            ordered_cols.append('Pred_Prob')

                    display_df = display_df.rename(columns=col_rename)

                    # Ensure ordering: features first, then Prediction, Pred_Prob
                    final_order = []
                    for col in ordered_cols:
                        if col not in ['Prediction', 'Pred_Prob']:
                            final_order.append(col)
                    if 'Prediction' in display_df.columns:
                        final_order.append('Prediction')
                    if 'Pred_Prob' in display_df.columns:
                        final_order.append('Pred_Prob')

                    # Build HTML table and render (escape=False to allow badge HTML)
                    html_table = display_df[final_order].to_html(classes="fetal-table", index=False, escape=False)
                    st.write(html_table, unsafe_allow_html=True)

                # Download button for full results
                csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions as CSV", data=csv_bytes, file_name='fetal_predictions.csv', mime='text/csv')

                # -----------------------------
                # Summary (WITH BADGES)
                # -----------------------------
                st.markdown("#### Summary")
                cnts = res_df['predicted_class'].value_counts().reset_index()
                cnts.columns = ['predicted_class', 'count']
                order = ['Normal', 'Suspect', 'Pathological']
                ordered_rows = []
                for k in order:
                    row = cnts[cnts['predicted_class'] == k]
                    if not row.empty:
                        ordered_rows.append(row.iloc[0])
                others = cnts[~cnts['predicted_class'].isin(order)]
                for _, r in others.iterrows():
                    ordered_rows.append(r)

                summary_html = "<table class='fetal-table' style='max-width:420px'><thead><tr><th>Prediction</th><th>Count</th></tr></thead><tbody>"
                for r in ordered_rows:
                    lbl = r['predicted_class']
                    cnt = int(r['count'])
                    badge = prediction_badge_html(lbl)
                    summary_html += f"<tr><td style='vertical-align:middle'>{badge}</td><td style='font-weight:700;vertical-align:middle'>{cnt}</td></tr>"
                summary_html += "</tbody></table>"
                st.write(summary_html, unsafe_allow_html=True)

                # Show distribution chart
                chart_df = res_df['predicted_class'].value_counts().reset_index()
                chart_df.columns = ['predicted_class', 'count']
                bar = alt.Chart(chart_df).mark_bar().encode(
                    x='predicted_class:N',
                    y='count:Q'
                ).properties(height=250)
                st.altair_chart(bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Main: Manual single-record prediction
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Manual Prediction")

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

    # Charts
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
