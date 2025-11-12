# blood_app_offline_colored_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

st.set_page_config(page_title="Brijesh Gupta's Everything about you in a drop (Offline)", layout="wide")

# ---------------- Helpers ----------------
def safe_read_csv(uploaded_file):
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")

def auto_clean(df):
    df = df.copy()
    drop_cols = [c for c in df.columns if any(x in c.lower() for x in ("id","serial","srno","index"))]
    df = df.drop(columns=drop_cols, errors='ignore')
    df.columns = [c.strip() for c in df.columns]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in df.columns if c not in num_cols]
    if num_cols:
        imputer_num = SimpleImputer(strategy="mean")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])
    label_encoders = {}
    for col in obj_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        except Exception:
            uniques = {v:i for i,v in enumerate(df[col].astype(str).unique())}
            df[col] = df[col].astype(str).map(uniques)
    return df, num_cols, obj_cols, label_encoders

def train_model(X, y):
    if y.dtype.kind in 'biufc' and y.nunique() > 20:
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model_type = "regressor"
    else:
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model_type = "classifier"
    model.fit(X, y)
    return model, model_type

def prepare_sample_for_prediction(sample_df, feature_cols, session_label_encoders, df_clean=None):
    if isinstance(sample_df, dict):
        sample_df = pd.DataFrame([sample_df])
    missing = [c for c in feature_cols if c not in sample_df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    row = []
    for col in feature_cols:
        val = sample_df.iloc[0][col]
        if session_label_encoders and col in session_label_encoders:
            le = session_label_encoders[col]
            try:
                encoded = le.transform([str(val)])[0]
                row.append(float(encoded)); continue
            except Exception:
                vv = str(val).strip().lower()
                if vv in ("m","male","man"): row.append(1.0); continue
                if vv in ("f","female","woman"): row.append(0.0); continue
                if vv in ("yes","y","true","t"): row.append(1.0); continue
                if vv in ("no","n","false","f"): row.append(0.0); continue
                if df_clean is not None and col in df_clean.columns:
                    row.append(float(df_clean[col].mean())); continue
                row.append(-1.0); continue
        if isinstance(val, str):
            vstr = val.strip()
            if vstr.lower() in ("m","male","man"): row.append(1.0); continue
            if vstr.lower() in ("f","female","woman"): row.append(0.0); continue
            if vstr.lower() in ("yes","y","true","t"): row.append(1.0); continue
            if vstr.lower() in ("no","n","false","f"): row.append(0.0); continue
            vstr2 = vstr.replace(",", "")
            try:
                row.append(float(vstr2)); continue
            except Exception:
                if df_clean is not None and col in df_clean.columns:
                    row.append(float(df_clean[col].mean())); continue
                row.append(-1.0); continue
        try:
            row.append(float(val))
        except Exception:
            if df_clean is not None and col in df_clean.columns:
                row.append(float(df_clean[col].mean()))
            else:
                row.append(-1.0)
    return np.array(row).reshape(1, -1)

# ---------------- Symptom dictionary ----------------
SYMPTOM_DICT = {
    "hemoglobin": {
        "low": {"symptoms":["fatigue","pale skin","shortness of breath","dizziness"], "next_steps":["Consult physician","Check iron levels","Repeat test"]},
        "high": {"symptoms":["headache","blurred vision"], "next_steps":["Check hydration","Consult physician"]}
    },
    "wbc": {
        "low": {"symptoms":["frequent infections"], "next_steps":["Consult doctor","Avoid exposure to infections"]},
        "high": {"symptoms":["fever","inflammation"], "next_steps":["Check for infection","Consult doctor"]}
    },
    "platelets": {
        "low": {"symptoms":["easy bruising","prolonged bleeding"], "next_steps":["Avoid blood-thinning drugs","Consult hematologist"]},
        "high": {"symptoms":["headache","dizziness"], "next_steps":["Evaluate causes","Consult physician"]}
    },
    "rbc": {
        "low": {"symptoms":["fatigue","weakness"], "next_steps":["Check for anemia causes","Consult doctor"]},
        "high": {"symptoms":["headache","dizziness"], "next_steps":["Check hydration","Consult physician"]}
    },
    "__generic__": {
        "low": {"symptoms":["lower-than-average value"], "next_steps":["Consult specialist","Repeat test"]},
        "high": {"symptoms":["higher-than-average value"], "next_steps":["Consult specialist","Review medications"]}
    }
}
def map_symptoms_and_steps(param_name, direction):
    p = param_name.lower()
    for key in SYMPTOM_DICT.keys():
        if key != "__generic__" and key in p:
            entry = SYMPTOM_DICT[key]
            return entry.get(direction, SYMPTOM_DICT["__generic__"].get(direction, {}))
    return SYMPTOM_DICT["__generic__"].get(direction, {})

# ---------------- Styling helpers ----------------
def styled_numeric_highlight(df, threshold_std=1.5):
    """
    Stronger, more visible colors:
      HIGH => #ff6666 (red)
      LOW  => #66b3ff (blue)
    """
    dfc = df.copy()
    numeric_cols = dfc.select_dtypes(include=[np.number]).columns.tolist()
    means = dfc[numeric_cols].mean()
    stds = dfc[numeric_cols].std().replace(0, np.nan)

    def cell_style(val, col):
        try:
            if pd.isna(val):
                return ""
            m = means[col]
            s = stds[col]
            if pd.isna(s) or s == 0:
                return ""
            if val > m + threshold_std * s:
                return "background-color: #ff6666; color:black; font-weight:600"
            if val < m - threshold_std * s:
                return "background-color: #66b3ff; color:black; font-weight:600"
            return ""
        except Exception:
            return ""

    sty = dfc.style
    for col in numeric_cols:
        sty = sty.applymap(lambda v, c=col: cell_style(v, c), subset=[col])
    sty = sty.format(precision=3, na_rep="")
    return sty

# ---------------- Session init ----------------
for k,v in {
    "df_raw": None, "df_clean": None, "num_cols": [], "obj_cols": [], 
    "label_encoders": {}, "model": None, "model_type": None, "feature_cols": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- UI ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Upload & Clean", "Train & Predict", "Dashboards", "Settings"])

# ---------- Welcome ----------
if page == "Welcome":
    st.title("Welcome to Brijesh Gupta's Everything about you in a drop")
    st.markdown("""
    **Overview**

    This offline project analyses blood-test CSVs: it cleans the data automatically, trains a model on your chosen target column, 
    predicts values for new samples, flags abnormal parameters, and suggests educational symptom lists and next steps — all offline.

    **Workflows**

    1. **Upload & Clean** — Upload CSV. The app auto-cleans: trims headers, drops ID columns, imputes numeric missing values, and label-encodes categorical columns.
    2. **Train & Predict** — Choose a target column and train a RandomForest model. After training, predict on a new single-row CSV or enter values manually.
    3. **Heuristic & Symptoms** — If a parameter is unusually HIGH/LOW (based on ±1.5×std), the app flags it and shows probable symptoms and suggested next steps from a local dictionary.
    4. **Dashboards & Visuals** — Explore the cleaned dataset using colorized tables, correlation heatmap, line/scatter/histogram/box plots and pie charts.
    5. **Save & Export** — Download the cleaned CSV and the trained model for reuse.

    **Tips**
    - Keep column names clean (no trailing spaces).  
    - If a CSV fails to parse, open it in Excel and "Save as CSV (UTF-8)".  
    - The symptom suggestions are rule-based and educational only — not medical advice.
    """)
    st.warning("**Medical disclaimer:** Educational purposes only. Consult a qualified healthcare professional for any medical concerns.")
    st.write("---")
    st.write("Quick tips:")
    st.write("- Use the 'Upload & Clean' page to preview the cleaned table.")
    st.write("- Train with at least a few dozen rows for sensible model behaviour.")
    st.write("- Use the Dashboards page to spot outliers visually.")

# ---------- Upload & Clean ----------
elif page == "Upload & Clean":
    st.header("Upload CSV & Auto-clean")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = safe_read_csv(uploaded)
        st.session_state.df_raw = df
        st.write("Raw preview (interactive)")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        if st.button("Auto-clean"):
            with st.spinner("Cleaning..."):
                try:
                    df_clean, num_cols, obj_cols, label_encs = auto_clean(df)
                    st.session_state.df_clean = df_clean
                    st.session_state.num_cols = num_cols
                    st.session_state.obj_cols = obj_cols
                    st.session_state.label_encoders = label_encs
                    st.success("Cleaned.")
                    st.write("Clean preview (interactive)")
                    st.dataframe(df_clean.head())
                    st.write("Colorized preview (values high/low highlighted):")
                    sty = styled_numeric_highlight(df_clean.head(50))
                    st.write(sty.to_html(), unsafe_allow_html=True)
                    csv = df_clean.to_csv(index=False).encode("utf-8")
                    st.download_button("Download cleaned CSV", csv, file_name="cleaned_dataset.csv")
                except Exception as e:
                    st.error("Cleaning failed: " + repr(e))
    else:
        st.info("Upload your CSV to begin cleaning.")

# ---------- Train & Predict ----------
elif page == "Train & Predict":
    st.header("Train & Predict")
    if st.session_state.df_clean is None:
        st.info("Upload & clean a dataset first.")
    else:
        dfc = st.session_state.df_clean
        st.write("Dataset shape:", dfc.shape)
        target = st.selectbox("Select target column to predict", dfc.columns)
        test_size = st.slider("Test size (%)", 10, 40, 20)
        if st.button("Train model"):
            X = dfc.drop(columns=[target])
            y = dfc[target]
            try:
                model, model_type = train_model(X, y)
            except Exception:
                X = X.select_dtypes(include=[np.number])
                model, model_type = train_model(X, y)
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.feature_cols = X.columns.tolist()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
            preds = model.predict(X_test)
            if model_type == "classifier":
                acc = accuracy_score(y_test, preds)
                st.write(f"Classifier accuracy: {acc:.3f}")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues")
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig)
            else:
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                st.write(f"Regressor MSE: {mse:.3f}, R2: {r2:.3f}")
            try:
                imp = model.feature_importances_
                fi = pd.DataFrame({"feature": st.session_state.feature_cols, "importance": imp}).sort_values("importance", ascending=False)
                st.subheader("Feature importance (table & colorized bar)")
                st.dataframe(fi.head(20))
                sty = fi.head(20).style.background_gradient(subset=["importance"], cmap="Greens")
                st.write(sty.to_html(), unsafe_allow_html=True)
            except Exception:
                st.info("Feature importance not available.")
            model_pack = {"model": model, "model_type": model_type, "feature_cols": st.session_state.feature_cols, "label_encoders": st.session_state.label_encoders}
            joblib.dump(model_pack, "trained_blood_model.joblib")
            st.success("Model trained and saved (trained_blood_model.joblib).")
        st.write("---")
        st.subheader("Predict on a new sample")
        if st.session_state.model is None:
            st.info("Train a model first.")
        else:
            uploaded_new = st.file_uploader("Upload a single-row CSV for prediction (optional)", type=["csv"], key="predcsv")
            sample_arr = None
            feature_cols = st.session_state.feature_cols
            if uploaded_new:
                newdf = safe_read_csv(uploaded_new)
                try:
                    new_aligned = newdf[feature_cols].copy()
                    sample_arr = prepare_sample_for_prediction(new_aligned, feature_cols, st.session_state.label_encoders, df_clean=st.session_state.df_clean)
                except Exception as e:
                    st.error(f"Uploaded CSV error: {e}")
            else:
                st.write("Enter values for each feature (text allowed for categorical).")
                input_dict = {}
                for c in feature_cols:
                    if c in st.session_state.obj_cols:
                        input_dict[c] = st.text_input(f"{c}", value=str(round(st.session_state.df_clean[c].mean(),3)), key=f"inp_{c}")
                    else:
                        try:
                            default = float(st.session_state.df_clean[c].mean())
                        except Exception:
                            default = 0.0
                        input_dict[c] = st.text_input(f"{c}", value=str(default), key=f"inp_{c}")
                if st.button("Predict"):
                    try:
                        sample_arr = prepare_sample_for_prediction(input_dict, feature_cols, st.session_state.label_encoders, df_clean=st.session_state.df_clean)
                    except Exception as e:
                        st.error(f"Could not prepare input: {e}")
            if sample_arr is not None:
                try:
                    model_pack = joblib.load("trained_blood_model.joblib") if os.path.exists("trained_blood_model.joblib") else None
                    model = model_pack["model"] if model_pack else st.session_state.model
                except Exception:
                    model = st.session_state.model
                try:
                    pred = model.predict(sample_arr)
                    st.subheader("Prediction result")
                    st.write(pred[0])
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(sample_arr)
                        st.write("Max confidence:", float(np.max(proba)))
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()
                st.subheader("Top contributing features (model importance)")
                try:
                    fi = model.feature_importances_
                    fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi}).sort_values("importance", ascending=False)
                    sty_fi = fi_df.head(8).style.background_gradient(subset=["importance"], cmap="Greens")
                    st.write(sty_fi.to_html(), unsafe_allow_html=True)
                except Exception:
                    st.info("Feature importance not available.")
                st.subheader("Heuristic explanation & symptom suggestions")
                heur = []
                suggestions = []
                for i, f in enumerate(feature_cols):
                    mean = st.session_state.df_clean[f].mean()
                    std = st.session_state.df_clean[f].std()
                    val = sample_arr[0,i]
                    if std > 0:
                        if val > mean + 1.5*std:
                            heur.append(f"`{f}` is HIGH ({val:.2f} vs mean {mean:.2f})")
                            mp = map_symptoms_and_steps(f, "high")
                            if mp:
                                suggestions.append((f, "high", mp))
                        elif val < mean - 1.5*std:
                            heur.append(f"`{f}` is LOW ({val:.2f} vs mean {mean:.2f})")
                            mp = map_symptoms_and_steps(f, "low")
                            if mp:
                                suggestions.append((f, "low", mp))
                if heur:
                    st.markdown("**Observations:**")
                    for h in heur:
                        st.write("-", h)
                    st.markdown("**Symptom suggestions (educational, rule-based):**")
                    for (param, direction, mp) in suggestions:
                        st.markdown(f"**{param} — {direction.upper()}**")
                        if "symptoms" in mp and mp["symptoms"]:
                            st.write("- Possible symptoms:", ", ".join(mp["symptoms"]))
                        if "next_steps" in mp and mp["next_steps"]:
                            st.write("- Suggested next steps:", "; ".join(mp["next_steps"]))
                else:
                    st.write("All parameters are within expected ranges compared to this dataset (heuristic).")

# ---------- Dashboards ----------
elif page == "Dashboards":
    st.header("Dashboards")
    if st.session_state.df_clean is None:
        st.info("Upload & clean dataset first.")
    else:
        dfc = st.session_state.df_clean
        st.subheader("Snapshot (interactive)")
        st.dataframe(dfc.head())
        st.subheader("Colorized table (high/low highlighted)")
        sty = styled_numeric_highlight(dfc.head(200))
        st.write(sty.to_html(), unsafe_allow_html=True)
        st.subheader("Numeric descriptive stats")
        st.dataframe(dfc.describe().T.style.background_gradient(cmap="Blues"))
        st.subheader("Correlation heatmap (annotated)")
        fig, ax = plt.subplots(figsize=(10,8))
        corr = dfc.corr()
        sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax, square=True, linewidths=.5)
        st.pyplot(fig)
        st.write("---")
        st.subheader("Choose plots for exploration")
        numeric_cols = dfc.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in dfc.columns if c not in numeric_cols]
        if not numeric_cols and not categorical_cols:
            st.info("No columns to plot.")
        else:
            # Line plot
            if numeric_cols:
                st.markdown("**Line plot (sequence)**")
                line_col = st.selectbox("Select numeric column for line plot", numeric_cols, key="line_col")
                fig1, ax1 = plt.subplots()
                ax1.plot(dfc[line_col].values, linewidth=1.5)
                ax1.set_xlabel("Row index"); ax1.set_ylabel(line_col); ax1.set_title(f"Line plot: {line_col}")
                st.pyplot(fig1)

            # Scatter plot
            if len(numeric_cols) >= 2:
                st.markdown("**Scatter plot**")
                xcol = st.selectbox("X-axis", numeric_cols, index=0, key="scatter_x")
                ycol = st.selectbox("Y-axis", numeric_cols, index=1, key="scatter_y")
                hue_col = None
                if categorical_cols:
                    hue_choice = st.selectbox("Optional categorical hue (color)", ["None"] + categorical_cols, key="hue")
                    if hue_choice != "None":
                        hue_col = hue_choice
                fig2, ax2 = plt.subplots()
                if hue_col:
                    cats = dfc[hue_col].astype(str)
                    pal = sns.color_palette("tab10", n_colors=cats.nunique())
                    sns.scatterplot(x=dfc[xcol], y=dfc[ycol], hue=cats, palette=pal, ax=ax2, s=50, alpha=0.7)
                    ax2.legend([],[], frameon=False)
                else:
                    ax2.scatter(dfc[xcol], dfc[ycol], alpha=0.7, s=40)
                ax2.set_xlabel(xcol); ax2.set_ylabel(ycol); ax2.set_title(f"Scatter: {xcol} vs {ycol}")
                st.pyplot(fig2)

            # Histogram
            if numeric_cols:
                st.markdown("**Histogram**")
                hist_col = st.selectbox("Select column for histogram", numeric_cols, key="hist_col")
                bins = st.slider("Bins", 5, 100, 20, key="hist_bins")
                fig3, ax3 = plt.subplots()
                sns.histplot(dfc[hist_col].dropna(), bins=bins, kde=False, ax=ax3)
                ax3.set_xlabel(hist_col); ax3.set_title(f"Histogram: {hist_col}")
                st.pyplot(fig3)

            # Box plot
            if numeric_cols:
                st.markdown("**Box plot (with points)**")
                box_col = st.selectbox("Select numeric column for boxplot", numeric_cols, key="box_col")
                fig4, ax4 = plt.subplots()
                sns.boxplot(y=dfc[box_col].dropna(), ax=ax4)
                sns.stripplot(y=dfc[box_col].dropna(), color="black", size=3, alpha=0.4, ax=ax4)
                ax4.set_title(f"Box plot: {box_col}")
                st.pyplot(fig4)

            # Pie charts
            st.markdown("**Pie charts**")
            pie_choice = st.radio("Choose pie chart type", ("Categorical distribution", "Numeric bins distribution"))
            if pie_choice == "Categorical distribution" and categorical_cols:
                cat_col = st.selectbox("Select categorical column for pie", categorical_cols, key="pie_cat")
                counts = dfc[cat_col].astype(str).value_counts()
                labels = counts.index.tolist()
                sizes = counts.values
                # dynamic colors
                colors = sns.color_palette("tab20", n_colors=len(labels))
                figp, axp = plt.subplots()
                axp.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
                axp.axis("equal")
                axp.set_title(f"Pie: {cat_col}")
                st.pyplot(figp)
            elif pie_choice == "Numeric bins distribution" and numeric_cols:
                num_col = st.selectbox("Select numeric column for pie bins", numeric_cols, key="pie_num")
                bins_p = st.slider("Number of bins", 2, 12, 4, key="pie_bins")
                binned = pd.cut(dfc[num_col].dropna(), bins=bins_p)
                counts = binned.value_counts().sort_index()
                labels = [str(i) for i in counts.index]
                sizes = counts.values
                colors = sns.color_palette("Set2", n_colors=len(labels))
                figp, axp = plt.subplots()
                axp.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
                axp.axis("equal")
                axp.set_title(f"Pie bins: {num_col}")
                st.pyplot(figp)
            else:
                st.info("No columns available for this pie chart type.")

# ---------- Settings ----------
elif page == "Settings":
    st.header("Settings")
    st.write("Offline version — no external APIs required.")
    if st.button("Load model from file (trained_blood_model.joblib)"):
        if os.path.exists("trained_blood_model.joblib"):
            pack = joblib.load("trained_blood_model.joblib")
            st.session_state.model = pack.get("model")
            st.session_state.model_type = pack.get("model_type")
            st.session_state.feature_cols = pack.get("feature_cols")
            st.session_state.label_encoders = pack.get("label_encoders", {})
            st.success("Model loaded into session.")
        else:
            st.error("trained_blood_model.joblib not found.")
    if st.button("Clear session"):
        for k in ["df_raw","df_clean","num_cols","obj_cols","label_encoders","model","model_type","feature_cols"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared.")
