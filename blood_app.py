# blood_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: LLM (OpenAI)
try:
    import openai
except Exception:
    openai = None

# Optional: shap (may be heavy)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Brilliance Hub — Smart Blood Analyzer", layout="wide")

# --------- Helpers ----------
def safe_read_csv(uploaded_file):
    # try common encodings and remove BOM
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue
    uploaded_file.seek(0)
    # fallback
    return pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")

def auto_clean(df):
    df = df.copy()
    # Drop obviously useless columns (ID-like)
    drop_cols = [c for c in df.columns if any(x in c.lower() for x in ("id","serial","srno","index"))]
    df = df.drop(columns=drop_cols, errors='ignore')

    # separate numeric and object
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in df.columns if c not in num_cols]

    # Impute numeric with mean
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="mean")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Fill categorical with mode, then label-encode
    label_encoders = {}
    for col in obj_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        except Exception:
            # fallback: map unique to ints
            uniques = {v:i for i,v in enumerate(df[col].unique())}
            df[col] = df[col].map(uniques)

    return df, num_cols, obj_cols, label_encoders

def train_model(X, y):
    # decide classifier or regressor
    if y.dtype.kind in 'biufc' and y.nunique() > 20:
        # regression
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model_type = "regressor"
    else:
        # classification
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model_type = "classifier"
    model.fit(X, y)
    return model, model_type

def llm_explain(openai_key, context_text, max_tokens=220):
    if openai is None:
        return None
    openai.api_key = openai_key
    prompt = (
        "You are a helpful medical-data-aware assistant (educational only, not a doctor). "
        "Given the following patient / dataset summary, identify which parameters are high or low "
        "compared to normal and list possible associated symptoms and suggested next steps. "
        "Keep it concise, add a short non-medical next-step recommendation and a clear disclaimer.\n\n"
        f"{context_text}\n\n"
        "Answer in simple Hindi/English neutral tone. Keep educational disclaimers."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # try fallback
        try:
            resp = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return resp.choices[0].text.strip()
        except Exception as e2:
            return None

# --------- UI Pages ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Upload & Clean", "Train & Predict", "Dashboards", "Settings"])

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_type" not in st.session_state:
    st.session_state.model_type = None
if "label_encoders" not in st.session_state:
    st.session_state.label_encoders = {}

# ---------- WELCOME ----------
if page == "Welcome":
    st.title("Welcome Brijesh Gupta's: Smart Blood Analyzer")
    st.markdown("""
    **Overview:** This app uploads a CSV of blood-test records, cleans it automatically, trains a model for any column you select, 
    predicts column-wise results, and provides AI-powered explanations and dashboards.
    
    **Quick flow:**
    1. Upload CSV → Clean (auto)  
    2. Select target column → Train model  
    3. Predict on new samples → Get explanations + symptoms & suggested next steps (educational)  
    4. Visualize dashboards and download cleaned data/model.
    """)
    st.warning("**Medical disclaimer:** This project is for educational purposes only. It is NOT medical advice. Consult a doctor for any health decisions.")
    st.write("---")
    st.write("Tips:")
    st.write("- Keep CSV header row as column names, avoid spaces or special characters if possible.")
    st.write("- If CSV fails to read, try saving as UTF-8 without BOM.")
    st.write("- For better models, make sure the dataset has enough labeled rows for the chosen target.")

# ---------- UPLOAD & CLEAN ----------
elif page == "Upload & Clean":
    st.header("Upload CSV & Automatic Cleaning")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        df = safe_read_csv(uploaded)
        st.session_state.df_raw = df
        st.write("Raw preview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

        if st.button("Auto-clean this dataset"):
            with st.spinner("Cleaning..."):
                df_clean, num_cols, obj_cols, label_encs = auto_clean(df)
                st.session_state.df_clean = df_clean
                st.session_state.num_cols = num_cols
                st.session_state.obj_cols = obj_cols
                st.session_state.label_encoders = label_encs
            st.success("Dataset cleaned (missing values handled and categorical encoded).")
            st.write("Clean preview:")
            st.dataframe(df_clean.head())

            csv = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button("Download cleaned CSV", csv, file_name="cleaned_dataset.csv")

    else:
        st.info("Upload a CSV to begin. The app will try multiple encodings and handle BOM automatically.")

# ---------- TRAIN & PREDICT ----------
elif page == "Train & Predict":
    st.header("Train model & Predict")
    if st.session_state.df_clean is None:
        st.info("Upload & clean a dataset first (go to Upload & Clean).")
    else:
        df_clean = st.session_state.df_clean
        st.write("Clean dataset shape:", df_clean.shape)
        target = st.selectbox("Select target column to predict (choose any column):", df_clean.columns)
        test_size = st.slider("Test set size (%)", 10, 50, 20)
        random_state = st.number_input("Random seed", value=42, step=1)

        if st.button("Train model now"):
            X = df_clean.drop(columns=[target])
            y = df_clean[target]
            # standardize numeric features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
            # if the dataset has non-numeric (should not after cleaning) make sure columns align
            st.session_state.scaler = scaler
            try:
                model, model_type = train_model(X, y)
            except Exception:
                # try using only numeric columns
                X_num = X.select_dtypes(include=[np.number])
                model, model_type = train_model(X_num, y)
                X = X_num
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.feature_cols = X.columns.tolist()
            # evaluate
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))
            preds = model.predict(X_test)
            if model_type == "classifier":
                acc = accuracy_score(y_test, preds)
                cm = confusion_matrix(y_test, preds)
                st.write(f"**Model type:** Classifier — Accuracy on test set: {acc:.3f}")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            else:
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                st.write(f"**Model type:** Regressor — MSE: {mse:.3f}, R2: {r2:.3f}")

            # feature importance (if supported)
            try:
                importances = model.feature_importances_
                fi = pd.DataFrame({"feature": st.session_state.feature_cols, "importance": importances}).sort_values("importance", ascending=False)
                st.subheader("Feature importance")
                st.dataframe(fi.head(20))
                fig2, ax2 = plt.subplots(figsize=(6, max(3, len(fi.head(10))*0.4)))
                sns.barplot(x="importance", y="feature", data=fi.head(10), ax=ax2)
                st.pyplot(fig2)
            except Exception:
                st.info("Feature importance not available for this model.")

            # save model + metadata
            joblib.dump({
                "model": model,
                "model_type": model_type,
                "feature_cols": st.session_state.feature_cols,
                "scaler": st.session_state.get("scaler", None),
                "label_encoders": st.session_state.label_encoders
            }, "trained_blood_model.joblib")
            st.success("Model trained and saved as trained_blood_model.joblib")

        st.write("---")
        st.subheader("Predict for a new sample")
        if st.session_state.model is None:
            st.info("Train a model first.")
        else:
            # allow manual entry
            st.write("Enter values for features (or upload a single-row CSV with same columns).")
            uploaded_new = st.file_uploader("Upload single-row CSV for prediction (optional)", type=["csv"], key="pred_csv")
            if uploaded_new:
                newdf = safe_read_csv(uploaded_new)
                # try to align columns
                newdf = newdf[st.session_state.feature_cols].copy()
                sample = newdf.iloc[0].values.reshape(1, -1)
            else:
                cols = st.session_state.feature_cols
                input_vals = []
                cols_left = cols
                with st.form("predict_form"):
                    for c in cols:
                        # show a numeric input by default
                        mean_val = float(df_clean[c].mean()) if c in df_clean.columns else 0.0
                        v = st.number_input(f"{c}", value=float(mean_val))
                        input_vals.append(v)
                    submitted = st.form_submit_button("Predict")
                if submitted:
                    sample = np.array(input_vals).reshape(1, -1)

            if 'sample' in locals():
                model_pack = joblib.load("trained_blood_model.joblib")
                model = model_pack["model"]
                feature_cols = model_pack["feature_cols"]
                # predict
                try:
                    pred = model.predict(sample)
                    st.subheader("Prediction result")
                    st.write(pred[0])
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(sample)
                        topprob = np.max(proba)
                        st.write(f"Confidence (max prob): {topprob:.3f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

                # Quick explanation using feature importances
                st.subheader("Explanation (top contributing features)")
                try:
                    fi = model.feature_importances_
                    fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi}).sort_values("importance", ascending=False)
                    st.table(fi_df.head(8))
                except Exception:
                    st.info("No feature importance available.")

                # Create a context string for AI explanation
                context = "Feature values:\n"
                for i, f in enumerate(feature_cols):
                    context += f"- {f}: {float(sample[0,i])}\n"
                context += f"\nModel prediction: {pred[0]}."

                # If user provided OpenAI key in settings, call LLM
                if st.session_state.get("openai_key", None):
                    with st.spinner("Calling AI to generate symptoms & suggestions..."):
                        ai_resp = llm_explain(st.session_state.openai_key, context)
                        if ai_resp:
                            st.subheader("AI explanation (educational)")
                            st.write(ai_resp)
                        else:
                            st.warning("AI call failed or not available. Showing heuristic suggestions instead.")
                            # fallback heuristics
                            st.markdown("- Parameter(s) high/low compared to dataset mean may indicate issues. Recommend consulting a doctor.")
                else:
                    st.subheader("Heuristic explanation (no OpenAI key configured)")
                    # simple heuristics: compare to dataset mean +/- sd
                    heur = []
                    for i, f in enumerate(feature_cols):
                        mean = df_clean[f].mean()
                        std = df_clean[f].std()
                        val = sample[0,i]
                        if std > 0:
                            if val > mean + 1.5*std:
                                heur.append(f"`{f}` is HIGH ({val:.2f} vs mean {mean:.2f})")
                            elif val < mean - 1.5*std:
                                heur.append(f"`{f}` is LOW ({val:.2f} vs mean {mean:.2f})")
                    if heur:
                        st.markdown("**Observations:**")
                        for h in heur:
                            st.write("-", h)
                        st.markdown("**Suggested next steps (educational):**")
                        st.write("- Review abnormal parameters with a healthcare professional.")
                        st.write("- Consider repeating the test to confirm.")
                        st.write("- Share detailed reports with a clinician for interpretation.")
                    else:
                        st.write("All parameters are within expected ranges compared to this dataset (heuristic).")

# ---------- DASHBOARDS ----------
elif page == "Dashboards":
    st.header("Dashboards & Visualizations")
    if st.session_state.df_clean is None:
        st.info("Upload & clean a dataset first.")
    else:
        dfc = st.session_state.df_clean
        st.subheader("Dataset snapshot")
        st.dataframe(dfc.head())

        st.subheader("Missing values (should be zero after cleaning)")
        st.bar_chart(dfc.isnull().sum())

        st.subheader("Descriptive statistics (numeric)")
        st.dataframe(dfc.describe().T)

        st.subheader("Correlation heatmap")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(dfc.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Histogram of numeric features")
        num_cols = dfc.select_dtypes(include=[np.number]).columns.tolist()
        chosen = st.multiselect("Choose columns to histogram (up to 4)", num_cols, default=(num_cols[:4] if len(num_cols)>0 else []))
        if chosen:
            fig2, axs = plt.subplots(len(chosen), 1, figsize=(7, 3*len(chosen)))
            if len(chosen)==1:
                axs = [axs]
            for ax, c in zip(axs, chosen):
                sns.histplot(dfc[c], kde=True, ax=ax)
                ax.set_title(c)
            st.pyplot(fig2)

# ---------- SETTINGS ----------
elif page == "Settings":
    st.header("Settings & Model Management")
    st.write("You can provide an OpenAI API key (optional) to enable AI explanations for predictions.")
    openai_key = st.text_input("OpenAI API key (optional)", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("OpenAI key stored in session (will not be saved to disk).")

    if st.button("Load existing trained model (trained_blood_model.joblib)"):
        if os.path.exists("trained_blood_model.joblib"):
            pack = joblib.load("trained_blood_model.joblib")
            st.session_state.model = pack.get("model")
            st.session_state.model_type = pack.get("model_type")
            st.session_state.feature_cols = pack.get("feature_cols")
            st.session_state.scaler = pack.get("scaler")
            st.success("Model loaded into session.")
        else:
            st.error("No 'trained_blood_model.joblib' file found in working directory.")

    if st.button("Clear session & uploaded data"):
        for k in ["df_raw", "df_clean", "model", "model_type", "feature_cols","label_encoders","scaler","openai_key"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared. Re-upload your CSV to start again.")
