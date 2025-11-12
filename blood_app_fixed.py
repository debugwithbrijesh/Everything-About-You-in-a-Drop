# blood_app_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Optional LLM import (safe if missing)
# --- at top of your file, import this instead of old openai usage ---
# ✅ FINAL VERSION — clean, works with openai>=1.0.0
from openai import OpenAI

def llm_explain(openai_key, context_text, max_tokens=220):
    """
    Uses GPT model for educational explanation.
    Returns (response_text, error_message)
    """
    if not openai_key:
        return None, "No OpenAI API key provided."

    try:
        client = OpenAI(api_key=openai_key)
    except Exception as e:
        return None, f"Failed to construct OpenAI client: {e}"

    prompt = (
        "You are an educational assistant (not a doctor). "
        "Given the following patient/dataset summary, identify which parameters are high or low "
        "compared to typical dataset mean, list possible symptoms and suggested non-prescriptive next steps. "
        "Add a brief disclaimer that it's for educational use only.\n\n"
        f"{context_text}\n\n"
        "Keep it concise and easy to understand."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, repr(e)

    # Try chat completions (gpt-3.5-turbo is widely available)
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        # New client returns .choices[0].message.content
        return resp.choices[0].message.content.strip(), None
    except Exception as e_chat:
        chat_err = repr(e_chat)

    # Fallback: responses API (attempt)
    try:
        resp2 = client.responses.create(
            model="gpt-3.5-turbo",
            input=prompt,
            max_tokens=max_tokens,
            temperature=0.3
        )
        text_out = None
        if hasattr(resp2, "output") and resp2.output:
            pieces = []
            for item in resp2.output:
                if isinstance(item, dict):
                    if "content" in item and isinstance(item["content"], list):
                        for c in item["content"]:
                            if isinstance(c, dict) and "text" in c:
                                pieces.append(c["text"])
                            elif isinstance(c, str):
                                pieces.append(c)
                    else:
                        for v in item.values():
                            if isinstance(v, str):
                                pieces.append(v)
                elif isinstance(item, str):
                    pieces.append(item)
            text_out = " ".join(pieces).strip()
        elif hasattr(resp2, "generations") and resp2.generations:
            try:
                text_out = resp2.generations[0][0].text
            except Exception:
                text_out = None

        if text_out:
            return text_out, None
        else:
            return None, f"Responses API returned no extractable text. raw: {repr(resp2)}"
    except Exception as e_resp:
        resp_err = repr(e_resp)

    combined = f"Chat error: {chat_err}\nResponses error: {resp_err}"
    return None, combined


st.set_page_config(page_title="GuptaJi", layout="wide")

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
    """
    - Drop ID-like columns
    - Impute numeric with mean
    - Fill categorical with mode and label-encode them
    Returns cleaned df, numeric cols, categorical cols, label_encoders dict
    """
    df = df.copy()
    # drop obvious ID columns
    drop_cols = [c for c in df.columns if any(x in c.lower() for x in ("id","serial","srno","index"))]
    df = df.drop(columns=drop_cols, errors='ignore')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in df.columns if c not in num_cols]

    # numeric impute
    if num_cols:
        imputer_num = SimpleImputer(strategy="mean")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    label_encoders = {}
    # categorical fill + encode
    for col in obj_cols:
        # fill missing with mode or empty string
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        except Exception:
            # fallback mapping
            uniques = {v:i for i,v in enumerate(df[col].astype(str).unique())}
            df[col] = df[col].astype(str).map(uniques)

    return df, num_cols, obj_cols, label_encoders

def train_model(X, y):
    # choose regressor if numeric with many uniques, else classifier
    if y.dtype.kind in 'biufc' and y.nunique() > 20:
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model_type = "regressor"
    else:
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model_type = "classifier"
    model.fit(X, y)
    return model, model_type

def prepare_sample_for_prediction(sample_df, feature_cols, session_label_encoders, df_clean=None):
    """
    Converts sample_df (DataFrame single-row or dict) to numeric numpy array aligned to feature_cols.
    Uses label encoders from session when available and sensible fallbacks for unseen categories.
    """
    if isinstance(sample_df, dict):
        sample_df = pd.DataFrame([sample_df])

    # Ensure columns exist
    missing = [c for c in feature_cols if c not in sample_df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    row = []
    for col in feature_cols:
        val = sample_df.iloc[0][col]
        # If we have a label encoder for this column, use it
        if session_label_encoders and col in session_label_encoders:
            le = session_label_encoders[col]
            try:
                encoded = le.transform([str(val)])[0]
                row.append(float(encoded))
                continue
            except Exception:
                # unseen category - try to map common tokens
                vv = str(val).strip().lower()
                if vv in ("m","male","man"):
                    row.append(float(1))
                    continue
                if vv in ("f","female","woman"):
                    row.append(float(0))
                    continue
                if vv in ("yes","y","true","t"):
                    row.append(float(1))
                    continue
                if vv in ("no","n","false"):
                    row.append(float(0))
                    continue
                # fallback to dataset mean if available
                if df_clean is not None and col in df_clean.columns:
                    row.append(float(df_clean[col].mean()))
                else:
                    row.append(float(-1.0))
                continue

        # No encoder: attempt numeric coercion or common mappings
        if isinstance(val, str):
            vstr = val.strip()
            if vstr.lower() in ("m","male","man"):
                row.append(float(1)); continue
            if vstr.lower() in ("f","female","woman"):
                row.append(float(0)); continue
            if vstr.lower() in ("yes","y","true","t"):
                row.append(float(1)); continue
            if vstr.lower() in ("no","n","false"):
                row.append(float(0)); continue
            vstr2 = vstr.replace(",", "")
            try:
                row.append(float(vstr2)); continue
            except Exception:
                if df_clean is not None and col in df_clean.columns:
                    row.append(float(df_clean[col].mean()))
                else:
                    row.append(float(-1.0))
                continue

        # numeric types
        try:
            row.append(float(val))
        except Exception:
            if df_clean is not None and col in df_clean.columns:
                row.append(float(df_clean[col].mean()))
            else:
                row.append(float(-1.0))

    arr = np.array(row).reshape(1, -1)
    return arr

# ---------------- Session init ----------------
if "df_raw" not in st.session_state: st.session_state.df_raw = None
if "df_clean" not in st.session_state: st.session_state.df_clean = None
if "num_cols" not in st.session_state: st.session_state.num_cols = []
if "obj_cols" not in st.session_state: st.session_state.obj_cols = []
if "label_encoders" not in st.session_state: st.session_state.label_encoders = {}
if "model" not in st.session_state: st.session_state.model = None
if "model_type" not in st.session_state: st.session_state.model_type = None
if "feature_cols" not in st.session_state: st.session_state.feature_cols = []
if "scaler" not in st.session_state: st.session_state.scaler = None
if "openai_key" not in st.session_state: st.session_state.openai_key = None

# ---------------- UI ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Upload & Clean", "Train & Predict", "Dashboards", "Settings"])

# ---------- Welcome ----------
if page == "Welcome":
    st.title("Welcome to Brijesh Gupta's: Everything about you in a drop")
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

# ---------- Upload & Clean ----------
elif page == "Upload & Clean":
    st.header("Upload CSV & Auto-clean")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = safe_read_csv(uploaded)
        st.session_state.df_raw = df
        st.write("Raw preview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

        if st.button("Auto-clean"):
            with st.spinner("Cleaning..."):
                df_clean, num_cols, obj_cols, label_encs = auto_clean(df)
                st.session_state.df_clean = df_clean
                st.session_state.num_cols = num_cols
                st.session_state.obj_cols = obj_cols
                st.session_state.label_encoders = label_encs
            st.success("Cleaned. Numeric imputed, categorical encoded.")
            st.write("Clean preview")
            st.dataframe(df_clean.head())
            csv = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button("Download cleaned CSV", csv, file_name="cleaned_dataset.csv")
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
            # try training with all columns; fallback to numeric only
            try:
                model, model_type = train_model(X, y)
            except Exception:
                X = X.select_dtypes(include=[np.number])
                model, model_type = train_model(X, y)
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.feature_cols = X.columns.tolist()
            # evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
            preds = model.predict(X_test)
            if model_type == "classifier":
                acc = accuracy_score(y_test, preds)
                st.write(f"Classifier accuracy: {acc:.3f}")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig)
            else:
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                st.write(f"Regressor MSE: {mse:.3f}, R2: {r2:.3f}")
            # feature importance
            try:
                imp = model.feature_importances_
                fi = pd.DataFrame({"feature": st.session_state.feature_cols, "importance": imp}).sort_values("importance", ascending=False)
                st.subheader("Feature importance")
                st.dataframe(fi.head(20))
                fig2, ax2 = plt.subplots(figsize=(6, max(3, len(fi.head(10))*0.4)))
                sns.barplot(x="importance", y="feature", data=fi.head(10), ax=ax2)
                st.pyplot(fig2)
            except Exception:
                st.info("Feature importance not available.")

            # Save model + encoders + metadata
            model_pack = {
                "model": model,
                "model_type": model_type,
                "feature_cols": st.session_state.feature_cols,
                "label_encoders": st.session_state.label_encoders,
            }
            joblib.dump(model_pack, "trained_blood_model.joblib")
            st.success("Model trained and saved (trained_blood_model.joblib).")

        st.write("---")
        st.subheader("Predict on a new sample")
        if st.session_state.model is None:
            st.info("Train a model first.")
        else:
            # allow upload single-row CSV or manual entry
            uploaded_new = st.file_uploader("Upload a single-row CSV for prediction (optional)", type=["csv"], key="predcsv")
            sample_arr = None
            feature_cols = st.session_state.feature_cols
            if uploaded_new:
                newdf = safe_read_csv(uploaded_new)
                # try align columns
                try:
                    new_aligned = newdf[feature_cols].copy()
                except Exception as e:
                    st.error(f"Uploaded CSV does not match model features: {e}")
                    new_aligned = None
                if new_aligned is not None:
                    try:
                        sample_arr = prepare_sample_for_prediction(new_aligned, feature_cols, st.session_state.label_encoders, df_clean=st.session_state.df_clean)
                    except Exception as e:
                        st.error(f"Could not prepare uploaded sample: {e}")
            else:
                st.write("Enter values for each feature (text allowed for categorical).")
                input_dict = {}
                # If a column was originally categorical (obj_cols), use text_input so user can type 'M'/'F'
                for c in feature_cols:
                    if c in st.session_state.obj_cols:
                        input_dict[c] = st.text_input(f"{c}", value=str(round(st.session_state.df_clean[c].mean(),3)))
                    else:
                        # numeric columns: allow numbers
                        try:
                            default = float(st.session_state.df_clean[c].mean())
                        except Exception:
                            default = 0.0
                        # use text input as well because user may paste 'F' accidentally; prepared function will handle mapping
                        input_dict[c] = st.text_input(f"{c}", value=str(default))
                if st.button("Predict"):
                    try:
                        sample_arr = prepare_sample_for_prediction(input_dict, feature_cols, st.session_state.label_encoders, df_clean=st.session_state.df_clean)
                    except Exception as e:
                        st.error(f"Could not prepare manual sample: {e}")
                        sample_arr = None

            if sample_arr is not None:
                # load model (from session or file)
                try:
                    model_pack = joblib.load("trained_blood_model.joblib")
                    model = model_pack["model"]
                    feature_cols = model_pack["feature_cols"]
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

                # quick feature importance explanation
                st.subheader("Top contributing features (model importance)")
                try:
                    fi = model.feature_importances_
                    fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi}).sort_values("importance", ascending=False)
                    st.table(fi_df.head(8))
                except Exception:
                    st.info("Feature importance not available.")

                # build context for AI explanation / heuristics
                context = "Feature values:\n"
                for i, f in enumerate(feature_cols):
                    context += f"- {f}: {float(sample_arr[0,i])}\n"
                context += f"\nModel prediction: {pred[0]}."

                # LLM explanation if key provided
                if st.session_state.openai_key:
                    with st.spinner("Calling AI for explanation..."):
                        ai_resp = llm_explain(st.session_state.openai_key, context)
                        if ai_resp:
                            st.subheader("AI explanation (educational)")
                            st.write(ai_resp)
                        else:
                            st.warning("AI call failed; showing heuristic explanation.")
                if not st.session_state.openai_key:
                    st.subheader("Heuristic explanation (no OpenAI key configured)")
                    heur = []
                    for i, f in enumerate(feature_cols):
                        mean = st.session_state.df_clean[f].mean()
                        std = st.session_state.df_clean[f].std()
                        val = sample_arr[0,i]
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
                    else:
                        st.write("All parameters are within expected ranges compared to this dataset (heuristic).")

# ---------- Dashboards ----------
elif page == "Dashboards":
    st.header("Dashboards")
    if st.session_state.df_clean is None:
        st.info("Upload & clean dataset first.")
    else:
        dfc = st.session_state.df_clean
        st.subheader("Snapshot")
        st.dataframe(dfc.head())
        st.subheader("Missing values (after cleaning)")
        st.bar_chart(dfc.isnull().sum())
        st.subheader("Numeric descriptive stats")
        st.dataframe(dfc.describe().T)
        st.subheader("Correlation heatmap")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(dfc.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ---------- Settings ----------
# In the Settings page where you accept key:
key = st.text_input("OpenAI key", type="password")
if key:
    st.session_state.openai_key = key
    st.success("OpenAI key saved to session (not on disk).")
    # --- Settings UI: Test key button (paste into your Settings page block) ---
if st.button("Test OpenAI key (show detailed error)"):
    key = st.session_state.get("openai_key")
    if not key:
        st.error("No OpenAI key found in session. Enter the key above and try again.")
    else:
        st.info("Testing OpenAI key (chat)...")
        sample_context = "Test: short ping. Say 'hello' in one sentence."
        text, err = llm_explain(key, sample_context, max_tokens=40)
        if err:
            st.error("AI test failed. Full error details (copy and share if you need help):")
            st.code(err)
            st.warning("Common causes: invalid key (401), rate limit (429), network/firewall, model not found.")
        else:
            st.success("AI test succeeded. Example response:")
            st.write(text)


if st.button("Test OpenAI key"):
    st.info("Testing key by making a small API call...")
    sample_context = "Test: one numeric feature A: 10, B: 20. Predict if A is high or low."
    resp_text, resp_err = llm_explain(st.session_state.openai_key, sample_context, max_tokens=80)
    if resp_err:
        st.error("AI test failed. Error message:")
        st.code(resp_err)
        st.warning("If it's an authentication error, check your API key and quota. If it's a network error, check your internet.")
    else:
        st.success("AI test succeeded. Example response:")
        st.write(resp_text)

