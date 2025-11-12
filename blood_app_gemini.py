# blood_app_gemini.py
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
import seaborn as sns

# Try to import Google Gemini client
try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_SDK_AVAILABLE = False

st.set_page_config(page_title="Welcome to Brijesh Gupta's Smart Analyzer", layout="wide")

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

# ---------------- Gemini LLM wrapper ----------------

# robust gemini wrapper (paste into your app)
import google.generativeai as genai

def choose_gemini_model(api_key, prefer_list=None):
    """Return a model name available in your account (or error)."""
    if prefer_list is None:
        prefer_list = ["chat-bison", "text-bison", "gemini-1.5", "gemini", "chat"]
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass
    try:
        gen = genai.list_models()
    except Exception as e:
        return None, f"list_models() failed: {repr(e)}"
    found = []
    try:
        for m in gen:
            name = None
            if hasattr(m, "name"): name = m.name
            elif hasattr(m, "id"): name = m.id
            else:
                try: name = str(m)
                except: name = None
            if not name: continue
            found.append(name)
            for p in prefer_list:
                if p.lower() in name.lower():
                    return name, None
    except Exception as e:
        return None, f"iterating models generator failed: {repr(e)}"
    if found:
        return found[0], None
    return None, "No models discovered in genai.list_models()"

def llm_explain_gemini(api_key, context_text, max_tokens=300, preferred_model=None):
    """
    Returns (text, error). Tries several SDK call patterns compatible with multiple versions.
    """
    if not api_key:
        return None, "No Gemini API key provided."

    # pick model
    model_name = preferred_model
    if not model_name:
        model_name, sel_err = choose_gemini_model(api_key)
        if sel_err:
            return None, f"Model selection failed: {sel_err}"

    # configure (if required)
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

    prompt = (
        "You are an educational assistant (not a doctor). Given the following patient/dataset summary, "
        "identify which parameters are high or low compared to the dataset mean, list possible associated symptoms "
        "and suggest non-prescriptive next steps. Add a short disclaimer that this is educational only.\n\n"
        f"{context_text}\n\n"
        "Keep it concise and clear."
    )

    tried = []

    # 1) Preferred: chat.completions.create (many SDKs expose this)
    try:
        if hasattr(genai, "chat") and hasattr(genai.chat, "completions"):
            tried.append(f"chat.completions.create model={model_name}")
            resp = genai.chat.completions.create(
                model=model_name,
                messages=[{"author":"user","content":prompt}],
                max_output_tokens=max_tokens
            )
            # extract content if present
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                content = getattr(cand, "content", None) or getattr(cand, "message", None) or getattr(cand, "text", None)
                if content:
                    # content may be string or list/dict: stringify sensibly
                    if isinstance(content, (list, tuple)):
                        return " ".join(str(x) for x in content).strip(), None
                    return str(content).strip(), None
            if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
                return str(resp["candidates"][0].get("content","")).strip(), None
    except Exception as e:
        tried.append(f"chat.completions error: {repr(e)}")

    # 2) Try genai.generate (older pattern) if present
    try:
        if hasattr(genai, "generate"):
            tried.append(f"genai.generate model={model_name}")
            resp = genai.generate(model=model_name, prompt=prompt, max_output_tokens=max_tokens)
            if hasattr(resp, "candidates") and resp.candidates:
                content = getattr(resp.candidates[0], "content", None)
                if content:
                    return (content if isinstance(content, str) else str(content)).strip(), None
            if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
                return str(resp["candidates"][0].get("content","")).strip(), None
    except Exception as e:
        tried.append(f"generate error: {repr(e)}")

    # 3) Try GenerativeModel patterns
    try:
        if hasattr(genai, "GenerativeModel"):
            tried.append(f"GenerativeModel({model_name})")
            model = genai.GenerativeModel(model_name)
            if hasattr(model, "generate_text"):
                r = model.generate_text(prompt, max_output_tokens=max_tokens)
                if hasattr(r, "text") and r.text:
                    return r.text.strip(), None
                if isinstance(r, dict) and "output" in r:
                    return str(r["output"]).strip(), None
            if hasattr(model, "generate_content"):
                r2 = model.generate_content(prompt)
                if hasattr(r2, "text") and r2.text:
                    return r2.text.strip(), None
                if hasattr(r2, "candidates") and r2.candidates:
                    cand = r2.candidates[0]
                    content = getattr(cand, "content", None) or cand.get("content", None)
                    if content:
                        return (content if isinstance(content, str) else str(content)).strip(), None
    except Exception as e:
        tried.append(f"GenerativeModel error: {repr(e)}")

    # nothing worked
    return None, "All Gemini call attempts failed. Tried: " + " | ".join(tried)


    # configure
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return None, f"Failed to configure Gemini SDK: {e}"

    prompt = (
        "You are an educational assistant (not a doctor). Given the following patient/dataset summary, "
        "identify which parameters are high or low compared to the dataset mean, list possible associated symptoms "
        "and suggest non-prescriptive next steps. Add a short disclaimer that this is educational only.\n\n"
        f"{context_text}\n\n"
        "Answer concisely and clearly."
    )

    # Try a couple of common call patterns for the SDK (some SDK versions expose different helpers)
    errors = []
    try:
        # Preferred modern call (may work with many versions)
        # genai.generate returns object with candidates[0].content in many SDK variants
        resp = genai.generate(model="gemini-2.5-Pro", prompt=prompt, max_output_tokens=max_tokens)
        # try to extract text
        if hasattr(resp, "candidates") and resp.candidates:
            text = resp.candidates[0].content
            return text.strip(), None
        if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
            return resp["candidates"][0].get("content","").strip(), None
    except Exception as e:
        errors.append(f"generate() error: {repr(e)}")

    try:
        # Alternative pattern: use GenerativeModel (some SDK versions)
        model = genai.GenerativeModel("gemini-2.5-Pro")
        # try generate_text or generate_content depending on SDK
        if hasattr(model, "generate_text"):
            resp2 = model.generate_text(prompt, max_output_tokens=max_tokens)
            if hasattr(resp2, "text"):
                return resp2.text.strip(), None
            if isinstance(resp2, dict) and "output" in resp2:
                return str(resp2["output"]).strip(), None
        elif hasattr(model, "generate_content"):
            resp2 = model.generate_content(prompt)
            if hasattr(resp2, "text"):
                return resp2.text.strip(), None
            if hasattr(resp2, "candidates") and resp2.candidates:
                return resp2.candidates[0].content.strip(), None
    except Exception as e:
        errors.append(f"GenerativeModel error: {repr(e)}")

    # If we reach here, all attempts failed
    return None, " | ".join(errors) or "Unknown Gemini error"

# ---------------- Session init ----------------
for k,v in {
    "df_raw": None, "df_clean": None, "num_cols": [], "obj_cols": [], 
    "label_encoders": {}, "model": None, "model_type": None, "feature_cols": [], "gemini_key": None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- UI ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Upload & Clean", "Train & Predict", "Dashboards", "Settings"])

# ---------- Welcome ----------
if page == "Welcome":
    st.title("Welcome to Brijesh Gupta's Smart blood Analyser")
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
                try:
                    df_clean, num_cols, obj_cols, label_encs = auto_clean(df)
                    st.session_state.df_clean = df_clean
                    st.session_state.num_cols = num_cols
                    st.session_state.obj_cols = obj_cols
                    st.session_state.label_encoders = label_encs
                    st.success("Cleaned. Numeric imputed, categorical encoded.")
                    st.write(df_clean.head())
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
            try:
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

                # evaluate
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
                except Exception:
                    st.info("Feature importance not available.")

                # save model
                model_pack = {"model": model, "model_type": model_type, "feature_cols": st.session_state.feature_cols, "label_encoders": st.session_state.label_encoders}
                joblib.dump(model_pack, "trained_blood_model.joblib")
                st.success("Model trained and saved.")
            except Exception as e:
                st.error("Training failed: " + repr(e))

        st.write("---")
        st.subheader("Predict on a new sample")
        if st.session_state.model is None:
            st.info("Train a model first.")
        else:
            uploaded_new = st.file_uploader("Upload single-row CSV for prediction (optional)", type=["csv"], key="predcsv")
            sample_arr = None
            feature_cols = st.session_state.feature_cols
            if uploaded_new:
                try:
                    newdf = safe_read_csv(uploaded_new)
                    new_aligned = newdf[feature_cols].copy()
                    sample_arr = prepare_sample_for_prediction(new_aligned, feature_cols, st.session_state.label_encoders, df_clean=st.session_state.df_clean)
                except Exception as e:
                    st.error("Uploaded CSV error: " + repr(e))
            else:
                st.write("Enter values for each feature (text allowed for categorical).")
                input_dict = {}
                for c in feature_cols:
                    default = ""
                    try:
                        default = str(round(float(st.session_state.df_clean[c].mean()),3))
                    except Exception:
                        default = ""
                    input_dict[c] = st.text_input(f"{c}", value=default, key=f"inp_{c}")
                if st.button("Predict"):
                    try:
                        sample_arr = prepare_sample_for_prediction(input_dict, feature_cols, st.session_state.label_encoders, df_clean=st.session_state.df_clean)
                    except Exception as e:
                        st.error("Preparing sample failed: " + repr(e))

            if sample_arr is not None:
                try:
                    if os.path.exists("trained_blood_model.joblib"):
                        pack = joblib.load("trained_blood_model.joblib")
                        model = pack.get("model", st.session_state.model)
                    else:
                        model = st.session_state.model
                    pred = model.predict(sample_arr)
                    st.subheader("Prediction result")
                    st.write(pred[0])
                    if hasattr(model, "predict_proba"):
                        st.write("Max confidence:", float(np.max(model.predict_proba(sample_arr))))
                except Exception as e:
                    st.error("Prediction failed: " + repr(e))
                    st.stop()

                # top features
                try:
                    fi = model.feature_importances_
                    fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi}).sort_values("importance", ascending=False)
                    st.table(fi_df.head(8))
                except Exception:
                    st.info("Feature importance not available.")

                # AI explanation attempt (Gemini)
                context = "Feature values:\n"
                for i,f in enumerate(feature_cols):
                    context += f"- {f}: {float(sample_arr[0,i])}\n"
                context += f"\nModel prediction: {pred[0]}."

                gemini_err = None
                gemini_text = None
                if st.session_state.gemini_key:
                    with st.spinner("Calling Gemini for explanation..."):
                        gemini_text, gemini_err = llm_explain_gemini(st.session_state.gemini_key, context)
                        if gemini_err:
                            st.warning("AI call failed; showing heuristic explanation.")
                            st.code(gemini_err)
                        else:
                            st.subheader("AI (Gemini) explanation (educational)")
                            st.write(gemini_text)

                if (not st.session_state.gemini_key) or gemini_err:
                    st.subheader("Heuristic explanation (fallback)")
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
elif page == "Settings":
    st.header("Settings")
    st.write("Google Gemini API key (optional) — stored only for this session")
    key = st.text_input("Gemini API key", type="password")
    if key:
        st.session_state.gemini_key = key
        st.success("Gemini key saved to session (not stored to disk).")

    if st.button("Test Gemini key (show detailed error)"):
        k = st.session_state.get("gemini_key")
        if not k:
            st.error("No Gemini key in session. Enter key above and try again.")
        else:
            st.info("Testing Gemini key...")
            sample_context = "Test: say hello in one short sentence."
            txt, err = llm_explain_gemini(k, sample_context, max_tokens=40)
            if err:
                st.error("Gemini test failed. Full error (copy & paste here if you want help):")
                st.code(err)
                st.warning("Common issues: invalid key, network/firewall, SDK mismatch.")
            else:
                st.success("Gemini test succeeded. Example response:")
                st.write(txt)

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
        for k in ["df_raw","df_clean","num_cols","obj_cols","label_encoders","model","model_type","feature_cols","gemini_key"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared.")
