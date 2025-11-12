import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Blood Test Analyzer - Brilliance Hub", layout="centered")
st.title("🩸 Everything About You in a Drop: A Smart Analyzer")

st.write("Upload your blood test dataset to clean, train, and analyze it with AI.")

# STEP 1: Upload CSV
uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # STEP 2: Data Cleaning
    st.subheader("🧹 Cleaning Data Automatically")

    # Fill missing values
    imputer = SimpleImputer(strategy="mean")
    df_clean = df.copy()

    # Handle non-numeric columns
    label_encoders = {}
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le

    df_clean = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)

    st.success("✅ Missing values handled and text columns encoded.")

    # STEP 3: Choose Target Column
    st.subheader("🎯 Choose Target Column")
    target_col = st.selectbox("Select the column to predict:", df_clean.columns)

    if st.button("Train Model"):
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("📈 Model Evaluation")
        st.write(f"**Accuracy:** {acc:.2f}")

        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # Feature importance
        st.subheader("📊 Feature Importance")
        fi = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(fi)

        # Save model
        joblib.dump(model, "blood_ai_model.joblib")
        st.success("💾 Model trained and saved as `blood_ai_model.joblib`")

        # STEP 4: Prediction
        st.subheader("🔮 Predict on New Sample")
        inputs = {}
        for col in X.columns:
            inputs[col] = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))
        if st.button("Predict Result"):
            sample = np.array(list(inputs.values())).reshape(1, -1)
            prediction = model.predict(sample)[0]
            st.success(f"🩺 Predicted Value: **{prediction}**")
