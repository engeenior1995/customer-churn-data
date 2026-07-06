"""
Customer Churn Prediction Dashboard
Deep Learning Assignment 01 - ANN Pipeline
Run with: streamlit run churn_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Churn Prediction | ANN Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SEED LOCK (reproducibility)
# =========================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e8c;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Artificial Neural Network (ANN) — Deep Learning Assignment 01</p>', unsafe_allow_html=True)
st.divider()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📁 Data & EDA", "⚙️ Preprocessing", "🧠 Model & Training", "📈 Evaluation", "🔮 Live Prediction"]
)

st.sidebar.divider()
st.sidebar.markdown("**Course:** Advanced Deep Learning (DL-501)")
st.sidebar.markdown("**Framework:** TensorFlow / Keras & Scikit-Learn")

# =========================================================
# DATA LOADING (cached so it only loads once)
# =========================================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# =========================================================
# PREPROCESSING FUNCTION (cached)
# =========================================================
@st.cache_data
def preprocess_data(df):
    df_clean = df.copy()

    # Fix hidden blank-string entries in TotalCharges
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    missing_before = df_clean['TotalCharges'].isnull().sum()
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())

    # Drop non-predictive ID column
    df_clean.drop('customerID', axis=1, inplace=True)

    # Target conversion
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

    # Categorical encoding
    categorical_cols = df_clean.select_dtypes(include='object').columns.tolist()
    binary_cols = [c for c in categorical_cols if df_clean[c].nunique() == 2]
    multiclass_cols = [c for c in categorical_cols if df_clean[c].nunique() > 2]

    le = LabelEncoder()
    for col in binary_cols:
        df_clean[col] = le.fit_transform(df_clean[col])

    df_clean = pd.get_dummies(df_clean, columns=multiclass_cols, drop_first=True)
    bool_cols = df_clean.select_dtypes(include='bool').columns
    df_clean[bool_cols] = df_clean[bool_cols].astype(int)

    return df_clean, missing_before, binary_cols, multiclass_cols

df_clean, missing_before, binary_cols, multiclass_cols = preprocess_data(df)

X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

# =========================================================
# TRAIN/TEST SPLIT + SCALING (used across pages)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# PAGE 1: DATA & EDA
# =========================================================
if page == "📁 Data & EDA":
    st.header("Part 1 — Data Exploration (EDA)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1] - 1)
    col3.metric("Churn Rate", f"{(df['Churn']=='Yes').mean()*100:.1f}%")
    col4.metric("Missing (hidden)", int(missing_before))

    st.subheader("First 5 Records")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Dataset Info")
    buf_col1, buf_col2 = st.columns(2)
    with buf_col1:
        st.write("**Data Types**")
        st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
    with buf_col2:
        st.write("**Descriptive Statistics**")
        st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Target Variable Distribution (Class Imbalance Check)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(df['Churn'].value_counts().rename("count"), use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(x='Churn', data=df, palette=['#1f4e8c', '#e07b39'], ax=ax)
        ax.set_title("Churn Class Distribution")
        st.pyplot(fig)

    st.info("💡 **Insight:** The dataset is imbalanced — most customers do NOT churn. "
            "This means raw accuracy alone can be misleading; Precision/Recall matter more.")

# =========================================================
# PAGE 2: PREPROCESSING
# =========================================================
elif page == "⚙️ Preprocessing":
    st.header("Part 2 — Data Preprocessing Pipeline")

    st.subheader("1. Missing Value Remediation")
    st.write(f"`TotalCharges` had **{missing_before}** hidden blank-string entries "
              "(stored as spaces, not true NaN). These were converted to NaN and filled using the **median**.")

    st.subheader("2. Dimensional Pruning")
    st.write("Dropped `customerID` — a unique identifier with zero predictive variance.")

    st.subheader("3. Categorical Encoding")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.write("**Binary columns (Label Encoded):**")
        st.write(binary_cols)
    with ec2:
        st.write("**Multi-class columns (One-Hot Encoded):**")
        st.write(multiclass_cols)

    st.subheader("4. Final Preprocessed Data")
    st.dataframe(df_clean.head(), use_container_width=True)
    st.write(f"Final feature count after encoding: **{X.shape[1]}** (this becomes the ANN input layer size)")

    st.subheader("5. Train/Test Split & Scaling")
    st.write("80% Training / 20% Testing split, with `StandardScaler` applied "
              "(fit on training data only, to avoid data leakage).")

# =========================================================
# PAGE 3: MODEL & TRAINING
# =========================================================
elif page == "🧠 Model & Training":
    st.header("Part 3, 4 & 5 — ANN Architecture, Compilation & Training")

    st.subheader("Network Architecture")
    st.code("""
Sequential([
    Dense(32, activation='relu', input_shape=(n_features,)),  # Hidden Layer 1
    Dense(16, activation='relu'),                              # Hidden Layer 2
    Dense(1, activation='sigmoid')                              # Output Layer
])
    """, language="python")

    st.subheader("Compilation Settings")
    s1, s2, s3 = st.columns(3)
    s1.metric("Optimizer", "Adam")
    s2.metric("Loss Function", "Binary Crossentropy")
    s3.metric("Metric", "Accuracy")

    st.subheader("Training Configuration")
    t1, t2, t3 = st.columns(3)
    t1.metric("Epochs", "50")
    t2.metric("Batch Size", "32")
    t3.metric("Validation Split", "20%")

    st.divider()

    if st.button("🚀 Train Model Now", type="primary"):
        n_features = X_train_scaled.shape[1]

        model = Sequential([
            Dense(32, activation='relu', input_shape=(n_features,)),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        progress_bar = st.progress(0, text="Training in progress...")

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_bar.progress((epoch + 1) / 50, text=f"Epoch {epoch+1}/50 — accuracy: {logs['accuracy']:.3f}")

        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.20,
            verbose=0,
            callbacks=[StreamlitCallback()]
        )

        progress_bar.empty()
        st.success("✅ Training complete!")

        # Store in session state so other pages can access it
        st.session_state['model'] = model
        st.session_state['history'] = history.history
        st.session_state['scaler'] = scaler
        st.session_state['feature_columns'] = X.columns.tolist()

        # Quick save option
        model.save('churn_ann_model.keras')
        with open('churn_ann_model.keras', 'rb') as f:
            st.download_button("💾 Download Trained Model (.keras)", f, file_name="churn_ann_model.keras")

    if 'model' in st.session_state:
        st.info("✅ A trained model is loaded in this session. Go to **Evaluation** or **Live Prediction** pages.")
    else:
        st.warning("⚠️ No model trained yet in this session. Click the button above to train.")

# =========================================================
# PAGE 4: EVALUATION
# =========================================================
elif page == "📈 Evaluation":
    st.header("Part 6, 7 & 8 — Model Evaluation & Visualization")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first on the **Model & Training** page.")
    else:
        model = st.session_state['model']
        history = st.session_state['history']

        y_pred_prob = model.predict(X_test_scaled, verbose=0)
        y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.subheader("Test Set Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Precision", f"{prec:.2%}")
        m3.metric("Recall", f"{rec:.2%}")
        m4.metric("F1-Score", f"{f1:.2%}")

        st.divider()

        col_cm, col_curves = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

        with col_curves:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        st.divider()
        st.subheader("Training Curves")
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history['accuracy'], label='Training Accuracy')
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linestyle='--')
        axes[0].set_title('Accuracy over Epochs')
        axes[0].set_xlabel('Epochs')
        axes[0].legend()

        axes[1].plot(history['loss'], label='Training Loss')
        axes[1].plot(history['val_loss'], label='Validation Loss', linestyle='--')
        axes[1].set_title('Loss over Epochs')
        axes[1].set_xlabel('Epochs')
        axes[1].legend()

        st.pyplot(fig2)

        st.subheader("Sample Predictions vs Actual")
        results_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Probability': y_pred_prob.flatten().round(4)
        }).reset_index(drop=True)
        st.dataframe(results_df.head(20), use_container_width=True)

# =========================================================
# PAGE 5: LIVE PREDICTION
# =========================================================
elif page == "🔮 Live Prediction":
    st.header("Live Churn Prediction Tool")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first on the **Model & Training** page.")
    else:
        st.write("Enter a customer's details below to predict churn probability.")

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                gender = st.selectbox("Gender", df['gender'].unique())
                senior = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Has Partner", ['Yes', 'No'])
                dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
                tenure = st.slider("Tenure (months)", 0, 72, 12)

            with c2:
                phone = st.selectbox("Phone Service", ['Yes', 'No'])
                internet = st.selectbox("Internet Service", df['InternetService'].unique())
                contract = st.selectbox("Contract", df['Contract'].unique())
                payment = st.selectbox("Payment Method", df['PaymentMethod'].unique())

            with c3:
                monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
                total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
                paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
                multiple_lines = st.selectbox("Multiple Lines", df['MultipleLines'].unique())

            submitted = st.form_submit_button("🔮 Predict Churn", type="primary")

        if submitted:
            # Build a single-row dataframe matching the original raw schema
            input_dict = {col: df[col].mode()[0] for col in df.columns if col not in ['customerID', 'Churn']}
            input_dict.update({
                'gender': gender, 'SeniorCitizen': senior, 'Partner': partner,
                'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone,
                'InternetService': internet, 'Contract': contract, 'PaymentMethod': payment,
                'MonthlyCharges': monthly, 'TotalCharges': total,
                'PaperlessBilling': paperless, 'MultipleLines': multiple_lines
            })
            input_df = pd.DataFrame([input_dict])

            # Apply the SAME preprocessing pipeline
            combined = pd.concat([df.drop(['customerID', 'Churn'], axis=1), input_df], ignore_index=True)
            categorical_cols = combined.select_dtypes(include='object').columns.tolist()
            binary_c = [c for c in categorical_cols if combined[c].nunique() == 2]
            multi_c = [c for c in categorical_cols if combined[c].nunique() > 2]

            le = LabelEncoder()
            for col in binary_c:
                combined[col] = le.fit_transform(combined[col])
            combined = pd.get_dummies(combined, columns=multi_c, drop_first=True)
            bool_cols = combined.select_dtypes(include='bool').columns
            combined[bool_cols] = combined[bool_cols].astype(int)

            # Align columns to match training feature set exactly
            input_row = combined.tail(1)
            input_row = input_row.reindex(columns=st.session_state['feature_columns'], fill_value=0)

            input_scaled = st.session_state['scaler'].transform(input_row)
            prob = st.session_state['model'].predict(input_scaled, verbose=0)[0][0]
            prediction = "Churn" if prob >= 0.5 else "No Churn"

            st.divider()
            r1, r2 = st.columns(2)
            r1.metric("Prediction", prediction)
            r2.metric("Churn Probability", f"{prob:.2%}")

            if prediction == "Churn":
                st.error(f"⚠️ This customer is **at risk of churning** (probability: {prob:.1%}). "
                          "Consider a targeted retention offer.")
            else:
                st.success(f"✅ This customer is **likely to stay** (churn probability: {prob:.1%}).")
