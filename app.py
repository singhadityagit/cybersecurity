
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="CyberShield | ML Intrusion Detector",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
h1, h2, h3 {
    color: #60a5fa;
}
div[data-testid="stMetric"] {
    background: #172238;
    border: 1px solid #293952;
    padding: 18px;
    border-radius: 12px;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "model" not in st.session_state:
    st.session_state.model = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None
if "target" not in st.session_state:
    st.session_state.target = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None
if "class_names" not in st.session_state:
    st.session_state.class_names = None

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛡️ CyberShield")
st.sidebar.caption("Machine Learning Security System")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "📂 Dataset Explorer",
        "🤖 Train Model",
        "🔍 Detect Traffic",
        "📚 Security Guide"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Educational prototype. Predictions depend on the quality "
    "and labels of your uploaded dataset."
)

st.title("🛡️ ML Network Intrusion Detector")
st.caption("Machine learning for network traffic analysis")

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.sidebar.file_uploader(
    "Upload network dataset (CSV)",
    type=["csv"]
)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.sidebar.error("The uploaded CSV is empty.")
            df = None
    except Exception as error:
        st.sidebar.error(f"Could not read CSV: {error}")

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":
    st.subheader("Security Operations Overview")
    st.write(
        "Analyze labeled network traffic with a machine learning model "
        "to identify patterns associated with normal activity and attacks."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Status", "Loaded" if df is not None else "Waiting")
    with col2:
        st.metric(
            "Dataset Rows",
            f"{len(df):,}" if df is not None else "—"
        )
    with col3:
        accuracy = st.session_state.accuracy
        st.metric(
            "Test Accuracy",
            f"{accuracy:.1%}" if accuracy is not None else "Not trained"
        )

    st.divider()
    st.subheader("What does this system do?")

    a, b = st.columns(2)
    with a:
        st.markdown("""
        ### 📊 Traffic Analysis
        - Explore uploaded CSV datasets
        - Review data types and missing values
        - Visualize the distribution of target labels

        ### 🤖 Machine Learning
        - Train a Random Forest classifier
        - Evaluate predictions on held-out test data
        - Review the confusion matrix and feature importance
        """)
    with b:
        st.markdown("""
        ### 🔍 Intrusion Detection
        - Classify selected traffic records
        - Review the predicted class
        - Inspect model confidence estimates

        ### 🔐 Security Awareness
        - Learn common network attack categories
        - Review basic security best practices
        - Understand the limits of ML detection
        """)

    st.warning(
        "This prototype analyzes data you provide. It does not monitor "
        "live network traffic or automatically block attacks."
    )

    if df is None:
        st.info("To begin, upload a labeled network traffic CSV using the sidebar.")

# ---------------- DATASET EXPLORER ----------------
elif page == "📂 Dataset Explorer":
    st.subheader("Dataset Explorer")

    if df is None:
        st.info("Upload a CSV dataset in the sidebar to explore it.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isna().sum().sum()))

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        with st.expander("Column details"):
            details = pd.DataFrame({
                "Column": df.columns,
                "Data Type": [str(t) for t in df.dtypes],
                "Missing Values": df.isna().sum().values,
                "Unique Values": df.nunique().values
            })
            st.dataframe(details, use_container_width=True)

        st.markdown("### Explore a column")
        selected_column = st.selectbox("Choose a column", df.columns)

        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.write(df[selected_column].describe())
            chart_data = df[selected_column].dropna()
            if not chart_data.empty:
                st.bar_chart(chart_data.reset_index(drop=True))
        else:
            counts = df[selected_column].astype(str).value_counts().head(20)
            st.bar_chart(counts)

        st.download_button(
            "Download dataset preview",
            df.head(1000).to_csv(index=False).encode("utf-8"),
            file_name="dataset_preview.csv",
            mime="text/csv"
        )

# ---------------- TRAIN MODEL ----------------
elif page == "🤖 Train Model":
    st.subheader("Train and Evaluate the Model")

    if df is None:
        st.info("Upload a labeled CSV dataset in the sidebar first.")
    else:
        target = st.selectbox(
            "Select the target label column",
            df.columns,
            help="Choose the column containing labels such as Normal or Attack."
        )

        st.caption("Other columns will be used as input features.")
        test_size = st.slider(
            "Test dataset percentage",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        )

        if st.button("🚀 Train Random Forest", use_container_width=True):
            try:
                data = df.dropna(subset=[target]).copy()
                if len(data) < 5:
                    st.error("The dataset needs at least five labeled records.")
                else:
                    y = data[target].astype(str)
                    X = data.drop(columns=[target])

                    # Remove columns that could leak labels or identify each row.
                    # Review your dataset and remove any other leakage columns.
                    X = X.drop(
                        columns=[
                            c for c in X.columns
                            if c.strip().lower() in ["label", "class", "attack_type", "target"]
                        ],
                        errors="ignore"
                    )

                    # Convert categorical features to numeric columns.
                    X = pd.get_dummies(X)
                    X = X.replace([float("inf"), float("-inf")], float("nan"))
                    X = X.fillna(0)

                    if X.shape[1] == 0:
                        st.error("No usable input features were found.")
                    elif y.nunique() < 2:
                        st.error("The target column must contain at least two classes.")
                    else:
                        # Stratify when every class has enough records.
                        stratify = y if y.value_counts().min() >= 2 else None

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=test_size / 100,
                            random_state=42,
                            stratify=stratify
                        )

                        model = RandomForestClassifier(
                            n_estimators=100,
                            random_state=42,
                            class_weight="balanced",
                            n_jobs=-1
                        )
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        accuracy = accuracy_score(y_test, predictions)

                        st.session_state.model = model
                        st.session_state.feature_columns = list(X.columns)
                        st.session_state.target = target
                        st.session_state.accuracy = accuracy
                        st.session_state.class_names = list(model.classes_)

                        st.success("Model training completed!")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Test Accuracy", f"{accuracy:.2%}")
                        m2.metric("Training Records", f"{len(X_train):,}")
                        m3.metric("Test Records", f"{len(X_test):,}")

                        st.markdown("### Confusion Matrix")
                        labels = list(model.classes_)
                        cm = confusion_matrix(y_test, predictions, labels=labels)
                        st.dataframe(
                            pd.DataFrame(
                                cm,
                                index=[f"Actual: {v}" for v in labels],
                                columns=[f"Predicted: {v}" for v in labels]
                            ),
                            use_container_width=True
                        )

                        st.markdown("### Classification Report")
                        report = classification_report(
                            y_test,
                            predictions,
                            output_dict=True,
                            zero_division=0
                        )
                        st.dataframe(pd.DataFrame(report).transpose())

                        st.markdown("### Top 10 Important Features")
                        importance = pd.Series(
                            model.feature_importances_,
                            index=X.columns
                        ).sort_values(ascending=False).head(10)

                        st.bar_chart(importance)

                        st.download_button(
                            "Download test predictions",
                            pd.DataFrame({
                                "Actual": y_test.to_numpy(),
                                "Predicted": predictions
                            }).to_csv(index=False).encode("utf-8"),
                            file_name="test_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as error:
                st.error(f"Training failed: {error}")
                st.info(
                    "Check the target column, dataset size, class counts, "
                    "and whether the selected features contain usable data."
                )

        if st.session_state.model is not None:
            st.info(
                "A trained model is available in this session. Retrain after "
                "changing the dataset or target column."
            )

# ---------------- DETECT TRAFFIC ----------------
elif page == "🔍 Detect Traffic":
    st.subheader("Network Traffic Classification")

    model = st.session_state.model
    feature_columns = st.session_state.feature_columns

    if model is None:
        st.info("Go to Train Model and train a model first.")
    elif df is None:
        st.info("Upload the same dataset used for training to select a traffic record.")
    else:
        target = st.session_state.target

        if target not in df.columns:
            st.error("The uploaded dataset does not contain the training target column.")
        else:
            candidates = df.drop(columns=[target]).copy()
            candidates = candidates.drop(
                columns=[
                    c for c in candidates.columns
                    if c.strip().lower() in ["label", "class", "attack_type", "target"]
                ],
                errors="ignore"
            )
            candidates = pd.get_dummies(candidates)
            candidates = candidates.replace(
                [float("inf"), float("-inf")], float("nan")
            ).fillna(0)
            candidates = candidates.reindex(
                columns=feature_columns, fill_value=0
            )

            row_number = st.number_input(
                "Select record number (starting at 1)",
                min_value=1,
                max_value=len(candidates),
                value=1,
                step=1
            )

            if st.button("🔎 Analyze Traffic", use_container_width=True):
                row = candidates.iloc[[int(row_number) - 1]]
                prediction = model.predict(row)[0]

                st.markdown("### Prediction Result")
                st.write(f"**Predicted class:** {prediction}")

                probability = model.predict_proba(row)[0]
                probabilities = pd.Series(
                    probability, index=model.classes_
                ).sort_values(ascending=False)

                is_attack = str(prediction).strip().lower() not in [
                    "normal", "benign", "0", "normal traffic"
                ]

                if is_attack:
                    st.error(
                        "This record was classified as a potential attack. "
                        "Verify it with security logs and other evidence."
                    )
                else:
                    st.success(
                        "This record was classified as normal/benign by the model."
                    )

                st.markdown("### Model Probability Estimates")
                st.bar_chart(probabilities)

                st.caption(
                    "These are model probability estimates, not proof that "
                    "traffic is safe or malicious. Confirm the label mapping "
                    "for your particular dataset."
                )

# ---------------- SECURITY GUIDE ----------------
elif page == "📚 Security Guide":
    st.subheader("Network Security Guide")
    st.write(
        "Network intrusion detection helps identify traffic patterns that "
        "may indicate unauthorized or harmful activity."
    )

    with st.expander("Common attack categories", expanded=True):
        st.markdown("""
        - **Denial of Service (DoS):** Attempts to overwhelm a service with traffic.
        - **Brute-force attack:** Repeated attempts to guess passwords or credentials.
        - **Port scanning:** Probing systems to discover reachable services.
        - **Malware communication:** Traffic associated with malicious software.
        - **Privilege abuse:** Misuse of authorized or compromised access.
        """)

    with st.expander("Recommended security practices"):
        st.markdown("""
        1. Keep operating systems and network devices patched.
        2. Use multifactor authentication and strong access controls.
        3. Monitor network logs and investigate unusual activity.
        4. Segment networks and restrict unnecessary ports.
        5. Maintain tested backups and an incident response plan.
        6. Validate machine learning alerts before taking action.
        """)

    with st.expander("Understanding model metrics"):
        st.markdown("""
        - **Accuracy:** Percentage of test records classified correctly.
        - **Precision:** How often predictions for a class are correct.
        - **Recall:** How many real records of a class are detected.
        - **F1-score:** A combined measure of precision and recall.
        - **Confusion matrix:** Shows correct and incorrect classifications.
        """)

    st.warning(
        "This educational tool is not a replacement for a production IDS, "
        "firewall, SIEM, or professional security investigation."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "CyberShield | Machine Learning Network Intrusion Detector | "
    "Educational project built with Python and Streamlit"
)
