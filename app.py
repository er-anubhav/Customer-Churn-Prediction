import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn & Segmentation Dashboard",
    page_icon="bar_chart",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"]          { background-color: #1a1d26; }
    .churn-yes { color: #ef4444; font-size: 1.8rem; font-weight: 800; }
    .churn-no  { color: #22c55e; font-size: 1.8rem; font-weight: 800; }
    .stat-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stat-label { color: #9ca3af; font-size: 0.85rem; }
    .stat-value { color: #ffffff; font-size: 1.5rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "Models")
DATA_PATH  = os.path.join(BASE_DIR, "Dataset", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── Load models & data ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "churn":  joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl")),
        "kmeans": joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl")),
        "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "meta":   joblib.load(os.path.join(MODEL_DIR, "metadata.pkl")),
    }

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Clean like the notebook
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    service_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]
    for c in service_cols:
        df[c] = df[c].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # Add cluster labels
    cluster_feats = ["tenure","MonthlyCharges","TotalCharges"]
    X_cluster = df[cluster_feats].copy()
    X_scaled = models["scaler"].transform(X_cluster)
    df["Cluster"] = models["kmeans"].predict(X_scaled)
    seg_map = models["meta"]["segment_names"]
    df["Segment"] = df["Cluster"].map(seg_map)

    # Encode Churn for prediction
    df["Churn_enc"] = (df["Churn"] == "Yes").astype(int)
    return df

models = load_models()
df = load_data()
meta = models["meta"]
segment_names = meta["segment_names"]
cluster_colors = {0: "#4f8ef7", 1: "#22c55e", 2: "#f59e0b", 3: "#ef4444"}
k_optimal = meta["k"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Telco Dashboard")
    st.markdown("---")
    st.subheader("Navigation")
    page = st.radio("Navigation", ["Segment Explorer", "Churn Predictor", "Business Insights"], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"**Model:** Logistic Regression (k={k_optimal})")
    st.caption(f"**Accuracy:** {meta['accuracy']:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Segment Explorer
# ══════════════════════════════════════════════════════════════════════════════
if page == "Segment Explorer":
    st.header("Customer Segment Explorer")

    col1, col2, col3, col4 = st.columns(4)
    seg_stats = df.groupby("Segment").agg(
        Customers=("Churn", "count"),
        ChurnRate=("Churn", lambda x: (x == "Yes").mean()),
        AvgTenure=("tenure", "mean"),
        AvgMonthly=("MonthlyCharges", "mean"),
    ).round(2)

    for col_idx, (seg_name, row) in enumerate(seg_stats.iterrows()):
        cluster_id = [k for k, v in segment_names.items() if v == seg_name][0]
        color = cluster_colors[cluster_id]
        with [col1, col2, col3, col4][col_idx]:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-label" style="color:{color};font-weight:700;">{seg_name}</div>'
                f'<div class="stat-value">{int(row["Customers"])}</div>'
                f'<div class="stat-label">Customers</div>'
                f'<div class="stat-label">Churn: <b>{row["ChurnRate"]:.1%}</b> | Tenure: {row["AvgTenure"]:.0f}m | ${row["AvgMonthly"]:.0f}/mo</div>'
                f'</div>', unsafe_allow_html=True
            )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Churn Rate by Segment")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f1117")
        ax.set_facecolor("#1e2130")
        churn_rates = df.groupby("Cluster")["Churn"].apply(lambda x: (x == "Yes").mean()) * 100
        colors_bar = [cluster_colors[i] for i in range(k_optimal)]
        bars = ax.bar(range(k_optimal), churn_rates.values, color=colors_bar, edgecolor="none")
        ax.set_xticks(range(k_optimal))
        ax.set_xticklabels([segment_names[i] for i in range(k_optimal)], rotation=15, color="white")
        ax.set_ylabel("Churn Rate (%)", color="white")
        ax.tick_params(colors="white")
        for bar, val in zip(bars, churn_rates.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontweight="bold", color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Segment Size Distribution")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f1117")
        ax.set_facecolor("#1e2130")
        sizes = df.groupby("Cluster").size()
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct="%1.1f%%",
            colors=[cluster_colors[i] for i in range(k_optimal)],
            startangle=90, textprops={"color": "white", "fontweight": "bold"}
        )
        ax.legend(
            [segment_names[i] for i in range(k_optimal)],
            loc="upper right", facecolor="#1e2130", labelcolor="white"
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Segment Profile Table")
    profile = df.groupby(["Cluster", "Segment"]).agg({
        "tenure": "mean",
        "MonthlyCharges": "mean",
        "TotalCharges": "mean",
        "Churn": lambda x: (x == "Yes").mean(),
    }).round(2)
    profile.columns = ["Avg Tenure", "Avg Monthly $", "Avg Total $", "Churn Rate"]
    profile["Churn Rate"] = (profile["Churn Rate"] * 100).round(1).astype(str) + "%"
    profile["Customers"] = df.groupby(["Cluster", "Segment"]).size().values
    st.dataframe(profile, use_container_width=True)

    st.markdown("---")
    st.subheader("Segment Explorer: Tenure vs Monthly Charges")
    filter_seg = st.multiselect(
        "Filter segments to display:",
        options=[segment_names[i] for i in range(k_optimal)],
        default=[segment_names[i] for i in range(k_optimal)]
    )
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1117")
    ax.set_facecolor("#1e2130")
    for cluster_id in range(k_optimal):
        seg_name = segment_names[cluster_id]
        if seg_name in filter_seg:
            subset = df[df["Cluster"] == cluster_id]
            ax.scatter(subset["tenure"], subset["MonthlyCharges"],
                       c=cluster_colors[cluster_id], label=seg_name,
                       alpha=0.5, edgecolors="none", s=20)
    ax.set_xlabel("Tenure (months)", color="white")
    ax.set_ylabel("Monthly Charges ($)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2130", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Churn Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Churn Predictor":
    st.header("Customer Churn Predictor")
    st.markdown("Fill in the customer profile below to see the churn prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["No", "Yes"])

    with col2:
        multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_back = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)

    total_charges = monthly * tenure

    # ── Build input matching the model's exact feature set ─────────────────
    # The notebook uses pd.get_dummies(drop_first=True) AFTER standardizing
    # 'No phone service' / 'No internet service' to 'No'. So only _Yes variants
    # and multi-category columns (InternetService, Contract, PaymentMethod)
    # have their non-reference levels.

    # Step 1: clean the selectbox values like the notebook does
    def _clean(x):
        if x in ("No phone service", "No internet service"):
            return "No"
        return x

    multi_lines = _clean(multi_lines)
    online_sec  = _clean(online_sec)
    online_back = _clean(online_back)
    device_prot = _clean(device_prot)
    tech_support = _clean(tech_support)
    streaming_tv = _clean(streaming_tv)
    streaming_mov = _clean(streaming_mov)

    # Step 2: compute cluster
    cluster_input = np.array([[tenure, monthly, total_charges]])
    cluster_scaled = models["scaler"].transform(cluster_input)
    pred_cluster = models["kmeans"].predict(cluster_scaled)[0]

    # Step 3: build the exact 27-feature dictionary
    input_dict = {
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges,
        "Cluster": pred_cluster,
        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "PhoneService_Yes": 1 if phone == "Yes" else 0,
        "MultipleLines_Yes": 1 if multi_lines == "Yes" else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if internet == "No" else 0,
        "OnlineSecurity_Yes": 1 if online_sec == "Yes" else 0,
        "OnlineBackup_Yes": 1 if online_back == "Yes" else 0,
        "DeviceProtection_Yes": 1 if device_prot == "Yes" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies_Yes": 1 if streaming_mov == "Yes" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
        "Segment_Low-Engagement New": 1 if pred_cluster == 1 else 0,
        "Segment_Mid-Value At-Risk":   1 if pred_cluster == 3 else 0,
        "Segment_Premium Short-Term":  1 if pred_cluster == 2 else 0,
    }

    expected_features = meta["features"]
    input_df = pd.DataFrame([input_dict])[expected_features]

    st.markdown("---")
    if st.button("Predict Churn", type="primary", width="stretch"):
        churn_model = models["churn"]
        prob = churn_model.predict_proba(input_df)[0]
        pred = churn_model.predict(input_df)[0]
        pred_segment = segment_names[pred_cluster]

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown("### Verdict")
            if pred == 1:
                st.markdown('<p class="churn-yes">At Risk of Churning</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="churn-no">Likely to Stay</p>', unsafe_allow_html=True)

            st.markdown(f"**Churn Probability:** {prob[1]*100:.1f}%")
            st.progress(float(prob[1]))
            st.markdown(f"**Stay Probability:** {prob[0]*100:.1f}%")
            st.progress(float(prob[0]))

        with res_col2:
            st.markdown("### Customer Segment")
            st.markdown(
                f'<div style="background-color:{cluster_colors[pred_cluster]};'
                f'border-radius:8px;padding:20px;text-align:center;">'
                f'<span style="color:white;font-size:1.3rem;font-weight:700;">{pred_segment}</span>'
                f'</div>', unsafe_allow_html=True
            )
            segment_churn = df[df["Cluster"] == pred_cluster]["Churn"].value_counts(normalize=True)
            churn_pct = segment_churn.get("Yes", 0) * 100
            st.markdown(f"**Segment churn rate:** {churn_pct:.1f}%")
            st.markdown(f"**Assigned to cluster:** {pred_cluster}")

        with res_col3:
            st.markdown("### Customer Snapshot")
            st.markdown(f"- Tenure: {tenure}m | ${monthly:.0f}/mo")
            st.markdown(f"- Contract: {contract}")
            st.markdown(f"- Internet: {internet}")
            st.markdown(f"- Payment: {payment.split('(')[0].strip()}")
            st.markdown(f"- Support: {tech_support}")

        # Recommendations based on segment
        st.markdown("---")
        st.subheader("Recommended Actions")
        recs = {
            "High-Value Loyal": (
                "Maintain satisfaction with loyalty perks. "
                "Low churn risk — focus on upsell opportunities."
            ),
            "Low-Engagement New": (
                "Proactive onboarding in first 90 days. "
                "Offer introductory bundles and educational content."
            ),
            "Premium Short-Term": (
                "Convert to long-term contract with incentives. "
                "Highest ROI for retention spend."
            ),
            "Mid-Value At-Risk": (
                "Urgent retention intervention needed. "
                "Survey pain points and offer targeted discounts."
            ),
        }
        st.info(recs.get(pred_segment, "Monitor customer activity."))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Business Insights
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.header("Business Insights & Recommendations")

    st.subheader("Segment Overview")
    st.markdown("""
    | Segment | Characteristics | Churn Risk | Strategy |
    |---------|----------------|-----------|----------|
    | **High-Value Loyal** | Long tenure, high spend, low churn | Low | Reward and retain with perks |
    | **Low-Engagement New** | Short tenure, low spend | Medium | Onboard and educate early |
    | **Premium Short-Term** | High monthly, short tenure | Medium-High | Convert to long-term |
    | **Mid-Value At-Risk** | Moderate spend, high churn | High | Urgent intervention |
    """)

    st.markdown("---")
    st.subheader("Top Churn Drivers (Model Coefficients)")
    churn_model = models["churn"]
    expected_features = meta["features"]
    coef_df = pd.DataFrame({
        "Feature": expected_features,
        "Coefficient": churn_model.coef_[0]
    }).sort_values("Coefficient", ascending=False)

    top_pos = coef_df.head(5)
    top_neg = coef_df.tail(5)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Factors that increase churn risk**")
        for _, row in top_pos.iterrows():
            feat = row["Feature"].replace("_Yes", "").replace("_", " ")
            st.markdown(f"- {feat} (+{row['Coefficient']:.3f})")

    with col_b:
        st.markdown("**Factors that decrease churn risk**")
        for _, row in top_neg.iterrows():
            feat = row["Feature"].replace("_Yes", "").replace("_", " ")
            st.markdown(f"- {feat} ({row['Coefficient']:.3f})")

    st.markdown("---")
    st.subheader("Key Takeaways")
    st.markdown("""
    1. **Month-to-month contracts** are the strongest churn driver — convert to annual plans
    2. **Fiber optic customers** churn more — may indicate service quality issues
    3. **Paperless billing + electronic check** customers churn more — simplify payment experience
    4. **Long tenure + tech support** strongly reduce churn — invest in support quality
    5. **No dependents/partner** customers churn more — build community features
    """)

    st.markdown("---")
    st.subheader("Action Plan by Segment")
    st.markdown("""
    ### High-Value Loyal
    - **Goal:** Maintain satisfaction, drive upsell
    - **Actions:** VIP support, loyalty discounts, referral program
    - **Budget:** Low (minimal retention spend needed)

    ### Low-Engagement New
    - **Goal:** Increase engagement in first 90 days
    - **Actions:** Welcome campaign, usage tips, bundled offers
    - **Budget:** Medium (high ROI if retained)

    ### Premium Short-Term
    - **Goal:** Lock in with long-term contracts
    - **Actions:** Annual plan discounts, loyalty points, priority service
    - **Budget:** High (highest revenue at risk)

    ### Mid-Value At-Risk
    - **Goal:** Prevent churn immediately
    - **Actions:** Win-back offer, satisfaction survey, service review call
    - **Budget:** High (urgent intervention required)
    """)
