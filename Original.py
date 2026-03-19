import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import random

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Awareness & Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# PROFESSIONAL STYLING
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    body {background-color: #fdfdfd; font-family: 'Helvetica', sans-serif;}
    h1, h2, h3 {color: #ad1457;}
    .stButton>button {
        background-color: #d81b60;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-size: 1.05em;
        font-weight: 600;
    }
    .metric-container {
        background-color: #fff0f5;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f8f9fa;
        padding: 14px;
        text-align: center;
        font-size: 0.96em;
        color: #444;
        border-top: 1px solid #e0e0e0;
    }
    .trust-badge {
        background-color: #e8f5e9;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.9em;
        color: #2e7d32;
        display: inline-block;
        margin: 8px;
    }
    .contact-card {
        background-color: #fff0f5;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        max-width: 600px;
        margin: 0 auto 30px auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .contact-icon {
        font-size: 2.2em;
        margin-bottom: 12px;
    }
    .contact-link {
        font-size: 1.3em;
        color: #d81b60;
        text-decoration: none;
        font-weight: 500;
    }
    .contact-link:hover {
        text-decoration: underline;
        color: #b71c5c;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# LOAD DATA & MODEL
# ────────────────────────────────────────────────
@st.cache_resource
def load_data_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return X, y, model, accuracy, y_test, y_pred, data.target_names

X, y, model, accuracy, y_test, y_pred, target_names = load_data_model()

# ────────────────────────────────────────────────
# SIDEBAR NAVIGATION – now includes Contact
# ────────────────────────────────────────────────
st.sidebar.image("https://1000logos.net/wp-content/uploads/2022/04/Breast-Cancer-Logo.png", width=220)
st.sidebar.title("Breast Cancer Awareness")

page = st.sidebar.radio("Navigation", [
    "Home",
    "Signs & Awareness",
    "Upload Scan Result",
    "Live Breast Scan (Camera)",
    "Find Nearby Services",
    "Data Insights",
    "Model Evaluation",
    "Feature Prediction",
    "Contact"
])

# ==============================
# HOME PAGE
# ==============================
if page == "Home":
    st.title("🩺 Breast Cancer Detection and Awareness")
    st.markdown("**A Machine Learning-powered application to detect breast cancer early. Knowledge empowers. Early action saves lives**.")
   
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader("Purpose of This Tool")
        st.write("""
        This professional hospital-grade application uses the renowned UCI Breast Cancer Wisconsin dataset
        and Random Forest AI to help you understand tumor classification.
        Built with care to educate and empower — always pair with professional medical care.
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Breast_cancer_cells.jpg",
                 caption="Microscopic View of Breast Cancer Cells", use_container_width=True)
       
        st.markdown('<div class="trust-badge">Backed by UCI Machine Learning Repository • Used in 1000+ research papers</div>', unsafe_allow_html=True)
       
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Samples", X.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Features", X.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
  
    st.markdown("---")
    st.subheader("Community Awareness & Hope")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Pink_ribbon.svg/1200px-Pink_ribbon.svg.png",
             caption="Symbol of Hope & Awareness", use_container_width=True)
    st.image("https://www.nationalbreastcancer.org/wp-content/uploads/breast-cancer-awareness-month-diverse-women.jpg",
             caption="Diverse women united in strength and support", use_container_width=True)

# ==============================
# SIGNS & AWARENESS (your existing content – kept as-is)
# ==============================
elif page == "Signs & Awareness":
    st.title("Recognizing Potential Signs")
    st.write("Educational examples of early warning signs in breast health. Always consult a doctor for any changes you notice — early detection saves lives.")

    st.subheader("Clinical Breast Examination by a Healthcare Professional")
    st.write("A trained doctor or nurse performs this exam during routine check-ups. It is quick, private, and an important step in early detection. Here are real examples from trusted health resources:")
    col_exam1, col_exam2 = st.columns(2)
    with col_exam1:
        st.image(
            "https://stellamattina.com/wp-content/uploads/2025/12/stella-breast-cancer-check-up.jpg",
            caption="Doctor performing a gentle clinical breast exam (professional and respectful setting)",
            use_container_width=True
        )
    with col_exam2:
        st.image(
            "https://ysm-res.cloudinary.com/image/upload/ar_16:9,c_fill,dpr_3.0,f_auto,g_faces:auto,q_auto:eco,w_500/v1/yms/prod/bbca6c22-f60e-48e3-9036-b76c2ddf26a8",
            caption="Healthcare provider guiding a patient through a breast health consultation and exam",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Visible Changes – Skin Dimpling / Orange-Peel Texture & Mammogram Findings")
    col_left, col_right = st.columns(2)
    with col_left:
        st.image(
            "https://www.breastcancer.org/sites/default/files/styles/hero_image/public/2023-10/skin-dimple.jpg?itok=3Z3y0f0D",
            caption="Skin dimpling – one of the changes worth discussing with your doctor",
            use_container_width=True
        )
        st.image(
            "https://www.breastcancer.org/sites/default/files/styles/hero_image/public/2023-10/peau-d-orange.jpg?itok=9k4zL5zL",
            caption="Peau d'orange appearance (orange-peel texture) of the breast skin",
            use_container_width=True
        )
    with col_right:
        st.image(
            "https://www.uclahealth.org/sites/default/files/styles/max_width_024000_960/public/images/9a/case-architectural-distortion-fig1.jpg?f=78a9272b&itok=KCsS5Jmp",
            caption="Mammogram highlighting architectural distortion (circled for educational clarity)",
            use_container_width=True
        )
        st.image(
            "https://radiologybusiness.com/sites/default/files/2022-03/breast_architectual_distortion_ajr.jpg",
            caption="Screening mammogram showing architectural distortion – seek evaluation if seen",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Real Stories – Strength from Breast Cancer Survivors")
    st.write("These are real women who have faced breast cancer and are now advocates for awareness, early detection, and support. Their stories remind us that hope, treatment, and community make a difference.")
    col_surv1, col_surv2, col_surv3 = st.columns(3)
    with col_surv1:
        st.image(
            "https://i0.wp.com/empoweratlantamagazine.com/wp-content/uploads/2025/11/AdobeStock_948596223.jpg?fit=5824%2C3264&ssl=1",
            caption="Diverse group of breast cancer survivors standing together in solidarity and strength",
            use_container_width=True
        )
    with col_surv2:
        st.image(
            "https://ysm-res.cloudinary.com/image/upload/ar_16:9,c_fill,dpr_3.0,f_auto,g_faces:auto,q_auto:eco,w_500/v1/yms/prod/bbca6c22-f60e-48e3-9036-b76c2ddf26a8",
            caption="Breast cancer survivor with her family – celebrating life after treatment",
            use_container_width=True
        )
    with col_surv3:
        st.image(
            "https://empoweratlantamagazine.com/wp-content/uploads/2025/11/AdobeStock_948596223.jpg",
            caption="Community of survivors in pink, united for awareness and hope",
            use_container_width=True
        )

    st.subheader("Symbol of Hope & Support")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Pink_ribbon.svg/1200px-Pink_ribbon.svg.png",
        caption="Pink Ribbon – Global Symbol of Breast Cancer Awareness and Solidarity",
        use_container_width=True
    )
    

    st.info("**All images are sourced from trusted medical organizations, awareness campaigns, and survivor stories.** They are shared for early detection and awareness only. Use our system for testing and if you have further concerns about your breast health, please visit a healthcare provider — facilities like CLEAR Radiology for confirmation and medication.")

# ==============================
# UPLOAD SCAN RESULT
# ==============================
elif page == "Upload Scan Result":
    st.title("Upload Scan Result (Educational Demo)")
    st.write("Upload a mammogram, ultrasound, or saved breast scan image. This is for visualization and learning — no real diagnosis is performed.")
   
    uploaded = st.file_uploader("Choose a scan image (JPG/PNG)", type=["jpg","jpeg","png"])
   
    if uploaded is not None:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Your Uploaded Scan", use_container_width=True)
            st.info("Image successfully loaded. In real medical practice, a qualified radiologist reviews such images.")
        except Exception as e:
            st.error(f"Error loading image: {e}")
        st.warning("This site is for awareness and early detection. Please consult a healthcare professional for further diagnosis and treatment if tested positive here. Go ahead and input your data at the FEATURE EVALUATION page below and predict for more accuracy.")

# ==============================
# LIVE BREAST SCAN (CAMERA)
# ==============================
elif page == "Live Breast Scan (Camera)":
    st.title("Live Breast Scan (Camera Capture)")
    st.write("Use your webcam (desktop) or phone camera (mobile) to take a photo. This is an educational simulation — allow camera access when prompted.")
   
    picture = st.camera_input("Take or capture a breast area photo(highly protected,we don't have access to your photo)")
   
    if picture is not None:
        st.image(picture, caption="Captured Image from Camera", use_container_width=True)
       
        if st.button("Simulate AI Analysis"):
            result = random.choice(["Benign appearance", "Potential concern"])
           
            if "concern" in result:
                st.error("**Simulated Result: No potential concern detected, below average detected.**")
                st.write("**Educational note:** This does not suggests a hundred percent accuracy you can still consult a doctor promptly for proper evaluation if symptoms persist.")
                st.info("**Health Tip:** Early professional screening (mammogram/ultrasound) is the gold standard.")
            else:
                st.success("**Simulated Result: Benign / Normal appearance**")
                st.info("**Health Tip:** Maintain regular self-checks and schedule routine treatment as recommended by your doctor.")
           
            st.warning("**Important:** Medical analysis or diagnosis. Most camera photos are not always accurate for breast health assessment. Always do all the other checks then consult a specialist if seen positive, inpute your breast health data at the FEATURE EVALUATION page below and predict for further accuracy.")

# ==============================
# FIND NEARBY SERVICES
# ==============================
elif page == "Find Nearby Services":
    st.title("Find Nearby Breast Health Services in Buea")
    st.write("Key facilities and organizations in/near Buea that offer or support breast cancer screening, awareness, and care (as of 2026). Always confirm current services directly.")

    st.info("This is detection and awareness only — if symptoms are positive Call a specialist immediately and book an appointment or reach out to us for guidance.")

    services = [
        {
            "name": "CLEAR Radiology Buea",
            "desc": "Digital mammography, 3D ultrasound, breast screening, biopsy, cancer care consultations.",
            "loc": "Mile 17, Buea (behind CNPS)",
            "contact": "+237 6 82 86 57 93",
            "note": "Main private center for breast imaging in Buea"
        },
        {
            "name": "Buea Regional Hospital",
            "desc": "Public hospital – clinical breast exams, ultrasound, referrals, treatment pathways.",
            "loc": "Buea town center",
            "contact": "Local directory or Ministry of Public Health",
            "note": "Primary public referral hospital"
        },
        {
            "name": "Hadassah Foundation",
            "desc": "NGO – breast cancer awareness, education, community support.",
            "loc": "Bakweri Town, Buea",
            "contact": "+237 6 71 36 41 33",
            "note": "Active in Pink October campaigns"
        }
    ]

    for item in services:
        st.markdown(f"""
        **{item['name']}**  
        {item['desc']}  
        **Location:** {item['loc']}  
        **Contact:** {item['contact']}  
        *{item['note']}*
        ---
        """)

# ==============================
# DATA INSIGHTS
# ==============================
elif page == "Data Insights":
    st.title("Dataset Overview & Visualizations")
    df = pd.concat([X, pd.Series(y, name="target").map({0: "Malignant", 1: "Benign"})], axis=1)
  
    st.subheader("First 5 Records")
    st.dataframe(df.head())
  
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Diagnosis Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="target", palette="Set2", ax=ax)
        ax.set_xticklabels(["Malignant","Benign"])
        st.pyplot(fig)
    with col2:
        st.subheader("Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

# ==============================
# MODEL EVALUATION
# ==============================
elif page == "Model Evaluation":
    st.title("Random Forest Model Evaluation")
  
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
  
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=target_names)
    st.text(report)

# ==============================
# FEATURE PREDICTION
# ==============================
elif page == "Feature Prediction":
    st.title("Tumor Feature Prediction (Demo)")
    st.write("Input tumor measurements to predict whether it is benign or malignant.")
  
    with st.form("predict_form"):
        input_vals = []
        cols = st.columns(3)
        for idx, col_name in enumerate(X.columns):
            with cols[idx % 3]:
                val = st.number_input(col_name, float(X[col_name].min()), float(X[col_name].max()), float(X[col_name].mean()), step=0.01)
                input_vals.append(val)
        submit = st.form_submit_button("Predict")
  
    if submit:
        arr = np.array(input_vals).reshape(1,-1)
        pred = model.predict(arr)[0]
        probs = model.predict_proba(arr)[0]
       
        if pred == 0:
            st.error("**Direct Result: Cancerous (Malignant Tumor Detected)**")
            stage = random.choice(["Stage I (Early)", "Stage II"])
            st.write(f"**Estimated Stage (demo):** {stage}")
            st.info("**Health Advice:** Please consult a doctor immediately. Early detection gives the best chance of successful treatment.")
        else:
            st.success("**Direct Result: Non-Cancerous (Benign)**")
            st.write("**Estimated Stage (demo):** Not applicable")
            st.info("**Health Advice:** Great result in this demo! Continue regular check-ups and healthy lifestyle habits.")
       
        st.write(f"Benign probability: {probs[1]*100:.1f}% | Malignant probability: {probs[0]*100:.1f}%")

# ==============================
# NEW – CONTACT PAGE
# ==============================
elif page == "Contact":
    st.title("📬 Contact Us")
    st.write("We're here to answer your questions about the app, breast cancer awareness, or how you can get involved anywhere you are.")

    st.markdown("""
    <div class="contact-card">
        <div class="contact-icon">📞</div>
        <a href="tel:+237652576134" class="contact-link">+237 652 576 134</a>
        <p style="margin: 8px 0 24px 0;">Call or WhatsApp us</p>

        <div class="contact-icon">✉️</div>
        <a href="mailto:ngilladiamod@gmail.com" class="contact-link">ngilladiamod@gmail.com</a>
        <p style="margin: 8px 0 0 0;">Send us an email</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info(".")

# ==============================
# FOOTER
# ==============================
st.markdown(""""
<div class="footer">
Hospital-Style Professional App • Breast Cancer Awareness • ML-Powered Detection • Always consult healthcare professionals • Supervised by Engr BADU ZAMANI
</div>
""", unsafe_allow_html=True)