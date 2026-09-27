"""
MedBot - Symptom Checker (Random Forest)
Jalankan dengan: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="MedBot - Symptom Checker",
    page_icon="🏥",
    layout="centered",
)

BINARY_COLS = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
CAT_COLS = ["Gender", "Blood Pressure", "Cholesterol Level"]


# =========================================================
# SYMPTOM CHECKER ENGINE
# =========================================================
class SymptomCheckerEngine:
    """Mengubah jawaban gejala pengguna menjadi prediksi Outcome + kandidat Disease."""

    def __init__(self, feature_cols, binary_cols, cat_cols, encoders,
                 target_le, disease_le, clf_outcome, clf_disease, top_k=3):
        self.feature_cols = feature_cols
        self.binary_cols = binary_cols
        self.cat_cols = cat_cols
        self.encoders = encoders
        self.target_le = target_le
        self.disease_le = disease_le
        self.clf_outcome = clf_outcome
        self.clf_disease = clf_disease
        self.top_k = top_k

    def _check_emergency(self, answers):
        if answers.get("Difficulty Breathing") == "Yes" and answers.get("Fever") == "Yes":
            return (
                "🚨 Kombinasi sesak napas dan demam bisa jadi tanda kondisi serius. "
                "Segera hubungi 119 atau ke IGD terdekat jika gejala memburuk!"
            )
        return None

    def _build_feature_row(self, answers):
        row = {}
        for c in self.binary_cols:
            row[c + "_enc"] = 1 if answers[c] == "Yes" else 0
        row["Age"] = answers["Age"]
        for c in self.cat_cols:
            le = self.encoders[c]
            val = answers[c] if answers[c] in le.classes_ else le.classes_[0]
            row[c + "_enc"] = le.transform([val])[0]
        return pd.DataFrame([row])[self.feature_cols]

    def predict(self, answers):
        emergency = self._check_emergency(answers)
        X_input = self._build_feature_row(answers)

        outcome_proba = self.clf_outcome.predict_proba(X_input)[0]
        outcome_idx = np.argmax(outcome_proba)
        outcome_label = self.target_le.inverse_transform([outcome_idx])[0]
        outcome_conf = outcome_proba[outcome_idx]

        disease_proba = self.clf_disease.predict_proba(X_input)[0]
        top_idx = np.argsort(disease_proba)[::-1][: self.top_k]
        top_diseases = [
            (self.disease_le.inverse_transform([i])[0], disease_proba[i]) for i in top_idx
        ]

        return {
            "emergency": emergency,
            "outcome_label": outcome_label,
            "outcome_conf": outcome_conf,
            "top_diseases": top_diseases,
        }

    def format_response(self, result):
        lines = []
        if result["emergency"]:
            lines.append(result["emergency"])
            lines.append("")

        lines.append(
            f"🔎 Prediksi kondisi: **{result['outcome_label']}** "
            f"(keyakinan {result['outcome_conf']*100:.1f}%)"
        )
        lines.append("")
        lines.append("🏥 3 kemungkinan penyakit teratas:")
        for name, prob in result["top_diseases"]:
            lines.append(f"- {name} — {prob*100:.1f}%")
        lines.append("")
        lines.append("⚠️ Ini bukan diagnosis medis. Untuk kepastian, konsultasikan ke dokter.")
        return "\n".join(lines)


# =========================================================
# LOAD DATA & TRAIN MODEL (di-cache supaya tidak retrain tiap interaksi)
# =========================================================
@st.cache_resource(show_spinner="Melatih model, mohon tunggu...")
def load_and_train(csv_path="Disease_symptom_and_patient_profile_dataset.csv"):
    df = pd.read_csv(csv_path)

    for c in BINARY_COLS:
        df[c + "_enc"] = df[c].map({"Yes": 1, "No": 0})

    encoders = {}
    for c in CAT_COLS:
        le = LabelEncoder()
        df[c + "_enc"] = le.fit_transform(df[c])
        encoders[c] = le

    target_le = LabelEncoder()
    df["Outcome_enc"] = target_le.fit_transform(df["Outcome Variable"])

    disease_le = LabelEncoder()
    df["Disease_enc"] = disease_le.fit_transform(df["Disease"])

    feature_cols = [c + "_enc" for c in BINARY_COLS] + ["Age"] + [c + "_enc" for c in CAT_COLS]
    X = df[feature_cols]

    # Model 1: prediksi Outcome (Positive/Negative)
    y_outcome = df["Outcome_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome
    )
    clf_outcome = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf_outcome.fit(X_train, y_train)

    # Model 2: prediksi Disease (top-k kandidat, dilatih di seluruh data)
    y_disease = df["Disease_enc"]
    clf_disease = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf_disease.fit(X, y_disease)

    engine = SymptomCheckerEngine(
        feature_cols, BINARY_COLS, CAT_COLS, encoders,
        target_le, disease_le, clf_outcome, clf_disease,
    )
    return engine, df


engine, df = load_and_train()


# =========================================================
# TAMPILAN
# =========================================================
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1a1f5e, #2d3a8c, #1565c0);
                padding:20px; border-radius:15px; color:white;">
        <h2 style="margin:0;">🏥 MedBot — Symptom Checker</h2>
        <p style="margin:5px 0 0 0;">🟢 Online · Isi gejala kamu untuk melihat kemungkinan kondisi</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background:#fff8e1; padding:10px; font-size:13px;
                border-left:4px solid orange; margin-top:10px; border-radius:4px;">
        ⚠️ Edukasi saja. Bukan pengganti dokter. Darurat: <b>119</b>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# --- inisialisasi riwayat chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- form input gejala ---
with st.form("gejala_form", clear_on_submit=False):
    st.subheader("Isi gejala kamu")

    col1, col2 = st.columns(2)
    with col1:
        fever = st.radio("Demam?", ["No", "Yes"], horizontal=True)
        fatigue = st.radio("Kelelahan?", ["No", "Yes"], horizontal=True)
    with col2:
        cough = st.radio("Batuk?", ["No", "Yes"], horizontal=True)
        breathing = st.radio("Sesak Napas?", ["No", "Yes"], horizontal=True)

    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Usia", min_value=0, max_value=120, value=25, step=1)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    with col4:
        bp = st.radio("Tekanan Darah", ["Low", "Normal", "High"], horizontal=True)
        chol = st.radio("Kolesterol", ["Low", "Normal", "High"], horizontal=True)

    submitted = st.form_submit_button("🔍 Cek Sekarang", use_container_width=True)

if submitted:
    answers = {
        "Fever": fever,
        "Cough": cough,
        "Fatigue": fatigue,
        "Difficulty Breathing": breathing,
        "Age": age,
        "Gender": gender,
        "Blood Pressure": bp,
        "Cholesterol Level": chol,
    }
    result = engine.predict(answers)
    response = engine.format_response(result)

    user_msg = (
        f"Demam={fever}, Batuk={cough}, Kelelahan={fatigue}, Sesak Napas={breathing}, "
        f"Usia={age}, Gender={gender}, TD={bp}, Kolesterol={chol}"
    )

    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- tombol clear ---
if st.session_state.messages:
    if st.button("🗑️ Clear riwayat"):
        st.session_state.messages = []
        st.rerun()

st.divider()
st.subheader("Riwayat Pemeriksaan")

if not st.session_state.messages:
    st.caption("Belum ada pemeriksaan. Isi form di atas lalu klik Cek Sekarang.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
