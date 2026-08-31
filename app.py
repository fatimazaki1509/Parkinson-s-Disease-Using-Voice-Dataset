import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import joblib
from xgboost import XGBClassifier

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Parkinson Voice Detection",
    page_icon="🎤",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main{
    background-color:#f8fafc;
}

.title{
    text-align:center;
    font-size:40px;
    font-weight:700;
    color:#0f172a;
}

.subtitle{
    text-align:center;
    color:#475569;
    margin-bottom:30px;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:12px;
    text-align:center;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);
}

.result-success{
    background:#dcfce7;
    padding:20px;
    border-radius:12px;
    color:#166534;
    font-size:20px;
    font-weight:bold;
}

.result-danger{
    background:#fee2e2;
    padding:20px;
    border-radius:12px;
    color:#991b1b;
    font-size:20px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = XGBClassifier()
model.load_model("parkinson_model.json")

scaler = joblib.load("scaler.pkl")

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    return np.mean(mfcc.T, axis=0)

# =========================
# PREDICTION
# =========================

def predict_audio(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    features = extract_features(temp_path)

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    confidence = (
        np.max(model.predict_proba(features))
        * 100
    )

    return prediction, confidence, temp_path

# =========================
# HEADER
# =========================

st.markdown(
    '<div class="title">🎤 Parkinson Voice Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Detection Using Voice Signal Analysis & XGBoost</div>',
    unsafe_allow_html=True
)

# =========================
# STATS
# =========================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='metric-card'>
    <h3>Model</h3>
    <h2>XGBoost</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='metric-card'>
    <h3>Accuracy</h3>
    <h2>98.23%</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-card'>
    <h3>Dataset</h3>
    <h2>1134 Samples</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload WAV Audio File",
    type=["wav"]
)

# =========================
# PROCESS
# =========================

if uploaded_file is not None:

    prediction, confidence, path = predict_audio(
        uploaded_file
    )

    audio, sr = librosa.load(path, sr=None)

    left, right = st.columns([1,1])

    with left:

        st.subheader("🎧 Uploaded Audio")
        st.audio(uploaded_file)

        st.write(
            f"Sampling Rate: **{sr} Hz**"
        )

        st.write(
            f"Duration: **{round(len(audio)/sr,2)} sec**"
        )

    with right:

        st.subheader("🩺 Prediction Result")

        if prediction == 1:

            st.markdown(
                f"""
                <div class='result-danger'>
                ⚠ Parkinson's Disease Detected
                <br><br>
                Confidence: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class='result-success'>
                ✅ Healthy Voice
                <br><br>
                Confidence: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(min(int(confidence),100))

    st.markdown("---")

    # =========================
    # WAVEFORM
    # =========================

    st.subheader("📈 Voice Waveform")

    fig, ax = plt.subplots(figsize=(10,3))

    librosa.display.waveshow(
        audio,
        sr=sr,
        ax=ax
    )

    ax.set_title("Audio Waveform")

    st.pyplot(fig)

    # =========================
    # SPECTROGRAM
    # =========================

    st.subheader("🎼 Spectrogram")

    X = librosa.stft(audio)

    Xdb = librosa.amplitude_to_db(
        np.abs(X)
    )

    fig2, ax2 = plt.subplots(figsize=(10,4))

    img = librosa.display.specshow(
        Xdb,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=ax2
    )

    fig2.colorbar(
        img,
        ax=ax2,
        format='%+2.0f dB'
    )

    ax2.set_title(
        "Audio Spectrogram"
    )

    st.pyplot(fig2)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Final Year Project | Parkinson Disease Detection using Voice Signals & Machine Learning"
)