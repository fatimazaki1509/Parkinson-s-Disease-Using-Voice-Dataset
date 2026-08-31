# 🎤 Parkinson Voice Detection System

### AI-Based Parkinson's Disease Detection Using Voice Signal Analysis & XGBoost

🔗 Live Demo: https://parkinson-s-disease-using-voice-dataset-ydhmo7em7hafdfwfklxczm.streamlit.app/

---

## 📌 Overview

Parkinson's Disease is a progressive neurological disorder that affects movement and speech patterns. This project leverages Machine Learning and Voice Signal Processing to identify potential Parkinson's symptoms from a user's voice sample.

The system extracts MFCC (Mel Frequency Cepstral Coefficients) features from uploaded audio recordings and uses a trained XGBoost model to classify whether the voice sample indicates Parkinson's Disease or a Healthy Condition.

The application is built using Streamlit and provides an intuitive web-based interface for real-time prediction.

---

##  Live Application

 **Try the App Here**

https://parkinson-s-disease-using-voice-dataset-ydhmo7em7hafdfwfklxczm.streamlit.app/

---

## ✨ Features

-  Upload WAV Voice Samples
-  Parkinson Detection using XGBoost
-  MFCC-based Audio Feature Extraction
-  Confidence Score Visualization
-  Audio Playback Support
-  Streamlit Cloud Deployment
- Real-Time Predictions
- User-Friendly Interface

---

##  Machine Learning Pipeline

### Step 1: Audio Upload

User uploads a `.wav` voice recording.

### Step 2: Feature Extraction

The system extracts:

- 13 MFCC Features
- Mean Feature Representation
- Audio Signal Characteristics

### Step 3: Data Preprocessing

- Feature Scaling using StandardScaler
- Input Transformation

### Step 4: Prediction

- XGBoost Classifier
- Binary Classification

Output:

- Healthy Voice
- Parkinson's Detected

### Step 5: Confidence Score

Prediction confidence is displayed as a percentage.

---

##  Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Frontend | Streamlit |
| Machine Learning | XGBoost |
| Audio Processing | Librosa |
| Data Handling | NumPy |
| Model Storage | Joblib |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

##  Project Structure

```bash
Parkinson-Voice-Detection/
│
├── app.py
├── parkinson_model.json
├── scaler.pkl
├── requirements.txt
├── README.md
│
└── assets/
```

---

##  Model Information

| Parameter | Value |
|------------|--------|
| Algorithm | XGBoost Classifier |
| Features | 13 MFCC Features |
| Accuracy | 98.23% |
| Dataset Samples | 1134 |
| Output Classes | Healthy / Parkinson's |

---

##  Installation

### Clone Repository

```bash
git clone https://github.com/fatimazaki1509/parkinson-s-disease-using-voice-dataset.git
```

### Navigate to Project

```bash
cd parkinson-s-disease-using-voice-dataset
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

##  How To Use

1. Open the application.
2. Upload a WAV audio file.
3. Wait for processing.
4. View prediction results.
5. Check confidence percentage.
6. Analyze the generated waveform.

---

##  Application Preview

### Home Screen

- Upload Audio File
- Model Statistics
- Real-Time Analysis

### Prediction Result

- Healthy Voice ✅
- Parkinson's Detected ⚠️
- Confidence Score 📈

---

##  Future Enhancements

- Live Voice Recording
- Deep Learning Models (CNN + LSTM)
- Spectrogram-Based Analysis
- Mobile Application
- Medical Report Generation
- Multi-Language Voice Support

---

##  Disclaimer

This project is developed for educational and research purposes only.

It is NOT a medical diagnostic tool and should not replace professional clinical evaluation.

---

##  Developer

**Fatima Zaki**

Final Year Engineering Project

---

##  Support

If you found this project useful:

⭐ Star this repository

 Fork the project

 Share it with others

---

### "AI for Early Detection, Better Healthcare & Smarter Diagnostics."
