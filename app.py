import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Page Settings
st.set_page_config(page_title="Acoustic Research & Classification", layout="wide")

def analyze_audio(file):
    try:
        y, sr = librosa.load(file, sr=None)
        y = y / (np.max(np.abs(y)) + 1e-8)
        frame_len = int(0.03 * sr)
        hop_len = int(0.015 * sr)
        frames = [y[i:i+frame_len] for i in range(0, len(y)-frame_len, hop_len)]
        f0_values = []
        energies = [np.sum(f**2)/len(f) for f in frames]
        energy_thresh = 0.2 * np.max(energies)
        f0_track = []
        for frame in frames:
            energy = np.sum(frame**2)/len(frame)
            if energy > energy_thresh:
                f_det = frame - np.mean(frame)
                corr = correlate(f_det, f_det, mode='full')[len(frame)-1:]
                low, high = int(sr/500), int(sr/75)
                if len(corr) > high:
                    peak = np.argmax(corr[low:high]) + low
                    freq = sr / peak
                    f0_values.append(freq)
                    f0_track.append(freq)
                else: f0_track.append(np.nan)
            else: f0_track.append(np.nan)
        return (np.mean(f0_values) if f0_values else 0), y, sr, f0_track
    except: return 0, None, 0, []

def get_meta_from_name(fname):
    parts = fname.replace(".wav", "").split("_")
    actual, age, emotion = "Unknown", "Unknown", "Unknown"
    if len(parts) >= 5:
        g_code = parts[2].upper()
        if g_code in ["M", "E"]: actual = "Male"
        elif g_code in ["F", "K"]: actual = "Female"
        elif g_code == "C": actual = "Child"
        age = parts[3]
        emotion = parts[4].capitalize()
    return actual, age, emotion

# --- UI HEADER ---
st.title("Acoustic Signal Analysis & Speaker Classification")
st.markdown("---")

# Sidebar
st.sidebar.header("Data Input")
uploaded_files = st.sidebar.file_uploader("Upload Audio Files", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    file_objects = {}
    correct_count = 0
    
    for file in uploaded_files:
        f0, sig, sr, track = analyze_audio(file)
        actual, age, emotion = get_meta_from_name(file.name)
        
        if f0 < 170: pred = "Male"
        elif f0 < 290: pred = "Female"
        else: pred = "Child"
        
        is_match = (pred == actual)
        if is_match: correct_count += 1
        
        results_list.append({
            "File Name": file.name,
            "Predicted": pred,
            "Actual": actual,
            "F0 Mean (Hz)": round(f0, 2),
            "Age": age,
            "Emotion": emotion,
            "Match": "✅" if is_match else "❌"
        })
        file_objects[file.name] = (f0, sig, sr, track, pred, actual, age, emotion)

    df = pd.DataFrame(results_list)
    current_accuracy = (correct_count / len(uploaded_files)) * 100

    # --- TOP METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Files", len(uploaded_files))
    m2.metric("Current Batch Accuracy", f"{current_accuracy:.2f}%")
    m3.metric("Project Goal Accuracy", "76.99%")
    m4.metric("Correct Predictions", f"{correct_count} / {len(uploaded_files)}")

    st.markdown("---")
    st.subheader("Batch Classification Breakdown")
    st.bar_chart(df['Predicted'].value_counts())
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Signal Visualization")
    selected_file = st.selectbox("Select a file to inspect:", options=list(file_objects.keys()))

    if selected_file:
        f0_s, sig_s, sr_s, track_s, pred_s, actual_s, age_s, emot_s = file_objects[selected_file]
        cl, cr = st.columns([2, 1])
        with cl:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            plt.subplots_adjust(hspace=0.3)
            ax1.plot(np.linspace(0, len(sig_s)/sr_s, len(sig_s)), sig_s, color='#34495e')
            ax1.set_title("Waveform")
            ax2.scatter(np.linspace(0, len(sig_s)/sr_s, len(track_s)), track_s, color='#e74c3c', s=2)
            ax2.set_title("Frequency Tracking (Hz)")
            st.pyplot(fig)
        with cr:
            # İSTEDİĞİN EKLEME BURADA:
            st.info(f"**Analysis Results**")
            st.write(f"**Calculated F0:** `{f0_s:.2f} Hz`")
            st.write(f"**Predicted Class:** `{pred_s}`")
            st.write(f"**Actual Gender:** `{actual_s}`")
            st.write(f"**Subject Age:** `{age_s}`")
            st.write(f"**Detected Emotion:** `{emot_s}`")
            st.write("---")
            if pred_s == actual_s:
                st.success("✅ Prediction Match")
            else:
                st.error("❌ Prediction Mismatch")
else:
    st.info("Please upload multiple .wav files to see the results and accuracy.")
    
    # streamlit run app.py çalıştırma
    # ctrl+c durdurma
