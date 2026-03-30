import streamlit as st
import numpy as np
import librosa
from scipy.signal import correlate
import tempfile

st.set_page_config(page_title="Gender Classification App", layout="centered")
st.title("Sound Signal Analysis and Gender Classification")

st.write("Bir WAV dosyası yükle. Sistem F0 hesaplayıp sınıf tahmini yapsın.")

def frame_signal(y, frame_length, hop_length):
    frames = []
    for i in range(0, len(y) - frame_length, hop_length):
        frames.append(y[i:i+frame_length])
    return frames

def short_time_energy(frame):
    return np.sum(frame ** 2) / len(frame)

def zero_crossing_rate(frame):
    return np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

def autocorr_f0(frame, sr, fmin=80, fmax=400):
    frame = frame - np.mean(frame)

    if np.all(frame == 0):
        return np.nan

    corr = correlate(frame, frame, mode="full")
    corr = corr[len(corr)//2:]

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    if max_lag >= len(corr) or min_lag >= max_lag:
        return np.nan

    corr_range = corr[min_lag:max_lag]

    if len(corr_range) == 0:
        return np.nan

    peak = np.argmax(corr_range) + min_lag

    if peak <= 0:
        return np.nan

    return sr / peak

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    if len(y) == 0:
        return np.nan, np.nan, np.nan, 0

    y = y / (np.max(np.abs(y)) + 1e-8)

    frame_length = int(0.03 * sr)   # 30 ms
    hop_length = int(0.015 * sr)    # 15 ms

    frames = frame_signal(y, frame_length, hop_length)

    if len(frames) == 0:
        return np.nan, np.nan, np.nan, sr

    energies = np.array([short_time_energy(f) for f in frames])
    zcrs = np.array([zero_crossing_rate(f) for f in frames])

    energy_threshold = 0.3 * np.max(energies)
    zcr_threshold = np.median(zcrs)

    voiced_idx = np.where((energies > energy_threshold) & (zcrs <= zcr_threshold))[0]

    f0_values = []
    for i in voiced_idx:
        f0 = autocorr_f0(frames[i], sr)
        if not np.isnan(f0):
            f0_values.append(f0)

    f0_mean = np.mean(f0_values) if len(f0_values) > 0 else np.nan
    zcr_mean = np.mean(zcrs) if len(zcrs) > 0 else np.nan
    energy_mean = np.mean(energies) if len(energies) > 0 else np.nan

    return f0_mean, zcr_mean, energy_mean, sr

def classify_rule(f0):
    if np.isnan(f0):
        return "Unknown"
    elif f0 < 180:
        return "Male"
    elif f0 < 300:
        return "Female"
    else:
        return "Child"

uploaded_file = st.file_uploader("WAV dosyası yükle", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    try:
        f0_mean, zcr_mean, energy_mean, sr = extract_features(temp_path)
        prediction = classify_rule(f0_mean)

        st.subheader("Sonuç")
        st.write(f"Tahmin edilen sınıf: **{prediction}**")
        st.write(f"Ortalama F0: **{f0_mean:.2f} Hz**" if not np.isnan(f0_mean) else "Ortalama F0: hesaplanamadı")
        st.write(f"Ortalama ZCR: **{zcr_mean:.4f}**" if not np.isnan(zcr_mean) else "Ortalama ZCR: hesaplanamadı")
        st.write(f"Ortalama Energy: **{energy_mean:.6f}**" if not np.isnan(energy_mean) else "Ortalama Energy: hesaplanamadı")
        st.write(f"Sampling Rate: **{sr} Hz**")

        if prediction == "Unknown":
            st.warning("Bu dosyada güvenilir F0 çıkarılamadı.")
        else:
            st.success("Analiz tamamlandı.")

    except Exception as e:
        st.error(f"Hata oluştu: {e}")

        ## python -m streamlit run app.py çalıştırmak için
        ## Ctrl+c durdurmak için
