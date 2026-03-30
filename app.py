import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Sayfa ayarları
st.set_page_config(page_title="Signal Analysis Pro", layout="wide")

def analyze_audio(file):
    y, sr = librosa.load(file, sr=None)
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    # 30ms Windowing [cite: 23]
    frame_len = int(0.03 * sr)
    hop_len = int(0.015 * sr)
    frames = [y[i:i+frame_len] for i in range(0, len(y)-frame_len, hop_len)]
    
    f0_values = []
    energies = [np.sum(f**2)/len(f) for f in frames]
    energy_thresh = 0.2 * np.max(energies)

    for frame in frames:
        if (np.sum(frame**2)/len(frame)) > energy_thresh:
            f_det = frame - np.mean(frame)
            # Autocorrelation Rτ = x[n]x[n-τ] [cite: 28, 29]
            corr = correlate(f_det, f_det, mode='full')[len(frame)-1:]
            low, high = int(sr/500), int(sr/75)
            if len(corr) > high:
                peak = np.argmax(corr[low:high]) + low
                f0_values.append(sr / peak)
    
    f0_mean = np.mean(f0_values) if f0_values else 0
    return f0_mean, y, sr

# --- UI TASARIMI ---
st.title("🎙️ Speech Analysis & Classification Dashboard")
st.markdown("---")

# Sol panel: Dosya Yükleme
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])
    st.info("Method: Autocorrelation (Time-Domain) [cite: 32]")

if uploaded_file:
    # Analizi çalıştır
    f0, signal, sr = analyze_audio(uploaded_file)
    
    # Sınıflandırma Mantığı (Senin Eşiklerin)
    if f0 < 170: 
        label, color = "MALE", "blue"
    elif f0 < 290: 
        label, color = "FEMALE", "pink"
    else: 
        label, color = "CHILD", "green"

    # Üst Kısım: Sonuç Kartları
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Class", label)
    col2.metric("Average F0", f"{f0:.2f} Hz")
    col3.metric("Sample Rate", f"{sr} Hz")

    st.markdown("---")

    # Orta Kısım: Grafik ve Tablo
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Waveform Analysis")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.linspace(0, len(signal)/sr, len(signal)), signal, color='gray')
        ax.set_title("Time Domain Signal")
        st.pyplot(fig)

    with c2:
        st.subheader("Analysis Summary")
        # Tablomsu yapı burada
        stats_data = {
            "Parameter": ["Fundamental Freq (F0)", "Classification", "Signal Status"],
            "Value": [f"{f0:.2f} Hz", label, "Processed"],
            "Status": ["✅", "🎯", "⚡"]
        }
        st.table(pd.DataFrame(stats_data))

    # Alt Kısım: Teknik Detay
    with st.expander("See Mathematical Details"):
        st.write("Fundamental frequency (F0) is calculated only on voiced regions[cite: 25, 40].")
        st.latex(r"R(\tau) = \sum_{n} x[n]x[n-\tau]") # Otokorelasyon formülü 

else:
    st.warning("Please upload a .wav file from the sidebar to begin analysis.")