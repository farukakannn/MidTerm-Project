import os
import glob
import numpy as np
import pandas as pd
import librosa
from scipy.signal import correlate

# 1. KONUM VE EXCEL YÜKLEME
dataset_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dataset_path)

excel_files = glob.glob(os.path.join(dataset_path, "*.xlsx"))
df = pd.read_excel(excel_files[0])
df.columns = df.columns.str.strip()

# Sütunları Tespit Et
file_col = next((c for c in df.columns if 'file' in c.lower() or 'name' in c.lower()), df.columns[0])
gender_col = next((c for c in df.columns if 'gender' in c.lower() or 'cinsiyet' in c.lower()), df.columns[1])

# 2. WAV DOSYALARINI HARİTALAMA
wav_files = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
wav_map = {os.path.basename(f).lower().strip(): f for f in wav_files}

def get_path(name):
    name = str(name).lower().strip()
    if not name.endswith(".wav"): name += ".wav"
    return wav_map.get(name)

df["full_path"] = df[file_col].apply(get_path)
df_valid = df.dropna(subset=["full_path"]).copy()

# 3. F0 ANALİZ FONKSİYONU (OTOKORELASYON)
def analyze_f0(path):
    try:
        y, sr = librosa.load(path, sr=None)
        y = y / (np.max(np.abs(y)) + 1e-8)
        frame_len, hop_len = int(0.03 * sr), int(0.015 * sr)
        frames = [y[i:i+frame_len] for i in range(0, len(y)-frame_len, hop_len)]
        
        f0_values = []
        energies = [np.sum(f**2)/len(f) for f in frames]
        energy_thresh = 0.2 * np.max(energies)

        for frame in frames:
            if (np.sum(frame**2)/len(frame)) > energy_thresh:
                f_det = frame - np.mean(frame)
                corr = correlate(f_det, f_det, mode='full')[len(frame)-1:]
                # İnsan sesi aralığı 75Hz - 500Hz
                low, high = int(sr/500), int(sr/75)
                if len(corr) > high:
                    peak = np.argmax(corr[low:high]) + low
                    f0_values.append(sr / peak)
        return np.mean(f0_values) if f0_values else np.nan
    except: return np.nan

# 4. İŞLEME VE SINIFLANDIRMA
print(f"Toplam {len(df_valid)} dosya işleniyor...")
results = []

for _, row in df_valid.iterrows():
    f0 = analyze_f0(row["full_path"])
    
    # SENİN BELİRLEDİĞİN EŞİKLER
    if np.isnan(f0): pred = "UNKNOWN"
    elif f0 < 170: pred = "MALE"
    elif f0 < 290: pred = "FEMALE"
    else: pred = "CHILD"
    
    # Excel'deki etiketi normalize et (M->MALE, F->FEMALE, C->CHILD)
    raw_gender = str(row[gender_col]).strip().upper()[0] # Sadece ilk harfi al (M, F, C)
    actual_map = {"M": "MALE", "F": "FEMALE", "C": "CHILD"}
    actual = actual_map.get(raw_gender, "UNKNOWN")
        
    results.append({
        "File": row[file_col],
        "Actual": actual,
        "Predicted": pred,
        "F0_Hz": round(f0, 2) if not np.isnan(f0) else 0
    })

results_df = pd.DataFrame(results)

# 5. ÖZET TABLO VE BAŞARI HESABI
# Sadece tahmini ve gerçeği bilinenleri al
valid_results = results_df[(results_df["Actual"] != "UNKNOWN") & (results_df["Predicted"] != "UNKNOWN")]

summary = valid_results.groupby("Actual").agg(
    Count=("File", "count"),
    Avg_F0=("F0_Hz", "mean"),
    Std_F0=("F0_Hz", "std")
).reset_index()

# Sınıf Bazlı Başarı
success_list = []
for cls in summary["Actual"]:
    subset = valid_results[valid_results["Actual"] == cls]
    acc = (subset["Actual"] == subset["Predicted"]).mean() * 100
    success_list.append(acc)

summary["Success_Percent"] = success_list

print("\n--- ÖZET TABLO ---")
print(summary)

total_acc = (valid_results["Actual"] == valid_results["Predicted"]).mean() * 100
print(f"\nGenel Başarı Oranı: %{total_acc:.2f}")

# Kaydet
results_df.to_excel("final_proje_sonuclari.xlsx", index=False)
summary.to_excel("proje_ozet_tablo.xlsx", index=False)