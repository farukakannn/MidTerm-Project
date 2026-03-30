import os
import pandas as pd
import numpy as np
import librosa
import glob
from scipy.signal import correlate

# 1. KONUM VE DOSYA HAZIRLIĞI
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

print(f"--- ANALİZ BAŞLADI ---")
# Tüm alt klasörlerdeki wav'ları tara [cite: 18]
all_wavs = glob.glob(os.path.join(current_dir, "**", "*.wav"), recursive=True)
wav_map = {os.path.basename(f).lower().strip(): f for f in all_wavs}

# Excel'i yükle ve sütunları temizle [cite: 17, 19]
df = pd.read_excel("MetaData.xlsx")
df.columns = [str(c).strip() for c in df.columns]

# Sütun isimlerini tespit et
file_col = next((c for c in df.columns if 'file' in c.lower() or 'dosya' in c.lower()), None)
gender_col = next((c for c in df.columns if 'gender' in c.lower() or 'cinsiyet' in c.lower()), None)

# EŞLEŞTİRME (Hata veren kısım düzeltildi)
def fix_match(name):
    name = str(name).lower().strip()
    if not name.endswith(".wav"): name += ".wav"
    return wav_map.get(name)

df["full_path"] = df[file_col].apply(fix_match)
df_clean = df.dropna(subset=["full_path"]).copy()

print(f"Bulunan Wav: {len(all_wavs)} | Eşleşen Kayıt: {len(df_clean)}")

# 2. ÖZNİTELİK ÇIKARIMI (TIME DOMAIN - PROJE İSTERLERİ) [cite: 20, 21]
def get_f0_autocorr(path):
    try:
        y, sr = librosa.load(path, sr=None)
        # 20-30 ms Pencereleme (Windowing) [cite: 23]
        frame_len = int(0.025 * sr)
        hop_len = int(0.01 * sr)
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len).T
        
        f0_list = []
        for frame in frames:
            # Enerji (STE) ile sessiz bölge eleme [cite: 24, 25]
            if np.sum(frame**2) > 0.005: 
                # Otokorelasyon Hesabı: Rτ = sum(x[n]x[n-τ]) [cite: 28, 29, 32]
                corr = correlate(frame, frame, mode='full')[len(frame)-1:]
                # İnsan sesi frekans sınırları (80Hz - 450Hz)
                low_lag, high_lag = int(sr / 450), int(sr / 80)
                
                if len(corr) > high_lag:
                    peak_lag = np.argmax(corr[low_lag:high_lag]) + low_lag
                    f0_list.append(sr / peak_lag)
        return np.mean(f0_list) if f0_list else 0
    except:
        return 0

# 3. ANALİZ DÖNGÜSÜ VE SINIFLANDIRMA 
results = []
print("Dosyalar işleniyor, lütfen bekle...")

for _, row in df_clean.iterrows():
    f0 = get_f0_autocorr(row['full_path'])
    
    # Kural Tabanlı Sınıflandırma (Gender Classification) [cite: 8, 11]
    if f0 < 155: pred = "Male"
    elif 155 <= f0 < 250: pred = "Female"
    else: pred = "Child"
    
    results.append({
        "File": row[file_col],
        "Actual": row[gender_col],
        "Predicted": pred,
        "F0_Mean": round(f0, 2)
    })

# 4. SONUÇLAR VE PERFORMANS [cite: 12, 49]
res_df = pd.DataFrame(results)
accuracy = (res_df["Actual"].str.upper() == res_df["Predicted"].str.upper().str[0]).mean() * 100 
# Not: Excel'de 'M', 'F', 'C' yazıyorsa yukarıdaki kıyaslama onu yakalar.

print("\n--- ANALİZ SONUÇLARI ---")
print(res_df.head(10))
print(f"\nGenel Başarı Oranı: %{accuracy:.2f}")

# Rapor için İstatistik Tablosu [cite: 50, 63]
print("\n--- RAPOR İÇİN SINIF BAZLI ÖZET ---")
for cls in ["Male", "Female", "Child"]:
    subset = res_df[res_df["Predicted"] == cls]
    if not subset.empty:
        print(f"{cls} -> Ort. F0: {subset['F0_Mean'].mean():.2f} Hz | Sayı: {len(subset)}")