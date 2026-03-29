from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

app = Flask(__name__)

WINDOW         = 8    
FORECAST_STEPS = 24   

# 1. Veri Çekme
def safe_get(url):
    try:
        return requests.get(url, timeout=10).json()
    except Exception:
        return None


def get_realtime_kp():
    """planetary_k_index_1m.json → anlık Kp değeri."""
    raw = safe_get("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
    if not raw:
        return None
    df = pd.DataFrame(raw)
    df.columns = [c.lower() for c in df.columns]
    # Olası sütun adları: kp_index, estimated_kp, kp
    kp_col = next((c for c in df.columns if "kp" in c), None)
    if kp_col is None:
        return None
    df[kp_col] = pd.to_numeric(df[kp_col], errors="coerce")
    df = df.dropna(subset=[kp_col])
    return round(float(df[kp_col].iloc[-1]), 2) if not df.empty else None


def get_kp_history():
    """3-saatlik Kp indeksi → son 7 gün (model eğitimi için)."""
    raw = safe_get("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
    if not raw:
        return None
    df = pd.DataFrame(raw[1:], columns=raw[0])
    df.columns = [c.lower() for c in df.columns]
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    df["kp"]       = pd.to_numeric(df["kp"], errors="coerce")
    df = df.dropna(subset=["kp"]).sort_values("time_tag").reset_index(drop=True)
    cutoff = datetime.utcnow() - timedelta(days=7)
    return df[df["time_tag"] >= cutoff].reset_index(drop=True)


# 2. Feature Engineering – Sliding Window
def build_features(series, window=WINDOW):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# 3. G Seviyesi ve UI Metinleri
def kp_to_G(kp):
    if kp >= 9:   return "G5"
    elif kp >= 8: return "G4"
    elif kp >= 7: return "G3"
    elif kp >= 6: return "G2"
    elif kp >= 5: return "G1"
    else:         return "G0"


# Slider değeri 1-5 → G1-G5 (G0 için 1 kullanılır)
G_META = {
    "G0": {
        "label":      "G0 - Sakin",
        "slider_val": 1,
        "alert_text": "Aktivite Yok – Sistem Normal",
        "risk":       "Jeomanyetik alan sakin. Sistemler üzerinde herhangi bir baskı veya sapma beklenmemektedir.",
        "threats": [
            "Herhangi bir voltaj dalgalanması veya anomali beklenmez. Şebeke yükü normal standartlardadır.",
            "Yörünge sapması, sinyal kaybı veya yüzey şarjı riski yoktur. Telemetri verileri temizdir.",
            "GPS kalibrasyonları tam isabetli çalışır, HF ve VHF radyo iletişiminde parazit oluşmaz."
        ],
        "precautions_list": [
            "Rutin izleme ve bakım prosedürlerine standart planlamaya uygun olarak devam edilebilir.",
            "Uydu ve uzay aracı sensörleri normal operasyon modunda (nominal) tutulabilir.",
            "Ekstra bir iletişim yedeklemesine veya yedek frekans bandı kullanımına gerek yoktur."
        ]
    },
    "G1": {
        "label":      "G1 - Hafif",
        "slider_val": 1,
        "alert_text": "Hafif Manyetik Hareketlilik",
        "risk":       "Hafif jeomanyetik aktivite. Çoğu sistem dayanıklıdır ancak çok hassas ekipmanlar etkilenebilir.",
        "threats": [
            "Güç sistemlerinde lokalize ufak gerilim dalgalanmaları (şebeke gürültüsü) yaşanabilir.",
            "Göçmen hayvanlar için hafif yönelim bozukluğu riski. Uzay araçlarında düşük seviyeli statik etkiler.",
            "HF (Yüksek Frekans) iletişim sistemlerinde yüksek enlemlerde nadiren küçük zayıflamalar görülebilir."
        ],
        "precautions_list": [
            "Şebeke dengeleyicilerinin eşik değerleri kontrol edilmeli, olağan dışı akım çekişleri izlenmelidir.",
            "Güneş paneli verimliliği izlenmeli, gerekirse hassas görevler bir sonraki yörüngeye ertelenmelidir.",
            "Havacılık telsizlerinde sinyal gürültüsü (S/N düşüşü) payı bırakılmalı, alternatif kanallar hazır tutulmalıdır."
        ]
    },
    "G2": {
        "label":      "G2 - Orta",
        "slider_val": 2,
        "alert_text": "Orta Şiddetli Fırtına Uyarısı",
        "risk":       "Orta düzey aktivite. Özellikle kuzey ve güney yüksek enlemlerindeki altyapılarda ölçülebilir etkiler başlar.",
        "threats": [
            "Özellikle yüksek enlemlerdeki transformatörlerde ısınma riski ve gerilim kontrol sorunları başlar.",
            "Uydu gövdelerinde hafif statik elektrik yüklenmeleri ve yörünge verilerinde ufak sapmalar görülebilir.",
            "Radyo dalgalarında (özellikle gece saatlerinde) zayıflamalar ve GPS sinyallerinde metrik hatalar olabilir."
        ],
        "precautions_list": [
            "Transformatör sıcaklıkları telemetri ile yakından takip edilmeli, kapasitör bankları devreye alınmalıdır.",
            "Uydu manevra hesaplamaları (drag estimation) güncellenmeli ve fazladan düzeltme yakıtı ayrılmalıdır.",
            "Hassas GPS kullanan tarım veya denizcilik otonom araçları manuel kontrol veya DGPS moduna geçirilmelidir."
        ]
    },
    "G3": {
        "label":      "G3 - Güçlü",
        "slider_val": 3,
        "alert_text": "Şiddetli Jeomanyetik Fırtına Bekleniyor",
        "risk":       "Geniş çaplı etkiler. Güç şebekelerinde koruyucu eylemler gerektirir. Uydu iletişiminde bozulmalar nettir.",
        "threats": [
            "Şebeke gerilim kontrollerinde zorlanmalar başlar, bazı koruyucu röleler sisteme yanlış alarm verebilir.",
            "Alçak Dünya Yörüngesi (LEO) uydularında atmosferik sürtünme belirginleşir, ciddi yönelim hataları oluşur.",
            "Uydu navigasyonlarında konum atlamaları (loss-of-lock) yaşanır; HF radyo iletişimi kutuplarda tamamen kesilebilir."
        ],
        "precautions_list": [
            "Bölgesel şebeke operatörleri voltaj regülatörlerini agresif konuma getirmeli, kritik yükleri izole etmelidir.",
            "Uydulardaki bazı yüksek gerilimli sensörler geçici olarak kapatılmalı (safe-hold), yeni yörünge tahminleri yüklenmelidir.",
            "Kıtalararası havacılık rotaları kutuplardan daha düşük enlemlere çekilmeli, UHF/VHF iletişim alternatifleri kullanılmalıdır."
        ]
    },
    "G4": {
        "label":      "G4 - Çok Güçlü",
        "slider_val": 4,
        "alert_text": "Kritik Fırtına Seviyesi Tespit Edildi",
        "risk":       "Çok şiddetli bölgesel tehlikeler. Teknolojik altyapıda kayıplar yaşanabilir, ciddi zararlar oluşabilir çok acil önlemler şarttır.",
        "threats": [
            "Yaygın voltaj kararsızlığı nedeniyle şebeke koruma sistemleri hatalı şekilde tüm bölgeleri elektriksiz bırakabilir. Yedek pil ve batarya sistemleri bulunmalıdır",
            "Uydu yüzeylerinde ağır yüklenme (surface charging), cihaz sıfırlanmaları ve telemetri takiplerinde kopmalar yaşanır. Uydudan alınan verilerde yaşanan sapma bilim insanlarının sonuçlarını etkilemektedir.",
            "Radyo ve navigasyon sistemlerinde saatler süren ağır kesintiler ve geniş çaplı GPS sinyal kararmaları görülebilir. Pilotlar, drone görevleri, gemi seyehatleri aksamaya uğrayabilir"
        ],
        "precautions_list": [
            "Elektrik şebekelerinde jeneratör modlarına geçilmeli, reaktif güç rezervleri maksimum seviyeye çıkarılmalıdır.",
            "Uydular acil durum 'Güneş Gösterisi (Sun-Point)' moduna çapraz yedekli kilitlenmeli, bilim enstrümanları tamamen kapatılmalıdır.",
            "Havacılık ve gemicilikte GPS kullanımına güvenilmemeli, ataletsel seyrüsefer (INS) sistemlerine tam geçiş yapılmalıdır."
        ]
    },
    "G5": {
        "label":      "G5 - Ekstrem",
        "slider_val": 5,
        "alert_text": "EKSTREM JEOMANYETİK FIRTINA!",
        "risk":       "Küresel çapta medeniyet seviyesinde (Carrington olayı benzeri) felaket fırtınası. Tam teknolojik kapanma riski.",
        "threats": [
            "Büyük transformatörlerin iç çekirdeklerinin kalıcı olarak yanması ve kıtaları kapsayan uzun süreli (haftalarca) tam kararmalar.",
            "Yüzlerce uydunun atmosfer sınırının çökmesiyle yörüngeden sapan uzay çöpüne dönüşmesi, iletişim ağının yok olması. Dünyanın dört bir yanında kullanılan CBS sistemlerinin ve navigasyonların çökmesi",
            "Yüksek frekans (HF), VHF ve UHF telsiz sistemlerinin tamamen çökmesi, günlerce süren askeri ve sivil GPS kararması. "
        ],
        "precautions_list": [
            "Tüm yedek güç (Dizel/Akü) jeneratörleri manuel devreye alınmalı, ana şebeke ile olan tüm fiziksel şalter bağlantıları hemen kesilmelidir.",
            "Tüm uzay operasyonları iptal edilmeli, derin uzay ve LEO uyduları tümüyle güçsüz 'tam kapalı (power-off)' izolasyona zorlanmalıdır.",
            "Elektronik olmayan (analog/mekanik) eski radyo istasyonları devreye sokulmalı, karasal fiber optik veya bakır hatlara geçilmelidir."
        ]
    }
}


def alert_level(kp):
    if kp >= 7:   return "HIGH"
    elif kp >= 5: return "MEDIUM"
    else:         return "LOW"

# 4. Ana Tahmin Motoru
def run_forecast():
    df          = get_kp_history()
    current_kp  = get_realtime_kp()   # planetary_k_index_1m.json

    if df is None or len(df) < WINDOW + 5:
        return {"error": "Yeterli veri alınamadı.", "history": [], "forecast": []}

    kp_values = df["kp"].values.astype(float)

    # Model eğitimi 
    X, y  = build_features(kp_values)
    
    # Model Doğruluğu Hesaplama 
    split_idx = int(len(X) * 0.8)
    if split_idx > 0 and split_idx < len(X):
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test   = X[split_idx:], y[split_idx:]
        eval_model = RandomForestRegressor(n_estimators=100, random_state=42)
        eval_model.fit(X_train, y_train)
        test_preds = eval_model.predict(X_test)
        mae = float(np.mean(np.abs(test_preds - y_test)))
    else:
        # Fallback (Veri çok azsa)
        mae = 1.0 

    accuracy_pct = round(max(0.0, 100.0 - (mae / 9.0) * 100.0), 1)

    # Nihai tahminler için modeli tüm veriyle eğit
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)

    # Geçmiş veri (7 gün) 
    history = [
        {
            "time": row["time_tag"].strftime("%Y-%m-%dT%H:%M"),
            "kp":   round(float(row["kp"]), 2),
            "G":    kp_to_G(float(row["kp"]))
        }
        for _, row in df.iterrows()
    ]

    # ── 3 günlük iteratif tahmin ─────────────
    window_buf = list(kp_values[-WINDOW:])
    last_time  = df["time_tag"].iloc[-1]
    forecasts  = []

    for i in range(FORECAST_STEPS):
        feat    = np.array(window_buf[-WINDOW:]).reshape(1, -1)
        pred_kp = float(model.predict(feat)[0])
        pred_kp = round(max(0.0, min(9.0, pred_kp)), 2)
        fut_t   = last_time + timedelta(hours=3 * (i + 1))

        forecasts.append({
            "time": fut_t.strftime("%Y-%m-%dT%H:%M"),
            "kp":   pred_kp,
            "G":    kp_to_G(pred_kp)
        })
        window_buf.append(pred_kp)

    # Özet 
    max_kp    = max(p["kp"] for p in forecasts)
    max_G     = kp_to_G(max_kp)
    meta      = G_META[max_G]

    # Anlık Kp 
    if current_kp is None:
        current_kp = round(float(kp_values[-1]), 2)
    current_G = kp_to_G(current_kp)

    # Gnülük tahmin özeti
    forecast_daily = []
    for d in range(3):
        s, e     = d * 8, d * 8 + 8
        day_fc   = forecasts[s:e]
        kps      = [f["kp"] for f in day_fc]
        max_kp_d = max(kps)
        fut_date = last_time + timedelta(days=d + 1)
        forecast_daily.append({
            "day":        d + 1,
            "date":       fut_date.strftime("%d %b"),
            "max_kp":     round(max_kp_d, 2),
            "avg_kp":     round(sum(kps) / len(kps), 2),
            "G":          kp_to_G(max_kp_d),
            "kp_series":  kps
        })

    return {
        # Ham veriler
        "history":        history,
        "forecast":       forecasts,
        "forecast_daily": forecast_daily,
        # Anlık (1m endpoint)
        "current_kp":  current_kp,
        "current_G":   current_G,
        # Tahmin özeti
        "max_kp":      round(max_kp, 2),
        "max_G":       max_G,
        "alert":       max_kp >= 5,
        "alert_level": alert_level(max_kp),
        # UI metin alanları
        "storm_label": meta["label"],
        "slider_val":  meta["slider_val"],
        "alert_text":  meta["alert_text"],
        "risk":        meta["risk"],
        "threats":     meta["threats"],
        "precautions": meta["precautions_list"],
        "accuracy":    accuracy_pct,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M") + "Z"
    }


# ─────────────────────────────────────────────
# 5. Flask Route'ları
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/data")
def data():
    return jsonify(run_forecast())


@app.route("/send-alert", methods=["POST"])
def send_alert():
    try:
        data = request.get_json()
        email = data.get("email")
        level = data.get("level")
        desc = data.get("desc")

        if not email or "@" not in email:
            return jsonify({"success": False, "message": "Geçerli bir e-posta adresi giriniz."})

        # Terminale mock mail logluyoruz (Gerçek e-posta göndermek için aşağıdaki bloğu açabilirsiniz)
        print("\n" + "!"*60)
        print("      [ E-POSTA SİMÜLASYONU ] -> ACİL UYARI GÖNDERİLİYOR!      ")
        print("!"*60)
        print(f"ALICI : {email}")
        print(f"KONU  : ACİL UYARI: Güneş Fırtınası Tespit Edildi ({level})")
        print(f"MESAJ : Sistemlerimizde {level} seviyesinde fırtına simüle edilmiştir.")
        print(f"DETAY : {desc}")
        print("!"*60 + "\n")

        """
        # GERÇEK E-POSTA GÖNDERİMİ İÇİN: (Gmail Uygulama Şifresi Gerekir)
        import smtplib
        from email.mime.text import MIMEText

        sender_email = "sizin_email@gmail.com"
        sender_pass = "845-117-422"

        msg = MIMEText(f"Sistemlerimizde {level} seviyesinde fırtına uyarısı tespit edildi.\n\nDetay:\n{desc}\n\nLütfen önlemleri uygulayınız.")
        msg['Subject'] = f"Güneş Fırtınası Uyarısı: {level}"
        msg['From'] = sender_email
        msg['To'] = email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, email, msg.as_string())
        """

        return jsonify({
            "success": True, 
            "message": f"Uyarı sinyali {email} adresine başarıyla gönderildi (Simülasyon)!"
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Sunucu hatası: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True)
