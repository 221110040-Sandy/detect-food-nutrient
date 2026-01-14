import streamlit as st
import requests
from datetime import date

st.set_page_config(page_title="DataMinds Nutrition Detector", page_icon="🍚", layout="wide")

# API Base URL
# API_BASE = "https://detect-food-nutrient-backend-production-e5fd.up.railway.app"
API_BASE = "https://detect-food-nutrient-production.up.railway.app"

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
    <style>
    /* Hilangkan padding atas */
    .block-container {
        padding-top: 2rem;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .output-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #28a745;
        color: white;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .tracker-card {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: #ffffff;
    }
    .tracker-card strong {
        color: #a0cfff;
    }
    .tracker-card small {
        color: #cccccc;
    }
    .status-kurang {
        color: #dc3545;
        font-weight: bold;
    }
    .status-terpenuhi, .status-aman {
        color: #28a745;
        font-weight: bold;
    }
    .status-berlebih {
        color: #ffc107;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>DataMinds – Deteksi Gizi Makanan</h1>
        <p>Sandy Agre Nicola | Alvin . Lo</p>
    </div>
""", unsafe_allow_html=True)

# Tabs untuk navigasi
tab_detect, tab_tracker = st.tabs(["🔍 Deteksi Makanan", "📊 Tracker Harian"])

# ============ TAB 1: DETEKSI MAKANAN ============
with tab_detect:
    # Layout 2 kolom
    col_left, col_right = st.columns([1, 1], gap="large")

    # Kolom Kiri - Input Section
    with col_left:
        st.markdown("### Input Gambar")
        
        uploaded = st.file_uploader("Upload foto makanan (JPG/PNG)", type=["jpg","jpeg","png"], help="Pilih gambar makanan yang ingin dianalisis", key="food_uploader")
        
        if uploaded:
            st.image(uploaded, caption="Foto yang diupload", use_container_width=True)
        else:
            st.info("👆 Silakan upload foto makanan terlebih dahulu")

    # Kolom Kanan - Output Section
    with col_right:
        st.markdown("### Hasil Analisis")
        
        if not uploaded:
            st.markdown("""
                <div style='text-align: center; padding: 3rem; color: #6c757d;'>
                    <h3>Menunggu Input</h3>
                    <p>Upload gambar makanan di sebelah kiri untuk melihat hasil analisis gizi</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            if 'last_uploaded_name' not in st.session_state or st.session_state.last_uploaded_name != uploaded.name:
                st.session_state.last_uploaded_name = uploaded.name
                
                with st.spinner("AI sedang menganalisis makanan..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    params = {"portion_g": 100}
                    resp = requests.post(f"{API_BASE}/predict", files=files, params=params)

                if resp.status_code != 200:
                    st.error(f"❌ Backend error: {resp.text}")
                    st.session_state.prediction_data = None
                else:
                    st.session_state.prediction_data = resp.json()
            
            if st.session_state.get('prediction_data') is None:
                st.error("❌ Gagal mendapatkan prediksi")
            else:
                data = st.session_state.prediction_data
                
                st.markdown("#### Makanan Terdeteksi")
                st.markdown(f'<div class="prediction-badge">{data["predicted_food"].upper()}</div>', unsafe_allow_html=True)
                st.progress(data['confidence'])
                st.caption(f"Confidence: {data['confidence']*100:.1f}%")
                
                st.markdown("---")
                
                with st.expander("Lihat Top-5 Prediksi Lainnya", expanded=False):
                    for i, (name, score) in enumerate(data["top5"], 1):
                        st.write(f"{i}. **{name}**: {score*100:.1f}%")
                
                st.markdown("---")
                
                if data["nutrition_per_100g"] is None:
                    st.warning("⚠️ Belum ada info gizi untuk makanan ini di database")
                else:
                    portion = st.slider("Perkiraan porsi (gram):", min_value=50, max_value=800, value=250, step=10, key="portion_slider")
                    
                    st.markdown("#### Informasi Gizi")
                    st.caption(f"Per {portion}g porsi")
                    
                    per100 = data["nutrition_per_100g"]
                    ratio = portion / 100.0
                    
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Kalori", f"{per100['calories_kcal'] * ratio:.0f} kcal")
                        st.metric("Protein", f"{per100['protein_g'] * ratio:.1f} g")
                        st.metric("Serat (Fiber)", f"{per100.get('fiber_g', 0.0) * ratio:.1f} g")
                        st.metric("Sodium", f"{per100.get('sodium_mg', 0.0) * ratio:.1f} mg")

                    with col2:
                        st.metric("Lemak", f"{per100['fat_g'] * ratio:.1f} g")
                        st.metric("Karbohidrat", f"{per100['carbs_g'] * ratio:.1f} g")
                        st.metric("Gula (Sugar)", f"{per100.get('sugar_g', 0.0) * ratio:.1f} g")
                    
                    st.markdown("---")
                    
                    # Tombol untuk menambahkan ke tracker
                    st.markdown("#### ➕ Tambahkan ke Tracker Harian")
                    if st.button("🍽️ Tambahkan Makanan Ini", key="add_to_tracker"):
                        # Hitung nutrisi untuk porsi yang dipilih
                        nutrition_data = {
                            "calories_kcal": round(per100['calories_kcal'] * ratio, 2),
                            "protein_g": round(per100['protein_g'] * ratio, 2),
                            "fat_g": round(per100['fat_g'] * ratio, 2),
                            "carbs_g": round(per100['carbs_g'] * ratio, 2),
                            "fiber_g": round(per100.get('fiber_g', 0.0) * ratio, 2),
                            "sugar_g": round(per100.get('sugar_g', 0.0) * ratio, 2),
                            "sodium_mg": round(per100.get('sodium_mg', 0.0) * ratio, 2),
                        }
                        
                        payload = {
                            "food_name": data["predicted_food"],
                            "portion_g": portion,
                            "nutrition": nutrition_data
                        }
                        
                        try:
                            resp = requests.post(f"{API_BASE}/tracker/add", json=payload)
                            if resp.status_code == 200:
                                st.toast(f"✅ {data['predicted_food']} ({portion}g) ditambahkan ke tracker!", icon="🍽️")
                            else:
                                st.error(f"❌ Gagal: {resp.text}")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")


# ============ TAB 2: TRACKER HARIAN ============
with tab_tracker:
    st.markdown("### 📊 Tracker Gizi Harian")
    st.caption("Pantau asupan gizi harianmu dan lihat kebutuhan yang masih kurang")
    
    # Pilih tanggal
    selected_date = st.date_input("Pilih Tanggal", value=date.today(), key="tracker_date")
    date_str = selected_date.isoformat()
    
    # Tombol refresh
    col_refresh, col_clear = st.columns([1, 1])
    with col_refresh:
        refresh_clicked = st.button("🔄 Refresh Data", key="refresh_tracker")
    with col_clear:
        if st.button("🗑️ Reset Tracker Hari Ini", key="clear_tracker"):
            try:
                resp = requests.post(f"{API_BASE}/tracker/clear", params={"date_str": date_str})
                if resp.status_code == 200:
                    st.success("✅ Tracker sudah direset!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Ambil data tracker
    try:
        resp = requests.get(f"{API_BASE}/tracker/status", params={"date_str": date_str})
        if resp.status_code == 200:
            tracker_data = resp.json()
        else:
            tracker_data = None
            st.error(f"❌ Gagal mengambil data: {resp.text}")
    except Exception as e:
        tracker_data = None
        st.error(f"❌ Error koneksi: {e}")
    
    if tracker_data:
        # Layout 2 kolom
        col_foods, col_analysis = st.columns([1, 1], gap="large")
        
        # Kolom Kiri - Daftar Makanan
        with col_foods:
            st.markdown("#### 🍽️ Makanan yang Dikonsumsi")
            
            if not tracker_data["entries"]:
                st.info("Belum ada makanan yang dicatat hari ini. Upload gambar makanan di tab 'Deteksi Makanan' dan tambahkan ke tracker!")
            else:
                for entry in tracker_data["entries"]:
                    with st.container():
                        st.markdown(f"""
                        <div class="tracker-card">
                            <strong>{entry['food_name'].upper()}</strong> - {entry['portion_g']}g
                            <br><small>🕐 {entry['timestamp']}</small>
                            <br><small>Kalori: {entry['nutrition']['calories_kcal']} kcal | 
                            Protein: {entry['nutrition']['protein_g']}g | 
                            Karbo: {entry['nutrition']['carbs_g']}g</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"❌ Hapus", key=f"del_{entry['index']}"):
                            try:
                                resp = requests.delete(f"{API_BASE}/tracker/remove/{entry['index']}", params={"date_str": date_str})
                                if resp.status_code == 200:
                                    st.success("✅ Dihapus!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
            
            st.markdown("---")
            st.markdown("#### 📈 Total Asupan Hari Ini")
            totals = tracker_data["total_nutrition"]
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric("Total Kalori", f"{totals['calories_kcal']:.0f} kcal")
                st.metric("Total Protein", f"{totals['protein_g']:.1f} g")
                st.metric("Total Lemak", f"{totals['fat_g']:.1f} g")
                st.metric("Total Serat", f"{totals['fiber_g']:.1f} g")
            with col_t2:
                st.metric("Total Karbohidrat", f"{totals['carbs_g']:.1f} g")
                st.metric("Total Gula", f"{totals['sugar_g']:.1f} g")
                st.metric("Total Sodium", f"{totals['sodium_mg']:.1f} mg")
        
        # Kolom Kanan - Analisis Kebutuhan
        with col_analysis:
            st.markdown("#### 🎯 Analisis Kebutuhan Gizi")
            st.caption("Berdasarkan AKG (Angka Kecukupan Gizi) harian")
            
            status_data = tracker_data["nutrient_status"]
            
            # Kelompokkan berdasarkan status
            kurang = []
            terpenuhi = []
            berlebih = []
            
            for nutrient, info in status_data.items():
                if info["status"] == "kurang":
                    kurang.append((nutrient, info))
                elif info["status"] in ["terpenuhi", "aman"]:
                    terpenuhi.append((nutrient, info))
                else:
                    berlebih.append((nutrient, info))
            
            # Tampilkan yang kurang
            if kurang:
                st.markdown("##### 🔴 Perlu Ditambah")
                for nutrient, info in kurang:
                    st.progress(min(info["percentage"] / 100, 1.0))
                    st.markdown(f"""
                    <div style='margin-bottom: 1rem;'>
                        <span class='status-kurang'>{info['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Tampilkan yang berlebih
            if berlebih:
                st.markdown("##### 🟡 Sudah Berlebih")
                for nutrient, info in berlebih:
                    st.progress(min(info["percentage"] / 100, 1.0))
                    st.markdown(f"""
                    <div style='margin-bottom: 1rem;'>
                        <span class='status-berlebih'>{info['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Tampilkan yang terpenuhi
            if terpenuhi:
                st.markdown("##### 🟢 Sudah Tercukupi")
                for nutrient, info in terpenuhi:
                    st.progress(min(info["percentage"] / 100, 1.0))
                    st.markdown(f"""
                    <div style='margin-bottom: 1rem;'>
                        <span class='status-terpenuhi'>{info['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Info referensi kebutuhan harian
            with st.expander("📋 Referensi Kebutuhan Gizi Harian"):
                st.markdown("""
                | Komponen Gizi | Rekomendasi Harian |
                |--------------|-------------------|
                | Energi | ± 2000 kkal |
                | Protein | ± 50 g |
                | Lemak total | ≤ 65 g |
                | Karbohidrat | ± 275 g |
                | Serat | ≥ 25 g |
                | Gula bebas | ≤ 50 g |
                | Natrium | ≤ 2000 mg |
                
                *Berdasarkan Angka Kecukupan Gizi (AKG) Indonesia*
                """)
