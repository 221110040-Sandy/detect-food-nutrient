import streamlit as st
import requests

st.set_page_config(page_title="DataMinds Nutrition Detector", page_icon="🍚", layout="wide")

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
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>DataMinds – Deteksi Gizi Makanan</h1>
        <p>Sandy Agre Nicola | Alvin . Lo</p>
    </div>
""", unsafe_allow_html=True)

# Layout 2 kolom
col_left, col_right = st.columns([1, 1], gap="large")

# Kolom Kiri - Input Section
with col_left:
    st.markdown("### Input Gambar")
    
    uploaded = st.file_uploader("Upload foto makanan (JPG/PNG)", type=["jpg","jpeg","png"], help="Pilih gambar makanan yang ingin dianalisis")
    
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
                # resp = requests.post("https://detect-food-nutrient-backend-production-e5fd.up.railway.app/predict", files=files, params=params)
                resp = requests.post("http://localhost:8000/predict", files=files, params=params)

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

    
