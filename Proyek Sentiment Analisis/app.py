import os
import re
import json
import string
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shopee Sentiment Studio — AI Demo",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model", "Model_best_logistic.pkl")
TFIDF_PATH = os.path.join(CURRENT_DIR, "model", "tfidf.pkl")
SLANGWORDS_PATH = os.path.join(CURRENT_DIR, "slangwords.json")
SAMPLE_CSV_PATH = os.path.join(CURRENT_DIR, "data", "ulasan_shoope.csv")

# Inject High-End Minimalist CSS
st.markdown("""
<style>
    /* Global Typography & Font Family */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* App Header Banner */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #334155;
    }
    .app-header-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-header-desc {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .header-badge {
        background: rgba(238, 77, 45, 0.15);
        color: #ff6b4a;
        border: 1px solid rgba(238, 77, 45, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Modern Card Container */
    .custom-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    .custom-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Result Hero Box */
    .sentiment-hero {
        border-radius: 16px;
        padding: 1.8rem 1.4rem;
        text-align: center;
        border: 1px solid;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .hero-pos {
        background: #f0fdf4;
        border-color: #86efac;
    }
    .hero-neu {
        background: #fffbeb;
        border-color: #fde68a;
    }
    .hero-neg {
        background: #fef2f2;
        border-color: #fca5a5;
    }
    .sentiment-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    .sentiment-subtitle {
        font-size: 0.86rem;
        color: #64748b;
        margin: 0;
    }

    /* Tag Pills for Keywords */
    .kw-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
        border: 1px solid;
    }
    .kw-pos {
        background: #dcfce7;
        color: #166534;
        border-color: #bbf7d0;
    }
    .kw-neg {
        background: #fee2e2;
        color: #991b1b;
        border-color: #fecaca;
    }
    .kw-neu {
        background: #f1f5f9;
        color: #475569;
        border-color: #e2e8f0;
    }

    /* Pipeline Step Box */
    .pipeline-step {
        background: #f8fafc;
        border-left: 3px solid #ee4d2d;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        font-size: 0.86rem;
    }
    .pipeline-label {
        font-weight: 600;
        color: #334155;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .pipeline-val {
        color: #0f172a;
        word-break: break-word;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Quick Preset Button Styling */
    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Tab header styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        padding: 0 16px;
        font-weight: 600;
        font-size: 0.92rem;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f1f5f9 !important;
        color: #ee4d2d !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Caching & NLP Logic
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_nltk():
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

init_nltk()

@st.cache_resource(show_spinner=False)
def load_assets():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    
    slangwords = {}
    if os.path.exists(SLANGWORDS_PATH):
        with open(SLANGWORDS_PATH, "r", encoding="utf-8") as f:
            slangwords = json.load(f)
            
    from nltk.corpus import stopwords
    try:
        id_stopwords = set(stopwords.words("indonesian"))
        en_stopwords = set(stopwords.words("english"))
        all_stopwords = id_stopwords.union(en_stopwords)
    except Exception:
        all_stopwords = set()
    
    extra = {'iya','yaa','gak','nya','na','sih','ku','di','ga','ya','gaa','loh','kah','woi','woii','woy'}
    all_stopwords.update(extra)
    
    return model, tfidf, slangwords, all_stopwords

model, tfidf, slangwords, stopwords_set = load_assets()

def cleaning_text(text: str) -> str:
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"https?://\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def casefolding_text(text: str) -> str:
    return text.lower()

def fix_slangwords(text: str, slang_dict: dict) -> str:
    words = text.split()
    fixed_words = [slang_dict.get(w.lower(), w) for w in words]
    return ' '.join(fixed_words)

def tokenize_and_filter(text: str, stop_words: set) -> list:
    from nltk.tokenize import word_tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return [w for w in tokens if w.lower() not in stop_words and len(w) > 1]

def full_preprocess(text: str) -> dict:
    cleaned = cleaning_text(text)
    casefolded = casefolding_text(cleaned)
    slang_fixed = fix_slangwords(casefolded, slangwords)
    tokens = tokenize_and_filter(slang_fixed, stopwords_set)
    final_sentence = ' '.join(tokens)
    return {
        "raw": text,
        "cleaned": cleaned,
        "casefolded": casefolded,
        "slang_fixed": slang_fixed,
        "tokens": tokens,
        "final": final_sentence
    }

def analyze_review(text: str, threshold_netral: float = 0.5):
    prep = full_preprocess(text)
    processed_text = prep["final"] if prep["final"].strip() else prep["slang_fixed"]
    
    vec = tfidf.transform([processed_text])
    probs = model.predict_proba(vec.toarray())[0] # [Negatif, Netral, Positif]
    
    prob_neg = float(probs[0])
    prob_neu = float(probs[1])
    prob_pos = float(probs[2])
    
    if prob_neu > threshold_netral:
        sentiment = "NETRAL"
        color = "#d97706"
        hero_class = "hero-neu"
        icon = "😐"
        badge_text = "Netral / Ambigu"
    elif prob_pos > prob_neg:
        sentiment = "POSITIF"
        color = "#16a34a"
        hero_class = "hero-pos"
        icon = "😊"
        badge_text = "Ulasan Positif"
    else:
        sentiment = "NEGATIF"
        color = "#dc2626"
        hero_class = "hero-neg"
        icon = "😡"
        badge_text = "Ulasan Negatif"

    # Extract detected vocabulary features & contributions
    detected_words = []
    tokens = processed_text.split()
    vocab = tfidf.vocabulary_
    for word in tokens:
        if word in vocab:
            idx = vocab[word]
            pos_weight = float(model.coef_[2, idx])
            neg_weight = float(model.coef_[0, idx])
            if pos_weight > neg_weight and pos_weight > 0.5:
                tag_type = "pos"
                impact = "Mendorong Positif"
            elif neg_weight > pos_weight and neg_weight > 0.5:
                tag_type = "neg"
                impact = "Mendorong Negatif"
            else:
                tag_type = "neu"
                impact = "Netral / Berimbang"
            detected_words.append({
                "word": word,
                "tag_type": tag_type,
                "impact": impact,
                "pos_weight": pos_weight,
                "neg_weight": neg_weight
            })

    return {
        "sentiment": sentiment,
        "badge_text": badge_text,
        "icon": icon,
        "color": color,
        "hero_class": hero_class,
        "prob_neg": prob_neg,
        "prob_neu": prob_neu,
        "prob_pos": prob_pos,
        "prep": prep,
        "processed_text": processed_text,
        "detected_words": detected_words
    }


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 1.8rem;">🛍️</span>
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0f172a;">Shopee Studio</h3>
            <p style="margin: 0; font-size: 0.78rem; color: #64748b;">Sentiment Intelligence AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #f1f5f9; border-radius: 8px; padding: 10px 12px; margin-bottom: 18px; border: 1px solid #e2e8f0; font-size: 0.8rem; color: #334155;">
        <span style="color: #10b981; font-weight: 700;">●</span> <strong>Model Siap Digunakan</strong><br/>
        Logistic Regression • 91.37% Accuracy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Konfigurasi Model")
    threshold_netral = st.slider(
        "Threshold Probabilitas Netral",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.05,
        help="Batas minimum probabilitas kelas Netral agar teks dinilai NETRAL. Default: 0.50."
    )
    



# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div>
        <h1 class="app-header-title">
            <span>🛍️</span> Analisis Sentimen Ulasan Shopee
        </h1>
        <p class="app-header-desc">
            Sistem klasifikasi opini pelanggan aplikasi Shopee berbasis Machine Learning & Natural Language Processing.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "💬 Analisis Ulasan Tunggal",
    "📂 Analisis Masal (Batch & CSV)",
    "📈 Dashboard & Eksplorasi Data",
    "🔬 Spesifikasi Model & Pipeline"
])


# =========================================================
# TAB 1: Analisis Ulasan Tunggal
# =========================================================
with tabs[0]:
    st.markdown("#### Uji Coba Prediksi Teks")
    st.caption("Ketik ulasan pelanggan secara langsung atau pilih template pengujian di bawah ini:")

    # Preset Chips
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    
    preset_chosen = None
    if col_p1.button("✨ Sangat Puas & Cepat", use_container_width=True):
        preset_chosen = "Saya sangat senang dengan produk ini! Pengiriman sangat cepat, seller ramah dan barang sesuai deskripsi."
    if col_p2.button("🎁 Promo & Ongkir", use_container_width=True):
        preset_chosen = "Mantap banget shopee selalu kasih gratis ongkir dan diskon besar. Puas belanja di sini!"
    if col_p3.button("⚠️ Barang Rusak / Kecewa", use_container_width=True):
        preset_chosen = "Barang jelek banget tidak sesuai pesanan, rusak dan pelayanan sangat buruk!"
    if col_p4.button("🐌 Aplikasi Lemot / Bug", use_container_width=True):
        preset_chosen = "Aplikasi lelet banget sering error pas checkout tolong diperbaiki."
    if col_p5.button("📦 Standar / Netral", use_container_width=True):
        preset_chosen = "banyak barangya"

    # Maintain state for input text
    if "current_input_text" not in st.session_state:
        st.session_state["current_input_text"] = "Aplikasi dan layanannya sangat memuaskan, pengiriman cepat dan packing aman!"
        
    if preset_chosen:
        st.session_state["current_input_text"] = preset_chosen

    user_text = st.text_area(
        label="Input Teks Ulasan",
        value=st.session_state["current_input_text"],
        height=110,
        placeholder="Tulis opini atau komentar ulasan di sini...",
        help="Mendukung singkatan/bahasa gaul (misal: 'gk', 'bgt', 'lemot', 'makasih', 'banget')."
    )

    col_action1, col_action2 = st.columns([1, 4])
    with col_action1:
        run_btn = st.button("🚀 Prediksi Sentimen", type="primary", use_container_width=True)

    if user_text.strip():
        res = analyze_review(user_text, threshold_netral=threshold_netral)
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        col_hero, col_metrics = st.columns([1, 1.4], gap="medium")
        
        with col_hero:
            st.markdown(f"""
            <div class="sentiment-hero {res['hero_class']}">
                <div style="font-size: 3.2rem; line-height: 1;">{res['icon']}</div>
                <div class="sentiment-title" style="color: {res['color']};">
                    {res['sentiment']}
                </div>
                <p class="sentiment-subtitle">{res['badge_text']}</p>
                <div style="margin-top: 14px; padding: 6px 14px; background: rgba(0,0,0,0.04); border-radius: 9999px; font-size: 0.8rem; font-weight: 600; color: #475569;">
                    Keyakinan: {max(res['prob_pos'], res['prob_neu'], res['prob_neg']) * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_metrics:
            st.markdown("""
            <div class="custom-card" style="margin-bottom: 0;">
                <h4 style="margin: 0 0 14px 0; font-size: 1.05rem; font-weight: 700; color: #0f172a;">
                    Distribusi Probabilitas Model
                </h4>
            """, unsafe_allow_html=True)
            
            # Interactive Plotly horizontal bar chart for clean SaaS look
            fig_prob = go.Figure()
            
            labels = ["Positif 😊", "Netral 😐", "Negatif 😡"]
            vals = [res["prob_pos"] * 100, res["prob_neu"] * 100, res["prob_neg"] * 100]
            bar_colors = ["#10b981", "#f59e0b", "#ef4444"]
            
            fig_prob.add_trace(go.Bar(
                x=vals,
                y=labels,
                orientation='h',
                marker=dict(color=bar_colors, cornerradius=6),
                text=[f"<b>{v:.1f}%</b>" for v in vals],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='white', size=13, family='Plus Jakarta Sans')
            ))
            
            fig_prob.update_layout(
                xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(autorange="reversed", showgrid=False, zeroline=False),
                margin=dict(l=0, r=0, t=10, b=10),
                height=160,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)






# =========================================================
# TAB 2: Analisis Masal (Batch & CSV)
# =========================================================
with tabs[1]:
    st.markdown("#### Analisis Masal / Impor Data Ulasan")
    st.caption("Lakukan inferensi sentimen sekaligus untuk puluhan ulasan pelanggan dari dataset Shopee atau file CSV Anda sendiri.")
    
    subtab1, subtab2 = st.tabs(["📦 Gunakan Data Asli Shopee Play Store", "📤 Unggah File CSV Sendiri"])
    
    batch_df = None
    target_col = None
    
    with subtab1:
        if os.path.exists(SAMPLE_CSV_PATH):
            st.markdown("""
            Dataset asli ulasan aplikasi Shopee (`ulasan_shoope.csv`) tersedia di folder `data/`.
            """)
            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                sample_qty = st.select_slider(
                    "Jumlah sampel ulasan yang akan diproses:",
                    options=[10, 25, 50, 100, 200],
                    value=25
                )
            with col_s2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                load_btn = st.button("⚡ Muat & Analisis Sampel Shopee", type="primary", use_container_width=True)
                
            if load_btn:
                raw_df = pd.read_csv(SAMPLE_CSV_PATH, nrows=sample_qty * 3)
                raw_df = raw_df.dropna(subset=['content']).drop_duplicates(subset=['content']).head(sample_qty)
                batch_df = raw_df
                target_col = "content"
        else:
            st.error("File `data/ulasan_shoope.csv` tidak ditemukan.")
            
    with subtab2:
        file_uploaded = st.file_uploader("Unggah file CSV (dengan kolom teks ulasan):", type=["csv"])
        if file_uploaded:
            user_csv = pd.read_csv(file_uploaded)
            st.markdown("Pratinjau Data (5 Baris):")
            st.dataframe(user_csv.head(5), use_container_width=True)
            col_chosen = st.selectbox("Pilih kolom ulasan teks:", user_csv.columns.tolist())
            if st.button("🚀 Proses CSV Ini", type="primary"):
                batch_df = user_csv
                target_col = col_chosen

    if batch_df is not None and target_col is not None:
        with st.spinner(f"Sedang menganalisis {len(batch_df)} ulasan dengan TF-IDF & Logistic Regression..."):
            texts = batch_df[target_col].astype(str).tolist()
            out = []
            for t in texts:
                r = analyze_review(t, threshold_netral=threshold_netral)
                out.append({
                    "Ulasan": t,
                    "Sentimen": r["sentiment"],
                    "Skor_Positif": round(r["prob_pos"], 4),
                    "Skor_Netral": round(r["prob_neu"], 4),
                    "Skor_Negatif": round(r["prob_neg"], 4)
                })
            result_df = pd.DataFrame(out)
            
        st.success(f"Analisis selesai! {len(result_df)} ulasan berhasil diklasifikasikan.")
        
        # Summary KPI Cards
        total_n = len(result_df)
        counts = result_df["Sentimen"].value_counts()
        p_c = counts.get("POSITIF", 0)
        u_c = counts.get("NETRAL", 0)
        n_c = counts.get("NEGATIF", 0)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Ulasan", f"{total_n:,}")
        k2.metric("😊 Positif", f"{p_c} ({p_c/total_n*100:.1f}%)")
        k3.metric("😐 Netral", f"{u_c} ({u_c/total_n*100:.1f}%)")
        k4.metric("😡 Negatif", f"{n_c} ({n_c/total_n*100:.1f}%)")
        
        # Donut & Distribution Chart
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns([1, 1.6])
        
        with col_c1:
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Positif", "Netral", "Negatif"],
                values=[p_c, u_c, n_c],
                hole=.6,
                marker=dict(colors=["#10b981", "#f59e0b", "#ef4444"])
            )])
            fig_donut.update_layout(
                title_text="Proporsi Sentimen",
                margin=dict(l=20, r=20, t=40, b=20),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_c2:
            fig_bar = go.Figure(data=[go.Bar(
                x=["Positif 😊", "Netral 😐", "Negatif 😡"],
                y=[p_c, u_c, n_c],
                marker_color=["#10b981", "#f59e0b", "#ef4444"],
                text=[f"{p_c} ({p_c/total_n*100:.1f}%)", f"{u_c} ({u_c/total_n*100:.1f}%)", f"{n_c} ({n_c/total_n*100:.1f}%)"],
                textposition='auto'
            )])
            fig_bar.update_layout(
                title_text="Jumlah Ulasan per Kategori",
                yaxis_title="Banyak Ulasan",
                margin=dict(l=20, r=20, t=40, b=20),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # Data Table & Export
        st.markdown("#### Tabel Hasil Prediksi Lengkap")
        st.dataframe(result_df, use_container_width=True)
        
        csv_download = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Analisis (CSV)",
            data=csv_download,
            file_name="hasil_sentimen_shopee.csv",
            mime="text/csv",
            type="primary"
        )


# =========================================================
# TAB 3: Dashboard & Eksplorasi Data
# =========================================================
with tabs[2]:
    st.markdown("#### Eksplorasi Dataset Asli Shopee Play Store")
    st.caption("Statistik 58.300 ulasan yang digunakan dalam pelatihan model machine learning.")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Total Scraping", "90.000")
    stat_col2.metric("Setelah Pembersihan", "58.300", help="Dihapus ulasan kosong & duplikat")
    stat_col3.metric("Positif (Training)", "49.57%")
    stat_col4.metric("Negatif (Training)", "44.13%")
    
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    # Pre-calculated distribution from README
    dist_df = pd.DataFrame({
        "Sentimen": ["Positif", "Negatif", "Netral"],
        "Jumlah Ulasan": [28901, 25730, 3669],
        "Persentase": [49.57, 44.13, 6.29]
    })
    
    c_pie, c_info = st.columns([1.2, 1])
    with c_pie:
        fig_master = go.Figure(data=[go.Pie(
            labels=dist_df["Sentimen"],
            values=dist_df["Jumlah Ulasan"],
            marker=dict(colors=["#10b981", "#ef4444", "#f59e0b"]),
            hoverinfo='label+percent+value',
            textinfo='label+percent'
        )])
        fig_master.update_layout(
            title_text="Distribusi Label Sentimen Dataset (58.300 Data)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_master, use_container_width=True)
        
    with c_info:
        st.markdown("""
        <div class="custom-card" style="height: 100%;">
            <h5 style="margin-top: 0; font-weight: 700; color: #0f172a;">Tentang Dataset</h5>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                Data dikumpulkan langsung dari Google Play Store pada aplikasi resmi <strong>Shopee Indonesia (<code>com.shopee.id</code>)</strong> menggunakan library <code>google-play-scraper</code>.
            </p>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                Pelabelan awal dilakukan melalui pendekatan <em>lexicon-based Indonesian sentiment dictionary</em> yang mengukur polaritas kata positif dan negatif, kemudian divalidasi dan dipelajari oleh model Machine Learning.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 4: Spesifikasi Model & Pipeline
# =========================================================
with tabs[3]:
    st.markdown("#### Arsitektur Model & Benchmark")
    st.caption("Hasil komparasi 4 arsitektur model dalam riset proyek ini.")
    
    benchmarks = pd.DataFrame({
        "Model": [
            "LSTM (Long Short-Term Memory)",
            "Bidirectional GRU",
            "Logistic Regression (Model Demo)",
            "DNN (Deep Neural Network)"
        ],
        "Representasi Fitur": [
            "Word Embedding (Tokenizer)",
            "Word Embedding (Tokenizer)",
            "TF-IDF Vectorizer (2.000 fitur)",
            "TF-IDF Vectorizer (5.000 fitur)"
        ],
        "Split Latih : Uji": [
            "80 : 20",
            "80 : 20",
            "70 : 30",
            "80 : 20"
        ],
        "Akurasi Test": [
            "92.20%",
            "92.11%",
            "91.37%",
            "90.42%"
        ],
        "Keunggulan": [
            "Tinggi, memahami dependensi sekuensial",
            "Efisiensi memori lebih baik dari LSTM",
            "Ringan, latensi sangat cepat (<10ms), interpretatif",
            "Fleksibel dengan arsitektur dense berlayer"
        ]
    })
    
    st.dataframe(benchmarks, use_container_width=True)
    
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-card">
        <h5 style="margin-top: 0; font-weight: 700; color: #0f172a;">Mengapa Logistic Regression Digunakan untuk Demo Live?</h5>
        <ul style="font-size: 0.88rem; color: #475569; line-height: 1.7; margin-bottom: 0;">
            <li><strong>Akurasi Kompetitif:</strong> Akurasi 91.37% hanya terpaut kurang dari 1% dari Deep Learning (LSTM 92.20%).</li>
            <li><strong>Latensi Eksekusi Ultra Cepat:</strong> Proses inferensi berlangsung dalam waktu di bawah 10 milidetik, sangat cocok untuk web app real-time.</li>
            <li><strong>Konsumsi Sumber Daya Ringan:</strong> Ukuran file model hanya ~48 KB dan TF-IDF ~73 KB, tidak membutuhkan GPU.</li>
            <li><strong>Transparansi Fitur:</strong> Bobot koefisien tiap kata dapat ditelusuri sehingga model tidak bertindak seperti <em>black-box</em>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
