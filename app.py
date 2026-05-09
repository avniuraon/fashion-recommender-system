import streamlit as st
import os
from PIL import Image
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="THREADS — Fashion Recommender",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.stApp { background: #f7f3ed; }

/* HEADER */
.site-header {
    padding: 20px 40px 16px;
    border-bottom: 0.5px solid rgba(26,18,8,0.15);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.75rem;
    font-weight: 300;
    letter-spacing: 0.35em;
    color: #1a1208;
}
.logo em { font-style: italic; color: #9b7a3f; }
.header-nav { display: flex; gap: 28px; font-size: 0.58rem; letter-spacing: 0.18em; text-transform: uppercase; color: #8a7d6b; }
.header-badge { background: #1a1208; color: #f7f3ed; font-size: 0.55rem; padding: 5px 14px; letter-spacing: 0.1em; text-transform: uppercase; }

/* HERO */
.hero { padding: 50px 40px 36px; border-bottom: 0.5px solid rgba(26,18,8,0.12); }
.eyebrow { display: flex; align-items: center; gap: 12px; font-size: 0.57rem; letter-spacing: 0.22em; text-transform: uppercase; color: #9b7a3f; margin-bottom: 18px; }
.eyebrow-line { width: 28px; height: 0.5px; background: #9b7a3f; display: inline-block; }
.hero-title { font-family: 'Cormorant Garamond', serif; font-size: clamp(2.6rem, 4.5vw, 4rem); font-weight: 300; line-height: 1.07; color: #1a1208; letter-spacing: -0.01em; margin-bottom: 16px; }
.hero-title em { font-style: italic; color: #9b7a3f; }
.hero-sub { font-size: 0.67rem; color: #8a7d6b; letter-spacing: 0.06em; line-height: 1.9; max-width: 540px; }

/* PIPELINE */
.pipeline { background: #e8e0d3; border-top: 0.5px solid rgba(26,18,8,0.1); border-bottom: 0.5px solid rgba(26,18,8,0.1); padding: 18px 40px; display: flex; align-items: flex-start; position: relative; }
.pipeline::before { content: ''; position: absolute; top: 32px; left: calc(40px + 48px); width: calc(100% - 80px - 96px); height: 0.5px; background: rgba(26,18,8,0.15); }
.pipe-step { flex: 1; text-align: center; }
.pipe-dot { width: 30px; height: 30px; border: 0.5px solid #1a1208; border-radius: 50%; margin: 0 auto 7px; display: flex; align-items: center; justify-content: center; font-size: 0.54rem; background: #f7f3ed; color: #1a1208; position: relative; z-index: 1; }
.pipe-dot.active { background: #1a1208; color: #f7f3ed; }
.pipe-name { font-size: 0.52rem; letter-spacing: 0.1em; text-transform: uppercase; color: #8a7d6b; line-height: 1.5; }

/* FILE UPLOADER */
[data-testid="stFileUploader"] { border: 1px dashed #c8a96e !important; background: rgba(200,169,110,0.03) !important; border-radius: 0 !important; padding: 18px !important; }
[data-testid="stFileUploader"]:hover { border-color: #9b7a3f !important; }

/* SECTION LABEL */
.sec-label { font-size: 0.55rem; letter-spacing: 0.22em; text-transform: uppercase; color: #9b7a3f; display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.sec-label::before { content: ''; display: inline-block; width: 16px; height: 0.5px; background: #9b7a3f; }

/* PANEL */
.panel-title { font-family: 'Cormorant Garamond', serif; font-size: 1.45rem; font-weight: 300; color: #1a1208; margin-bottom: 5px; }
.panel-sub { font-size: 0.6rem; color: #8a7d6b; letter-spacing: 0.07em; line-height: 1.75; margin-bottom: 20px; }

/* BUTTON */
.stButton > button { background: #1a1208 !important; color: #f7f3ed !important; border: none !important; border-radius: 0 !important; padding: 13px 32px !important; font-family: 'DM Mono', monospace !important; font-size: 0.6rem !important; font-weight: 400 !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; width: 100% !important; box-shadow: none !important; transition: background 0.2s !important; margin-top: 8px !important; }
.stButton > button:hover { background: #9b7a3f !important; }

/* IMG LABEL */
.img-label { font-size: 0.54rem; letter-spacing: 0.15em; text-transform: uppercase; color: #8a7d6b; padding: 4px 0 10px; }

/* RESULT CARD */
.result-card { background: #fff9f3; border: 0.5px solid rgba(26,18,8,0.12); overflow: hidden; transition: border-color 0.2s; }
.result-card:hover { border-color: #c8a96e; }
.rank-badge { background: #1a1208; color: #f7f3ed; font-size: 0.5rem; letter-spacing: 0.12em; padding: 3px 8px; text-transform: uppercase; display: inline-block; margin-bottom: 4px; }
.card-footer { padding: 7px 10px 10px; border-top: 0.5px solid rgba(26,18,8,0.08); }
.card-name { font-family: 'Cormorant Garamond', serif; font-size: 0.95rem; font-weight: 400; color: #1a1208; margin-bottom: 2px; }
.card-meta { font-size: 0.53rem; color: #8a7d6b; letter-spacing: 0.1em; text-transform: uppercase; }

/* TECH PILLS */
.tech-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 20px; }
.tech-pill { font-size: 0.5rem; letter-spacing: 0.12em; text-transform: uppercase; padding: 4px 10px; border: 0.5px solid rgba(26,18,8,0.18); color: #8a7d6b; }

/* EMPTY STATE */
.empty-state { border: 0.5px solid rgba(26,18,8,0.1); padding: 70px 40px; text-align: center; }
.empty-icon { font-size: 1.5rem; opacity: 0.18; margin-bottom: 16px; }
.empty-title { font-family: 'Cormorant Garamond', serif; font-size: 1.25rem; font-weight: 300; color: #1a1208; margin-bottom: 8px; }
.empty-sub { font-size: 0.58rem; letter-spacing: 0.12em; color: #8a7d6b; text-transform: uppercase; }

/* ERROR */
.err-box { border-left: 2px solid #c8a96e; padding: 14px 18px; background: rgba(200,169,110,0.06); font-size: 0.63rem; color: #5c5147; letter-spacing: 0.05em; line-height: 1.7; }

/* STATUS TEXT */
.status-text { font-size: 0.57rem; letter-spacing: 0.12em; text-transform: uppercase; color: #9b7a3f; margin: 6px 0 10px; }

/* PROGRESS */
div[data-testid="stProgress"] > div > div { background: #c8a96e !important; border-radius: 0 !important; }
.stSpinner > div { color: #9b7a3f !important; }

/* FOOTER */
.site-footer { border-top: 0.5px solid rgba(26,18,8,0.12); padding: 16px 40px; display: flex; justify-content: space-between; margin-top: 48px; }
.footer-note { font-size: 0.54rem; letter-spacing: 0.1em; color: #8a7d6b; text-transform: uppercase; }

footer, #MainMenu, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Load model & data ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_data():
    feat = np.array(pickle.load(open('features.pkl', 'rb')))
    fnames = pickle.load(open('filenames.pkl', 'rb'))
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    mdl = tensorflow.keras.Sequential([base, GlobalMaxPool2D()])
    return feat, fnames, mdl


def save_file(uploaded_file):
    os.makedirs('uploads', exist_ok=True)
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception:
        return False


def feature_extraction(img_path, mdl):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded)
    result = mdl.predict(preprocessed).flatten()
    return result / norm(result)


def get_recommendations(features, feat_list):
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    nbrs.fit(feat_list)
    dists, idxs = nbrs.kneighbors([features])
    return idxs, dists


def render_pipeline(active=0, placeholder=None):
    steps = [("01", "Upload"), ("02", "Extract"), ("03", "Normalise"), ("04", "KNN"), ("05", "Results")]
    dots = ""
    for i, (num, name) in enumerate(steps):
        cls = "active" if i < active else ""
        dots += f'<div class="pipe-step"><div class="pipe-dot {cls}">{num}</div><div class="pipe-name">{name}</div></div>'
    if placeholder:
        placeholder.markdown(f'<div class="pipeline">{dots}</div>', unsafe_allow_html=True)


# ─────────────────────────────── UI ───────────────────────────────────────────

st.markdown("""
<div class="site-header">
  <div class="logo">THREAD<em>s</em></div>
  <div class="header-nav"><span>Discover</span><span>Collections</span><span>Trending</span></div>
  <div class="header-badge">ResNet-50 · KNN</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow"><span class="eyebrow-line"></span> here we go</div>
  <div class="hero-title">Find your<br><em>perfect</em> match.</div>
  <div class="hero-sub">
    Upload any fashion item.
  </div>
</div>
""", unsafe_allow_html=True)

pipeline_ph = st.empty()
render_pipeline(0, pipeline_ph)

with st.spinner("Loading ResNet-50 model…"):
    feature_list, filenames, model = load_model_and_data()

col_left, col_right = st.columns([1, 1.65], gap="small")

# ── LEFT ──────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div style="padding: 36px 28px 0;">
      <div class="sec-label">Step 01 — Upload</div>
      <div class="panel-title">Drop your piece</div>
      <div class="panel-sub">Any clothing item, outfit, or accessory.<br>JPG · PNG · WEBP accepted.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding: 0 28px;">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a fashion image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            disp_img = Image.open(uploaded_file)
            st.markdown('<div class="img-label">Your uploaded piece</div>', unsafe_allow_html=True)
            st.image(disp_img, use_container_width=True)
            analyze_btn = st.button("◈  Find Similar Items  →")
        else:
            st.markdown("""
            <div style="text-align:center; padding:50px 20px; color:#8a7d6b;">
              <div style="font-size:1.6rem; opacity:0.18; margin-bottom:14px;">◈</div>
              <div style="font-size:0.6rem; letter-spacing:0.15em; text-transform:uppercase;">Upload a fashion photo to begin</div>
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 16px 28px 0;">
      <div class="tech-row">
        
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col_right:
    if uploaded_file is not None and analyze_btn:
        st.markdown('<div style="padding: 36px 32px 0;">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Processing</div>', unsafe_allow_html=True)

        prog = st.progress(0)
        status = st.empty()

        saved = save_file(uploaded_file)

        if saved:
            for step_num, pct, label in [
                (1, 18, "Saving uploaded image…"),
                (2, 38, "Preprocessing → 224×224 RGB…"),
                (3, 60, "Running ResNet-50 inference…"),
                (4, 80, "Applying L2 normalisation…"),
                (5, 94, "Searching embeddings with KNN…"),
            ]:
                render_pipeline(step_num, pipeline_ph)
                status.markdown(f'<div class="status-text">{label}</div>', unsafe_allow_html=True)
                prog.progress(pct)
                time.sleep(0.3)

            try:
                feats = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
                indices, distances = get_recommendations(feats, feature_list)

                prog.progress(100)
                render_pipeline(5, pipeline_ph)
                status.markdown('<div class="status-text">Complete — 5 matches found.</div>', unsafe_allow_html=True)

                # ── BEST MATCH (large) ──
                st.markdown("""
                <div style="margin-top: 26px;">
                  <div class="sec-label">Best Match</div>
                  <div class="panel-title">Closest Visual Match</div>
                  <div class="panel-sub">Highest embedding similarity from your catalogue</div>
                </div>
                """, unsafe_allow_html=True)

                top_idx = indices[0][1]
                top_dist = distances[0][1]
                top_sim = round(max(0, (1 - top_dist)) * 100, 1)

                top_col, _ = st.columns([1, 0.55])
                with top_col:
                    try:
                        top_img = Image.open(filenames[top_idx])
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        fname = os.path.basename(filenames[top_idx]).split('.')[0].replace('_', ' ').title()
                        st.markdown('<span class="rank-badge">★ Best Match</span>', unsafe_allow_html=True)
                        st.image(top_img, use_container_width=True)
                        st.markdown(f"""
                        <div class="card-footer">
                          <div class="card-name">{fname}</div>
                          <div class="card-meta">Rank 01 · {top_sim}% similarity</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception:
                        st.image(filenames[top_idx], use_container_width=True)

                # ── MORE MATCHES (4 cards) ──
                st.markdown("""
                <div style="margin: 26px 0 10px;">
                  <div class="sec-label">More Similar Items</div>
                </div>
                """, unsafe_allow_html=True)

                rec_cols = st.columns(4, gap="small")
                for col_i, rank in enumerate(range(2, 6)):
                    img_idx = indices[0][rank]
                    dist = distances[0][rank]
                    sim = round(max(0, (1 - dist)) * 100, 1)
                    with rec_cols[col_i]:
                        try:
                            rec_img = Image.open(filenames[img_idx])
                            fname = os.path.basename(filenames[img_idx]).split('.')[0].replace('_', ' ').title()
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.markdown(f'<span class="rank-badge">#{rank:02d}</span>', unsafe_allow_html=True)
                            st.image(rec_img, use_container_width=True)
                            st.markdown(f"""
                            <div class="card-footer">
                              <div class="card-name">{fname[:20]}</div>
                              <div class="card-meta">{sim}% match</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception:
                            st.image(filenames[img_idx], use_container_width=True)

            except Exception as e:
                prog.empty()
                status.empty()
                st.markdown(f'<div class="err-box">⚠ Feature extraction failed: {e}</div>', unsafe_allow_html=True)

        else:
            prog.empty()
            status.empty()
            st.markdown('<div class="err-box">⚠ Could not save uploaded file. Check write permissions for the uploads/ folder.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding: 36px 32px;">
          <div class="sec-label">Step 02–05 — Results</div>
          <div class="empty-state">
            <div class="empty-icon">◈</div>
            <div class="empty-title">Your recommendations will appear here</div>
            <div class="empty-sub">Upload an image and click Find Similar Items</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
  <span class="footer-note">THREADS · Fashion Recommender</span>
  <span class="footer-note">ResNet-50 · features.pkl · filenames.pkl</span>
</div>
""", unsafe_allow_html=True)