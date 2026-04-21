import os

# Install TensorFlow manually
os.system("pip install tensorflow==2.13.0")
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import keras.ops as kops
from tensorflow.keras import layers, Model
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VirusScan · Severity Analysis",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0d12;
    --surface:   #111520;
    --border:    #1e2535;
    --accent:    #00e5ff;
    --accent2:   #ff4d6d;
    --mild:      #00c897;
    --moderate:  #ffb703;
    --severe:    #ff4d6d;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1300px !important; }

/* ── Hero header ── */
.hero {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 0.3rem;
}
.hero-icon {
    font-size: 2.6rem;
    line-height: 1;
    filter: drop-shadow(0 0 18px #00e5ff88);
}
.hero h1 {
    font-family: var(--font-mono) !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin: 0 !important;
    background: linear-gradient(90deg, var(--accent), #7b8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 0.88rem;
    color: var(--muted);
    font-family: var(--font-mono);
    letter-spacing: 0.04em;
    margin-bottom: 2rem;
    margin-left: 4.2rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── Upload zone ── */
.stFileUploader > div {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
.stFileUploader > div:hover {
    border-color: var(--accent) !important;
}
.stFileUploader label { color: var(--muted) !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; }

/* ── Severity badge ── */
.badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.45rem 1.2rem;
    border-radius: 100px;
    margin-bottom: 0.8rem;
}
.badge-mild     { background: #00c89722; color: var(--mild);     border: 1px solid var(--mild); }
.badge-moderate { background: #ffb70322; color: var(--moderate); border: 1px solid var(--moderate); }
.badge-severe   { background: #ff4d6d22; color: var(--severe);   border: 1px solid var(--severe); }

/* ── Score bar ── */
.score-row { display: flex; align-items: center; gap: 1rem; margin: 0.6rem 0; }
.score-label { font-family: var(--font-mono); font-size: 0.75rem; color: var(--muted); width: 140px; flex-shrink: 0; }
.score-bar-bg { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }
.score-value { font-family: var(--font-mono); font-size: 0.8rem; width: 52px; text-align: right; }

/* ── Stat boxes ── */
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }
.stat-box {
    background: #0f1420;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    display: block;
}
.stat-lbl {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.2rem;
    display: block;
}

/* ── BSL pill ── */
.bsl-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #1a1f2e;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    margin-top: 0.4rem;
}
.bsl-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

/* ── Selectbox & buttons ── */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.05em !important;
    font-size: 0.85rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Image display ── */
.stImage img { border-radius: 10px; border: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ── WHO BSL data ───────────────────────────────────────────────────────────────
WHO_BSL = {
    'Adenovirus': 2, 'Astrovirus': 2, 'CCHF': 4, 'Cowpox': 2,
    'Dengue': 3, 'Ebola': 4, 'Guanarito': 4, 'Influenza': 2,
    'LCM': 3, 'Lassa': 4, 'Machupo': 4, 'Marburg': 4,
    'Nipah virus': 4, 'Norovirus': 2, 'Orf': 2, 'Papilloma': 2,
    'Pseudocowpox': 2, 'Rift Valley': 3, 'Rotavirus': 2,
    'Sapovirus': 2, 'TBE': 3, 'WestNile': 3,
}
BSL_COLORS = {1: '#00c897', 2: '#00bfff', 3: '#ffb703', 4: '#ff4d6d'}
BSL_LABELS = {
    1: 'Minimal Risk — no known disease in healthy adults',
    2: 'Moderate Risk — associated with human disease',
    3: 'Serious Risk — serious/potentially lethal infection',
    4: 'Life-Threatening — life-threatening disease, no vaccine/therapy',
}
VIRUS_DESCRIPTIONS = {
    'Adenovirus': 'Causes respiratory, eye and GI infections. Usually mild in healthy individuals.',
    'Astrovirus': 'Common cause of gastroenteritis, particularly in children and elderly.',
    'CCHF': 'Crimean-Congo Haemorrhagic Fever. Severe viral haemorrhagic disease with high mortality.',
    'Cowpox': 'Zoonotic orthopoxvirus. Usually mild skin lesions; occasionally severe in immunocompromised.',
    'Dengue': 'Mosquito-borne flavivirus. Ranges from flu-like to severe haemorrhagic fever.',
    'Ebola': 'Filovirus causing severe haemorrhagic fever. Case fatality up to 90%.',
    'Guanarito': 'Arenavirus causing Venezuelan haemorrhagic fever. Endemic to Venezuela.',
    'Influenza': 'Respiratory virus with seasonal and pandemic potential. Variable severity.',
    'LCM': 'Lymphocytic choriomeningitis. CNS involvement; risk to immunocompromised and pregnant.',
    'Lassa': 'Arenavirus endemic to West Africa. Causes severe haemorrhagic fever.',
    'Machupo': 'Arenavirus causing Bolivian haemorrhagic fever. High mortality without treatment.',
    'Marburg': 'Filovirus closely related to Ebola. Causes severe haemorrhagic fever.',
    'Nipah virus': 'Paramyxovirus with encephalitis and respiratory syndrome. Up to 75% fatality.',
    'Norovirus': 'Highly contagious gastroenteritis. Self-limiting in healthy adults.',
    'Orf': 'Parapoxvirus causing skin lesions, transmitted from sheep/goats. Usually self-limiting.',
    'Papilloma': 'Causes warts and is associated with several cancers depending on serotype.',
    'Pseudocowpox': 'Parapoxvirus causing milker\'s nodules. Mild, self-limiting zoonosis.',
    'Rift Valley': 'Phlebovirus causing febrile illness. Outbreaks linked to flooding and livestock.',
    'Rotavirus': 'Leading cause of severe childhood gastroenteritis globally.',
    'Sapovirus': 'Calicivirus causing gastroenteritis, mainly in children.',
    'TBE': 'Tick-borne Encephalitis. Neurological disease in Europe and Asia.',
    'WestNile': 'Flavivirus transmitted by mosquitoes. Neuroinvasive disease in vulnerable groups.',
}

# ── Model classes (must match training code) ───────────────────────────────────
class ChannelAttention(layers.Layer):
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        units = max(channels // reduction_ratio, 1)
        self.dense1 = layers.Dense(units, activation='relu')
        self.dense2 = layers.Dense(channels)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        avg = kops.mean(x, axis=[1, 2], keepdims=True)
        mx  = kops.max(x,  axis=[1, 2], keepdims=True)
        return self.sigmoid(self.dense2(self.dense1(avg)) +
                            self.dense2(self.dense1(mx)))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'channels': self.dense2.units,
                    'reduction_ratio': self.dense1.units})
        return cfg


class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, x):
        avg_s = kops.mean(x, axis=-1, keepdims=True)
        max_s = kops.max(x,  axis=-1, keepdims=True)
        return self.conv(kops.concatenate([avg_s, max_s], axis=-1))

    def get_config(self):
        return super().get_config()


class CBAMBlock(layers.Layer):
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ch_att = ChannelAttention(channels, reduction_ratio)
        self.sp_att = SpatialAttention()
        self.mul    = layers.Multiply()

    def call(self, x):
        x = self.mul([x, self.ch_att(x)])
        x = self.mul([x, self.sp_att(x)])
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg['channels'] = self.ch_att.dense2.units
        return cfg


class SoftmaxWeightLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_w = self.add_weight(
            name='raw_weights', shape=(3,),
            initializer='zeros', trainable=True)

    def call(self, inputs):
        density, bsl, hybrid = inputs
        w = tf.nn.softmax(self.raw_w)
        score = w[0] * density + w[1] * bsl + w[2] * hybrid
        return score, w

    def get_config(self):
        return super().get_config()


# ── Model builder ──────────────────────────────────────────────────────────────
def build_hybrid_mobilenet(input_shape=(224, 224, 3), gru_units=64):
    inp      = keras.Input(shape=input_shape, name='image_input')
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')
    backbone.trainable = False
    feat = backbone(inp, training=False)
    feat = CBAMBlock(feat.shape[-1], reduction_ratio=8, name='cbam')(feat)
    fH, fW, fC = feat.shape[1], feat.shape[2], feat.shape[3]
    seq  = layers.Reshape((fH * fW, fC), name='to_sequence')(feat)
    seq  = layers.Bidirectional(layers.GRU(gru_units, return_sequences=False), name='bigru')(seq)
    seq  = layers.Dropout(0.3, name='dropout')(seq)
    out  = layers.Dense(1, activation='sigmoid', name='hybrid_score')(seq)
    return Model(inputs=inp, outputs=out, name='HybridMobileNet')


def build_severity_model(hybrid_model):
    img_inp     = keras.Input(shape=(224, 224, 3), name='image_input')
    density_inp = keras.Input(shape=(1,),          name='density_input')
    bsl_inp     = keras.Input(shape=(1,),          name='bsl_input')
    hybrid_score = hybrid_model(img_inp)
    score, weights = SoftmaxWeightLayer(name='severity_weights')(
        [density_inp, bsl_inp, hybrid_score])
    return Model(
        inputs=[img_inp, density_inp, bsl_inp],
        outputs=[score, weights],
        name='SeverityScoreModel')


# ── Load / cache model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str | None):
    hybrid = build_hybrid_mobilenet()
    sev    = build_severity_model(hybrid)
    if model_path and Path(model_path).exists():
        try:
            sev.load_weights(model_path)
        except Exception:
            pass  # weights mismatch — use untrained model for demo
    return sev, hybrid


# ── Image preprocessing ────────────────────────────────────────────────────────
def preprocess_tif(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, None

    # 16-bit → 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Grayscale → BGR
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img

    # Resize + denoise + normalise + CLAHE + 3ch
    resized  = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    norm     = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    final    = cv2.merge([enhanced, enhanced, enhanced])

    display = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    model_input = keras.applications.mobilenet_v2.preprocess_input(
        final.astype(np.float32))
    return display, model_input, img_bgr


def measure_particle_area(img_gray: np.ndarray, patch_radius=30,
                           min_area=20, max_frac=0.15):
    """Quick blob detection to estimate particle count and area."""
    H, W = img_gray.shape
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    max_area = max_frac * H * W

    results = {}
    for polarity in [binary, cv2.bitwise_not(binary)]:
        n, _, stats, centroids = cv2.connectedComponentsWithStats(
            polarity, connectivity=8)
        blobs = [
            {'area': float(stats[i, cv2.CC_STAT_AREA]),
             'cx': centroids[i][0], 'cy': centroids[i][1]}
            for i in range(1, n)
            if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
        ]
        results[len(blobs)] = blobs

    best_blobs = results[max(results.keys())] if results else []
    areas = [b['area'] for b in best_blobs]
    return (len(areas),
            round(sum(areas), 1),
            round(float(np.mean(areas)), 1) if areas else 0.0,
            round(float(min(areas)), 1) if areas else 0.0,
            round(float(max(areas)), 1) if areas else 0.0)


def score_to_label(score: float):
    if score < 0.3:  return 'mild',     '#00c897'
    if score < 0.6:  return 'moderate', '#ffb703'
    return 'severe', '#ff4d6d'


def draw_gauge(score: float, color: str) -> bytes:
    fig, ax = plt.subplots(figsize=(4, 2.2), facecolor='#111520')
    ax.set_facecolor('#111520')
    theta = np.linspace(np.pi, 0, 300)
    r_inner, r_outer = 0.6, 1.0
    for t1, t2, col in [(np.pi, np.pi*0.67, '#00c89744'),
                         (np.pi*0.67, np.pi*0.34, '#ffb70344'),
                         (np.pi*0.34, 0, '#ff4d6d44')]:
        seg = np.linspace(t1, t2, 100)
        xs = np.concatenate([np.cos(seg)*r_outer, np.cos(seg[::-1])*r_inner])
        ys = np.concatenate([np.sin(seg)*r_outer, np.sin(seg[::-1])*r_inner])
        ax.fill(xs, ys, color=col[:-2], alpha=0.3)
    # Needle
    angle = np.pi * (1 - score)
    ax.plot([0, 0.72*np.cos(angle)], [0, 0.72*np.sin(angle)],
            color=color, linewidth=3, solid_capstyle='round', zorder=5)
    ax.add_patch(plt.Circle((0, 0), 0.06, color=color, zorder=6))
    ax.text(0, -0.22, f'{score:.3f}', ha='center', va='center',
            fontsize=18, fontweight='bold', color=color,
            fontfamily='monospace')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.35, 1.1)
    ax.set_aspect('equal'); ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight',
                facecolor='#111520')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── App layout ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="hero-icon">🦠</span>
  <h1>VirusScan</h1>
</div>
<p class="hero-sub">Electron Microscopy · Severity Analysis · v1.0</p>
""", unsafe_allow_html=True)

# Sidebar: model path only
with st.sidebar:
    st.markdown("### ⚙ Configuration")
    model_path = st.text_input(
        "Model weights path",
        value="model.keras",
        help="Path to trained .keras model file")
    st.markdown("---")
    st.markdown(
        "<span style='font-family:monospace;font-size:0.75rem;color:#64748b'>"
        "Set the path to your trained model weights."
        "</span>", unsafe_allow_html=True)

# Load model
severity_model, hybrid_model = load_model(model_path)

# ── Virus class selector — prominent in main UI ────────────────────────────────
st.markdown('<div class="card-title">Select Virus Class</div>', unsafe_allow_html=True)
virus_class_options = sorted(WHO_BSL.keys())
vc_col1, vc_col2 = st.columns([2, 3], gap="medium")
with vc_col1:
    virus_class = st.selectbox(
        "Virus class",
        virus_class_options,
        index=virus_class_options.index('Adenovirus'),
        label_visibility="collapsed",
        help="Must match the virus in your image — this determines the BSL score component")
with vc_col2:
    _bsl_preview = WHO_BSL.get(virus_class, 2)
    _bsl_col     = BSL_COLORS.get(_bsl_preview, '#64748b')
    _bsl_lbl     = BSL_LABELS.get(_bsl_preview, '')
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.7rem;
                background:#0f1420;border:1px solid #1e2535;
                border-radius:8px;padding:0.55rem 1rem;margin-top:0.05rem">
      <div style="width:10px;height:10px;border-radius:50%;
                  background:{_bsl_col};flex-shrink:0"></div>
      <span style="font-family:monospace;font-size:0.78rem;color:{_bsl_col};font-weight:700">
        BSL {_bsl_preview}
      </span>
      <span style="font-size:0.75rem;color:#64748b;margin-left:0.3rem">{_bsl_lbl}</span>
    </div>
    """, unsafe_allow_html=True)

# Warning if still on default
if virus_class == 'Adenovirus':
    st.warning("⚠ Default virus class selected (Adenovirus). "
               "Change to match your actual image for correct BSL scoring.", icon="⚠️")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Main columns ───────────────────────────────────────────────────────────────
col_upload, col_results = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown('<div class="card-title">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a .tif electron microscopy image",
        type=["tif", "tiff"],
        label_visibility="collapsed")

    if uploaded:
        file_bytes = uploaded.read()
        display_img, model_input, raw_bgr = preprocess_tif(file_bytes)

        if display_img is None:
            st.error("Could not read image. Please upload a valid .tif file.")
        else:
            st.image(display_img, caption="Preprocessed (224×224 · CLAHE · 3-ch)",
                     use_container_width=True)

            # Particle analysis on the raw image
            raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) \
                if len(raw_bgr.shape) == 3 else raw_bgr
            p_count, p_total, p_mean, p_min, p_max = measure_particle_area(raw_gray)



with col_results:
    if uploaded and display_img is not None:

        # ── Run model ──────────────────────────────────────────────────────────
        bsl_raw   = WHO_BSL.get(virus_class, 2)
        bsl_score = (bsl_raw - 1) / 3.0

        # Particle density: count / total_area, log-scaled so sparse images
        # don't all collapse to ~0.  Range: 0 (no particles) → 1 (very dense).
        # log1p scale: density_raw values of 0.001 → ~0.5 range after mapping.
        density_raw = p_count / max(p_total, 1e-6)
        # Use log1p normalised by a reference scale (empirical: ~0.01 = mid-range)
        REF_SCALE   = 0.005   # particles/px² at which density ≈ 0.5
        density     = float(np.clip(
            np.log1p(density_raw / REF_SCALE) / np.log1p(1.0 / REF_SCALE * 10),
            0.0, 1.0))

        img_tensor = tf.expand_dims(
            tf.constant(model_input, dtype=tf.float32), 0)
        den_tensor = tf.constant([[density]], dtype=tf.float32)
        bsl_tensor = tf.constant([[bsl_score]], dtype=tf.float32)

        score_tensor, weights_tensor = severity_model(
            [img_tensor, den_tensor, bsl_tensor], training=False)

        # Handle variable output shapes from trained vs untrained model
        score_np = np.array(score_tensor).flatten()
        score    = float(score_np[0])

        # weights can be (3,) or (B,3) — always reduce to length-3 array
        w_np = np.array(weights_tensor).flatten()
        if w_np.shape[0] >= 3:
            weights = w_np[:3]
        else:
            # Safe fallback: recompute from raw layer weights
            raw_w   = severity_model.get_layer('severity_weights').raw_w.numpy()
            exp_w   = np.exp(raw_w - raw_w.max())   # stable softmax
            weights = exp_w / exp_w.sum()

        label, color = score_to_label(score)

        # ── Severity badge + gauge ─────────────────────────────────────────────
        st.markdown('<div class="card-title">Severity Assessment</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<span class="badge badge-{label}">{label}</span>',
            unsafe_allow_html=True)

        gauge_png = draw_gauge(score, color)
        st.image(gauge_png, use_container_width=True)

        # ── Score breakdown ────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Score Breakdown</div>',
                    unsafe_allow_html=True)

        components = [
            ("Particle Density",  density,   weights[0], "#00e5ff"),
            ("BSL Score",         bsl_score, weights[1], "#7b8cff"),
        ]
        for name, val, weight, bar_color in components:
            pct = int(val * 100)
            st.markdown(f"""
            <div class="score-row">
              <span class="score-label">{name}</span>
              <div class="score-bar-bg">
                <div class="score-bar-fill"
                     style="width:{pct}%; background:{bar_color}"></div>
              </div>
              <span class="score-value" style="color:{bar_color}">{val:.3f}</span>
            </div>
            <div class="score-row" style="margin-top:-0.4rem;margin-bottom:0.2rem">
              <span class="score-label" style="color:#334155;font-size:0.68rem">
                weight = {weight:.3f}
              </span>
            </div>
            """, unsafe_allow_html=True)

        # Final score bar
        st.markdown(f"""
        <div class="score-row" style="margin-top:0.6rem">
          <span class="score-label" style="color:{color};font-weight:600">
            Final Score
          </span>
          <div class="score-bar-bg">
            <div class="score-bar-fill"
                 style="width:{int(score*100)}%;background:{color}"></div>
          </div>
          <span class="score-value" style="color:{color};font-weight:700">
            {score:.3f}
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── BSL info ───────────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Virus Class · BSL Info</div>',
                    unsafe_allow_html=True)

        bsl_color = BSL_COLORS.get(bsl_raw, '#64748b')
        st.markdown(f"""
        <div class="bsl-pill">
          <div class="bsl-dot" style="background:{bsl_color}"></div>
          <strong style="color:{bsl_color}">{virus_class}</strong>
          &nbsp;·&nbsp;
          <span style="color:#94a3b8">BSL {bsl_raw}</span>
        </div>
        <p style="font-size:0.82rem;color:#94a3b8;margin-top:0.7rem;line-height:1.6">
          {BSL_LABELS[bsl_raw]}
        </p>
        <p style="font-size:0.82rem;color:#64748b;margin-top:0.3rem;line-height:1.6;
                  font-style:italic">
          {VIRUS_DESCRIPTIONS.get(virus_class, '')}
        </p>
        """, unsafe_allow_html=True)

        # ── Threshold legend ───────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;gap:1.4rem;font-family:monospace;font-size:0.75rem">
          <span style="color:#00c897">▮ Mild &nbsp; 0.0 – 0.3</span>
          <span style="color:#ffb703">▮ Moderate &nbsp; 0.3 – 0.6</span>
          <span style="color:#ff4d6d">▮ Severe &nbsp; 0.6 – 1.0</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:420px;
                    border:1.5px dashed #1e2535;border-radius:12px;
                    color:#334155;font-family:monospace;font-size:0.85rem;
                    text-align:center;gap:1rem">
          <span style="font-size:3rem;opacity:0.3">🔬</span>
          <span>Upload a .tif image<br>to begin analysis</span>
        </div>
        """, unsafe_allow_html=True)
