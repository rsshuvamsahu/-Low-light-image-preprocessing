"""
Interactive Web Application for Low-Light Image Enhancement.
Built with Streamlit.

Features:
- Image Upload (JPG, PNG, WebP) + Sample Presets
- Method Selection: Classical (CLAHE, MSRCR, Gamma), Deep Learning CNN, and Hybrid Preprocessing Pipeline
- Real-Time Parameter Sliders (Gamma, CLAHE clip limit, Denoising, Retinex scales)
- Image Quality Assessment (Entropy, RMS Contrast, Luminance, PSNR/SSIM if Ground Truth provided)
- RGB & Luminance Histogram Analysis
- One-Click High-Quality PNG Export
"""

import os
import sys
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.classical import (
    gamma_correction,
    adaptive_gamma_correction,
    clahe_lab,
    clahe_hsv,
    msrcr,
    bilateral_denoise,
    gray_world_white_balance,
    ClassicalPipeline
)
from core.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_shannon_entropy,
    calculate_rms_contrast
)

# Page configuration
st.set_page_config(
    page_title="Low-Light Image Enhancement Suite",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E0E0E0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_deep_models():
    """Attempt to load PyTorch CNN models if available."""
    try:
        import torch
        from core.models import build_model, load_checkpoint
        from core.hybrid import HybridEnhancer

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model("res_cnn", device=dev)
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pth")
        if os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path, device=dev)
        hybrid = HybridEnhancer(model=model, device=dev)
        return {"model": model, "hybrid": hybrid, "device": dev, "available": True}
    except Exception as e:
        return {"available": False, "error": str(e)}


def plot_rgb_histograms(img_before: np.ndarray, img_after: np.ndarray):
    """Plot side-by-side RGB histogram comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    colors = ('r', 'g', 'b')

    # Before
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_before], [i], None, [256], [0, 256])
        axes[0].plot(hist, color=col, alpha=0.8, linewidth=1.5)
    axes[0].set_title("Input Histogram (Dark/Compressed)", fontsize=11, fontweight="bold")
    axes[0].set_xlim([0, 256])
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # After
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_after], [i], None, [256], [0, 256])
        axes[1].plot(hist, color=col, alpha=0.8, linewidth=1.5)
    axes[1].set_title("Enhanced Histogram (Expanded Dynamic Range)", fontsize=11, fontweight="bold")
    axes[1].set_xlim([0, 256])
    axes[1].set_xlabel("Pixel Value")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


def main():
    st.markdown("<div class='main-title'>🌌 Low-Light Image Preprocessing & Enhancement Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Classical Computer Vision vs. Deep Learning CNN vs. Hybrid Pipeline Benchmark</div>", unsafe_allow_html=True)

    dl_suite = load_deep_models()

    # Sidebar Controls
    st.sidebar.header("⚙️ Enhancement Settings")
    method_category = st.sidebar.selectbox(
        "Choose Pipeline Category",
        ["Classical Vision Suite", "Deep Learning CNN", "Hybrid Pipeline (Pre-Denoise + CNN + Guided)"]
    )

    # Image source
    st.sidebar.subheader("📁 Image Source")
    data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
    existing_samples = []
    if os.path.exists(data_folder):
        existing_samples = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    source_choice = st.sidebar.radio("Select Image Input", ["Upload Your Photo", "Select from Samples"])

    input_image_rgb = None
    ref_image_rgb = None

    if source_choice == "Upload Your Photo":
        uploaded_file = st.sidebar.file_uploader("Upload Dark Image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            input_image_rgb = np.array(pil_img)

        # Optional reference image
        ref_file = st.sidebar.file_uploader("Optional Ground Truth / Reference", type=["png", "jpg", "jpeg"])
        if ref_file is not None:
            ref_image_rgb = np.array(Image.open(ref_file).convert("RGB"))
    else:
        if existing_samples:
            selected_sample = st.sidebar.selectbox("Choose sample file", existing_samples)
            sample_path = os.path.join(data_folder, selected_sample)
            input_image_rgb = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)
        else:
            st.sidebar.info("No samples found in data/ folder. Using synthetic generator.")
            from core.classical import ensure_uint8
            h, w = 350, 500
            grad = np.linspace(0.05, 0.2, w)[None, :]
            dummy = np.ones((h, w, 3), dtype=np.float32) * grad[:, :, None]
            dummy[80:250, 100:300] = [0.08, 0.25, 0.12]
            dummy[120:220, 250:420] = [0.25, 0.10, 0.10]
            input_image_rgb = (dummy * 255).astype(np.uint8)

    if input_image_rgb is None:
        st.info("👈 Please upload an image or choose a sample from the sidebar to begin!")
        return

    # Method-Specific Parameters in Sidebar
    enhanced_rgb = None
    method_name = ""

    if method_category == "Classical Vision Suite":
        classical_algo = st.sidebar.selectbox(
            "Algorithm",
            ["Multi-Scale Retinex (MSRCR)", "Lab-Space CLAHE", "HSV-Space CLAHE", "Adaptive Gamma Correction", "Composite Classical Pipeline"]
        )

        if classical_algo == "Multi-Scale Retinex (MSRCR)":
            method_name = "Multi-Scale Retinex (MSRCR)"
            alpha = st.sidebar.slider("Color Restoration Alpha", 50.0, 250.0, 125.0)
            beta = st.sidebar.slider("Color Restoration Beta", 20.0, 100.0, 46.0)
            dyn_factor = st.sidebar.slider("Dynamic Range Factor", 1.0, 3.5, 2.0)
            enhanced_rgb = msrcr(input_image_rgb, alpha=alpha, beta=beta, dynamic_range_factor=dyn_factor)

        elif classical_algo == "Lab-Space CLAHE":
            method_name = "Lab-Space CLAHE"
            clip_lim = st.sidebar.slider("CLAHE Clip Limit", 1.0, 6.0, 2.5, step=0.5)
            grid = st.sidebar.slider("Tile Grid Size", 4, 16, 8, step=2)
            enhanced_rgb = clahe_lab(input_image_rgb, clip_limit=clip_lim, tile_grid_size=(grid, grid), is_rgb=True)

        elif classical_algo == "HSV-Space CLAHE":
            method_name = "HSV-Space CLAHE"
            clip_lim = st.sidebar.slider("CLAHE Clip Limit", 1.0, 6.0, 2.5, step=0.5)
            grid = st.sidebar.slider("Tile Grid Size", 4, 16, 8, step=2)
            enhanced_rgb = clahe_hsv(input_image_rgb, clip_limit=clip_lim, tile_grid_size=(grid, grid), is_rgb=True)

        elif classical_algo == "Adaptive Gamma Correction":
            method_name = "Adaptive Gamma Correction"
            alpha = st.sidebar.slider("Gamma Sensitivity (Alpha)", 0.2, 1.0, 0.5, step=0.05)
            enhanced_rgb = adaptive_gamma_correction(input_image_rgb, alpha=alpha)

        elif classical_algo == "Composite Classical Pipeline":
            method_name = "Composite Classical Pipeline"
            denoise = st.sidebar.selectbox("Denoising Filter", ["bilateral", "guided", "none"])
            strength = st.sidebar.slider("Denoise Strength", 10.0, 80.0, 35.0)
            enhancer = st.sidebar.selectbox("Illumination Enhancer", ["clahe_lab", "clahe_hsv", "msrcr", "gamma"])
            wb = st.sidebar.checkbox("Auto White Balance (Gray World)", value=True)

            pipe = ClassicalPipeline(
                denoise_type=denoise,
                denoise_strength=strength,
                enhancer_type=enhancer,
                apply_white_balance=wb
            )
            enhanced_rgb = pipe.process(input_image_rgb, is_rgb=True)

    elif method_category == "Deep Learning CNN":
        method_name = "Deep Learning CNN (LowLightResCNN)"
        if not dl_suite.get("available", False):
            st.sidebar.warning(f"PyTorch not loaded ({dl_suite.get('error')}). Using classical fallback.")
            enhanced_rgb = msrcr(input_image_rgb)
        else:
            import torch
            model = dl_suite["model"]
            dev = dl_suite["device"]
            ckpt_found = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pth"))
            if not ckpt_found:
                st.sidebar.info("💡 Note: Running initialized CNN. To train full weights on 485 LOL images, run `python src/train.py`.")

            t_in = torch.from_numpy(input_image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                out_t = model(t_in)
            out_np = (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            enhanced_rgb = out_np

    elif method_category == "Hybrid Pipeline (Pre-Denoise + CNN + Guided)":
        method_name = "Hybrid Preprocessing + CNN Pipeline"
        if not dl_suite.get("available", False):
            st.sidebar.warning("PyTorch not loaded. Running classical hybrid simulation.")
            pipe = ClassicalPipeline(denoise_type="bilateral", enhancer_type="clahe_lab", apply_white_balance=True)
            enhanced_rgb = pipe.process(input_image_rgb, is_rgb=True)
        else:
            hybrid = dl_suite["hybrid"]
            hybrid.pre_denoise = st.sidebar.checkbox("Pre-Stage: Bilateral Denoise", value=True)
            hybrid.denoise_strength = st.sidebar.slider("Pre-Denoise Strength", 10.0, 60.0, 25.0)
            hybrid.post_guided_refine = st.sidebar.checkbox("Post-Stage: Guided Detail Recovery", value=True)
            hybrid.guided_radius = st.sidebar.slider("Guided Filter Radius", 2, 8, 3)
            enhanced_rgb = hybrid.enhance(input_image_rgb, is_rgb=True)

    # Display Visual Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Low-Light Input")
        st.image(input_image_rgb, use_container_width=True)

    with col2:
        st.subheader(f"✨ Enhanced Output ({method_name})")
        st.image(enhanced_rgb, use_container_width=True)

    # Metric Dashboard
    st.markdown("---")
    st.subheader("📊 Image Quality & Information Metrics")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        ent_before = calculate_shannon_entropy(input_image_rgb)
        ent_after = calculate_shannon_entropy(enhanced_rgb)
        st.metric("Shannon Entropy (Information Richness)", f"{ent_after:.2f}", delta=f"{ent_after - ent_before:+.2f}")

    with m_col2:
        rms_before = calculate_rms_contrast(input_image_rgb)
        rms_after = calculate_rms_contrast(enhanced_rgb)
        st.metric("RMS Contrast", f"{rms_after:.3f}", delta=f"{rms_after - rms_before:+.3f}")

    with m_col3:
        mean_before = float(np.mean(input_image_rgb))
        mean_after = float(np.mean(enhanced_rgb))
        st.metric("Mean Luminance (0-255)", f"{mean_after:.1f}", delta=f"{mean_after - mean_before:+.1f}")

    with m_col4:
        if ref_image_rgb is not None:
            psnr_val = calculate_psnr(enhanced_rgb, ref_image_rgb)
            ssim_val = calculate_ssim(enhanced_rgb, ref_image_rgb)
            st.metric("PSNR (vs Ref)", f"{psnr_val:.2f} dB")
            st.metric("SSIM (vs Ref)", f"{ssim_val:.4f}")
        else:
            st.metric("Reference Status", "No-Reference Mode")

    # RGB Histograms
    st.markdown("---")
    st.subheader("📈 Dynamic Range Histogram Analysis")
    hist_fig = plot_rgb_histograms(input_image_rgb, enhanced_rgb)
    st.pyplot(hist_fig)

    # Download Button
    st.markdown("---")
    res_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", res_bgr)
    st.download_button(
        label="📥 Download Enhanced Image (PNG)",
        data=buffer.tobytes(),
        file_name="enhanced_output.png",
        mime="image/png"
    )


if __name__ == "__main__":
    main()
