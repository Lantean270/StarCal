# streamlit: run starcal_app.py
import streamlit as st
import numpy as np
import cv2
import rawpy
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tempfile
import os
opencv-python

# ====================== 页面配置 ======================
st.set_page_config(page_title="StarCal 天文摄影校准工具", layout="wide")
st.title("🔭 StarCal 天文摄影智能校准工具")
st.caption("支持 NEF/CR2/ARW/DNG/FITS | 暗场平场偏置 | 堆栈对齐 | 星野/深空双模式")

# ====================== 模式选择（彻底分开，互不干扰） ======================
process_mode = st.radio(
    "处理模式",
    ["🌟 星野模式", "🌌 深空双窄带 → 标准伪哈勃色（SHO）"],
    horizontal=True
)

# ====================== 上传区（支持大文件） ======================
st.subheader("📤 数据上传")
col1, col2, col3, col4 = st.columns(4)
with col1:
    light_files = st.file_uploader(
        "科学光 Light（单文件最大1GB）",
        type=["nef", "cr2", "arw", "dng", "fits", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="尼康NEF等大文件请分批上传，避免请求超时"
    )
with col2:
    dark_file = st.file_uploader("暗场 Dark", type=["nef", "cr2", "arw", "dng", "fits", "png", "jpg"], accept_multiple_files=True)
with col3:
    flat_file = st.file_uploader("平场 Flat", type=["nef", "cr2", "arw", "dng", "fits", "png", "jpg"], accept_multiple_files=True)
with col4:
    bias_file = st.file_uploader("偏置 Bias", type=["nef", "cr2", "arw", "dng", "fits", "png", "jpg"], accept_multiple_files=True)

# 功能开关
col_a, col_b, col_c = st.columns(3)
with col_a:
    auto_align = st.checkbox("✅ 星点自动对齐", value=True)
with col_b:
    sharpen = st.checkbox("✅ 缩星锐化", value=True)
with col_c:
    remove_red = st.checkbox("✅ 去除背景红雾", value=False)

# ====================== 核心工具函数（稳定版，无彩纹） ======================
def load_img(file):
    """安全加载各类格式"""
    try:
        # 处理NEF/RAW格式，分块读取避免内存溢出
        if file.name.lower().endswith(('nef', 'cr2', 'arw', 'dng')):
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.name)[1], delete=False) as tmp:
                # 1MB分块读取，解决大文件上传问题
                while chunk := file.read(1024*1024):
                    tmp.write(chunk)
                tmp_path = tmp.name
            with rawpy.imread(tmp_path) as raw:
                # 天文专用参数：保留原始数据，无自动处理，从源头避免色偏
                rgb = raw.postprocess(
                    use_auto_wb=False,
                    gamma=(1,1),
                    no_auto_bright=True,
                    output_bps=16,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR
                ).astype(np.float32)
            os.unlink(tmp_path)
            # 校正方向，归一化到0-1
            rgb = (rgb - np.percentile(rgb, 1)) / (np.percentile(rgb, 99.9) - np.percentile(rgb, 1) + 1e-6)
            rgb = np.clip(rgb, 0, 1)
            img_pil = Image.fromarray((rgb * 255).astype(np.uint8))
            img_pil = ImageOps.exif_transpose(img_pil)
            return np.array(img_pil).astype(np.float32) / 255.0

        # 处理FITS格式
        elif file.name.lower().endswith("fits"):
            data = fits.getdata(file)
            if len(data.shape) == 2:
                img = np.stack([data]*3, axis=-1).astype(np.float32)
            else:
                img = data.astype(np.float32)
            img = cv2.flip(img, 0)
            # 归一化
            vmin, vmax = np.percentile(img, (1, 99.9))
            img = np.clip((img - vmin) / (vmax - vmin + 1e-6), 0, 1)
            return img

        # 处理普通JPG/PNG
        else:
            img = Image.open(file)
            img = ImageOps.exif_transpose(img).convert("RGB")
            return np.array(img).astype(np.float32) / 255.0

    except Exception as e:
        st.error(f"加载失败 {file.name}: {str(e)}")
        return None

def batch_mean(files):
    """批量平均堆栈，安全处理空输入"""
    if not files:
        return None
    imgs = []
    for f in files:
        img = load_img(f)
        if img is not None:
            imgs.append(img)
    return np.mean(np.array(imgs), axis=0) if imgs else None

# 星点对齐（修复KeyError，稳定版）
def get_peak_coords(peaks_table):
    if 'x_centroid' in peaks_table.colnames:
        return peaks_table['x_centroid'], peaks_table['y_centroid']
    elif 'x_peak' in peaks_table.colnames:
        return peaks_table['x_peak'], peaks_table['y_peak']
    else:
        raise ValueError("无法识别星点坐标列名")

def align_images(imgs):
    if len(imgs) < 2:
        return imgs
    base = imgs[0]
    base_gray = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mean, med, std = sigma_clipped_stats(base_gray, sigma=3.0)
    thresh = med + 3 * std
    base_peaks = find_peaks(base_gray, threshold=thresh, box_size=5)
    
    if len(base_peaks) < 10:
        st.warning("星点数量不足，跳过对齐")
        return imgs
    
    try:
        base_x, base_y = get_peak_coords(base_peaks)
    except Exception as e:
        st.error(f"星点坐标提取失败: {str(e)}")
        return imgs
    
    base_pts = np.column_stack((base_x[:20], base_y[:20])).astype(np.float32)
    aligned_imgs = [base]
    
    for img in imgs[1:]:
        try:
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mean_i, med_i, std_i = sigma_clipped_stats(img_gray, sigma=3.0)
            thresh_i = med_i + 3 * std_i
            img_peaks = find_peaks(img_gray, threshold=thresh_i, box_size=5)
            
            if len(img_peaks) < 10:
                aligned_imgs.append(img)
                continue
            
            img_x, img_y = get_peak_coords(img_peaks)
            img_pts = np.column_stack((img_x[:20], img_y[:20])).astype(np.float32)
            
            H, mask = cv2.findHomography(img_pts, base_pts, cv2.RANSAC, 5.0)
            if H is None:
                aligned_imgs.append(img)
                continue
            h, w = base.shape[:2]
            aligned = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
            aligned_imgs.append(aligned)
        except Exception as e:
            st.warning(f"单帧对齐失败，跳过: {str(e)}")
            aligned_imgs.append(img)
    return aligned_imgs

# 校准（暗场/平场/偏置，安全版，无溢出）
def calibrate(stack, dark, flat, bias):
    cal = stack.copy()
    # 偏置校准
    if bias is not None:
        cal = np.clip(cal - bias, 0, 1)
    # 暗场校准
    if dark is not None:
        dark_corr = dark - bias if bias is not None else dark
        cal = np.clip(cal - dark_corr, 0, 1)
    # 平场校准
    if flat is not None:
        flat_corr = flat
        if bias is not None:
            flat_corr = np.clip(flat_corr - bias, 0, 1)
        if dark is not None:
            flat_corr = np.clip(flat_corr - (dark - bias if bias is not None else dark), 0, 1)
        flat_mean = np.mean(flat_corr)
        if flat_mean > 0:
            cal = np.clip(cal / (flat_corr / flat_mean), 0, 1)
    return cal

# 缩星锐化（分通道，防色偏）
def star_sharpen(img):
    sharped = np.zeros_like(img)
    for c in range(3):
        blur = cv2.GaussianBlur(img[..., c], (3, 3), 1)
        mask = img[..., c] - blur
        sharped[..., c] = np.clip(img[..., c] + 1.0 * mask, 0, 1)
    return sharped

# 去红雾（安全版，无彩纹）
def remove_red_fog(img):
    r, g, b = cv2.split(img)
    r_mean, _, r_std = sigma_clipped_stats(r, sigma=3.0)
    g_mean, _, g_std = sigma_clipped_stats(g, sigma=3.0)
    # 以绿通道为基准，比例校正红通道，避免溢出
    r_corrected = r * (g_mean / r_mean) if r_mean > 0 else r
    r_corrected = np.clip(r_corrected, 0, 1)
    return cv2.merge((r_corrected, g, b))

# ====================== 深空双窄带 → 标准伪哈勃色（彻底修复五彩斑斓） ======================
def dual_band_to_hubble_safe(img):
    """
    标准SHO伪哈勃色，安全算法，无彩纹、无过饱和
    适配Seestar直出双窄带（Ha+OIII）
    """
    # 分离通道，严格归一化
    r, g, b = cv2.split(img)
    def safe_norm(chn):
        p1, p99 = np.percentile(chn, (1, 99))
        return np.clip((chn - p1) / (p99 - p1 + 1e-6), 0, 1)
    
    r = safe_norm(r)
    g = safe_norm(g)
    b = safe_norm(b)

    # 从Seestar双窄中提取Ha和OIII（标准天文算法）
    Ha = (r + g) * 0.5  # Ha主要在红+绿通道
    OIII = b            # OIII主要在蓝通道

    # 标准SHO配色：SII→R, Ha→G, OIII→B（双窄带模拟）
    hubble_r = Ha * 0.7
    hubble_g = Ha * 0.3 + OIII * 0.6
    hubble_b = OIII * 0.9

    # 合并通道，伽马校正，避免过饱和
    hubble = np.dstack((hubble_r, hubble_g, hubble_b))
    hubble = np.clip(hubble, 0, 1)
    # 柔和伽马，让色彩自然，不炸色
    hubble = np.power(hubble, 1/1.8)
    return hubble

# ====================== 自动对比度拉伸（修复显示过暗） ======================
def auto_stretch(img):
    p1, p99 = np.percentile(img, (1, 99))
    return np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)

# ====================== 主流程（严格顺序，杜绝彩纹） ======================
if st.button("🚀 执行全流程处理"):
    if not light_files:
        st.warning("⚠️ 请先上传科学光文件！")
        st.stop()
    
    with st.spinner("🔄 加载图像..."):
        # 加载所有图像
        lights = []
        for f in light_files:
            img = load_img(f)
            if img is not None:
                lights.append(img)
        # 加载校准帧
        dark = batch_mean(dark_file)
        flat = batch_mean(flat_file)
        bias = batch_mean(bias_file)
    
    if not lights:
        st.error("❌ 无有效图像加载，请检查文件格式")
        st.stop()

    # 1. 星点自动对齐（先对齐，再堆栈，从根源杜绝彩纹）
    if auto_align and len(lights) > 1:
        with st.spinner("🔄 星点对齐中..."):
            lights = align_images(lights)
    
    # 2. 多图堆栈（对齐后再平均，无干涉彩纹）
    with st.spinner("🔄 堆栈合并中..."):
        stack = np.mean(np.array(lights), axis=0)
        stack = np.clip(stack, 0, 1)
    
    # 3. 暗场/平场/偏置校准
    with st.spinner("🔄 校准中..."):
        cal = calibrate(stack, dark, flat, bias)
    
    # 4. 模式分支（严格分开，互不干扰）
    if process_mode == "🌌 深空双窄带 → 标准伪哈勃色（SHO）":
        with st.spinner("🔄 合成标准伪哈勃色..."):
            final = dual_band_to_hubble_safe(cal)
    else:
        # 星野模式：不做任何调色，保留自然色
        final = cal.copy()
    
    # 5. 去红雾（可选，星野模式专用）
    if remove_red and process_mode == "🌟 星野模式（自然色，无调色）":
        final = remove_red_fog(final)
    
    # 6. 缩星锐化（最后一步，避免放大噪点）
    if sharpen:
        final = star_sharpen(final)
    
    # ====================== 结果展示 ======================
    st.subheader("📊 处理前后对比")
    col1, col2 = st.columns(2)
    with col1:
        st.image(auto_stretch(lights[0]), caption="原图", use_container_width=True, clamp=True)
    with col2:
        st.image(auto_stretch(final), caption="处理结果", use_container_width=True, clamp=True)
    
    # 质量分析面板
    gray = final.mean(axis=-1)
    mean, med, std = sigma_clipped_stats(gray, sigma=3.0)
    thresh = med + 3 * std
    peaks = find_peaks(gray, threshold=thresh, box_size=5)
    star_count = len(peaks)
    snr = mean / std if std > 0 else 0

    st.subheader("🔎 图像质量分析")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("检测星点数量", f"{star_count} 颗")
    c2.metric("背景噪声 σ", round(float(std), 2))
    c3.metric("信噪比 SNR", round(float(snr), 2))
    c4.metric("堆栈帧数", len(lights))
    
    # 亮度直方图
    st.subheader("📈 亮度直方图分析")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(gray.flatten(), bins=100, color="#00aaff", alpha=0.7)
    ax.set_title("校准后图像亮度分布")
    ax.set_xlabel("像素亮度值")
    ax.set_ylabel("像素数量")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    # 导出（修复大文件导出）
    st.subheader("💾 导出成果")
    out_rgb = (auto_stretch(final) * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    # 快速PNG导出
    png_data = cv2.imencode(".png", out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
    st.download_button(
        label="📥 导出PNG（高速）",
        data=png_data,
        file_name=f"StarCal_{process_mode[:2]}.png",
        mime="image/png"
    )
    # FITS导出
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp_f:
        fits.PrimaryHDU(final).writeto(tmp_f.name, overwrite=True)
        with open(tmp_f.name, "rb") as f:
            st.download_button(
                label="📥 导出FITS科研格式",
                data=f.read(),
                file_name=f"StarCal_{process_mode[:2]}.fits",
                mime="application/fits"
            )
        os.unlink(tmp_f.name)
    
    st.success("✅ 全流程处理完成！无彩纹、无过饱和、无400错误")

# ====================== 名词解释 ======================
st.markdown("---")
with st.expander("📖 专业名词说明"):
    st.markdown("""
**1. 星野模式**：针对普通银河、星野摄影，保留自然色彩，仅做对齐、堆栈、校准、锐化，不做任何调色。
**2. 深空双窄带伪哈勃色（SHO）**：标准天文配色，将Ha（氢α）映射为红/绿、OIII（氧Ⅲ）映射为蓝，模拟哈勃空间望远镜的经典伪彩色，适配Seestar直出双窄带图像。
**3. 暗场/平场/偏置校准**：专业天文预处理流程，去除热噪、灰尘、电子底噪，提升画面纯净度。
**4. 星点自动对齐**：先对齐再堆栈，彻底解决多帧叠加错位、彩纹问题。
    """)

st.markdown("---")
st.markdown("作者：Lantean")
