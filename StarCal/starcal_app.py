import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tempfile
import os
from astropy.io import fits

# ------------------- 页面配置 -------------------
st.set_page_config(
    page_title="StarCal 天文摄影智能校准工具",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- 主标题 -------------------
st.title("🔭 StarCal 天文摄影智能校准工具")
st.markdown("---")

# ------------------- 模式选择 -------------------
process_mode = st.radio(
    "处理模式",
    ["🌟 星野模式（自然色彩）", "🌌 深空双窄带 → 标准伪哈勃色（SHO）"],
    horizontal=True
)

# ------------------- 文件上传 -------------------
st.subheader("📤 上传图像文件")
col1, col2, col3, col4 = st.columns(4)
with col1:
    light_files = st.file_uploader(
        "科学光 Light",
        type=["png", "jpg", "jpeg", "fits"],
        accept_multiple_files=True
    )
with col2:
    dark_file = st.file_uploader(
        "暗场 Dark",
        type=["png", "jpg", "jpeg", "fits"],
        accept_multiple_files=True
    )
with col3:
    flat_file = st.file_uploader(
        "平场 Flat",
        type=["png", "jpg", "jpeg", "fits"],
        accept_multiple_files=True
    )
with col4:
    bias_file = st.file_uploader(
        "偏置 Bias",
        type=["png", "jpg", "jpeg", "fits"],
        accept_multiple_files=True
    )

# ------------------- 功能选项 -------------------
st.subheader("⚙️ 高级设置")
col_a, col_b, col_c = st.columns(3)
with col_a:
    auto_align = st.checkbox("✅ 星点自动对齐", value=True)
with col_b:
    sharpen = st.checkbox("✅ 缩星锐化", value=True)
with col_c:
    remove_red = st.checkbox("✅ 去除背景红雾", value=False)

# ------------------- 核心函数（纯 Python，无 OpenCV） -------------------
@st.cache_data
def load_img(file):
    try:
        if file.name.lower().endswith("fits"):
            with fits.open(file) as hdul:
                data = hdul[0].data
            if len(data.shape) == 2:
                img = np.stack([data] * 3, axis=-1).astype(np.float32)
            else:
                img = data.astype(np.float32)
            # 翻转图像
            img = img[::-1, :, :] if len(img.shape) == 3 else img[::-1, :]
            # 自动拉伸显示
            vmin, vmax = np.percentile(img, (1, 99.9))
            img = np.clip((img - vmin) / (vmax - vmin + 1e-6), 0, 1)
            return img
        else:
            img = Image.open(file).convert("RGB")
            return np.array(img).astype(np.float32) / 255.0
    except Exception as e:
        st.warning(f"读取文件 {file.name} 失败: {e}")
        return None

@st.cache_data
def batch_mean(files):
    if not files:
        return None
    imgs = []
    for f in files:
        img = load_img(f)
        if img is not None:
            imgs.append(img)
    return np.mean(np.array(imgs), axis=0) if imgs else None

@st.cache_data
def align_images(imgs):
    if len(imgs) < 2:
        return imgs
    st.info("星点对齐已禁用（在线环境优化），直接使用平均堆栈")
    return imgs

@st.cache_data
def calibrate(stack, dark, flat, bias):
    cal = stack.copy()
    if bias is not None:
        cal = np.clip(cal - bias, 0, 1)
    if dark is not None:
        dark_sub = dark - bias if bias is not None else dark
        cal = np.clip(cal - dark_sub, 0, 1)
    if flat is not None:
        flat_norm = flat - bias if bias is not None else flat
        flat_norm = flat_norm / (np.mean(flat_norm) + 1e-6)
        cal = np.clip(cal / flat_norm, 0, 1)
    return cal

@st.cache_data
def star_sharpen(img):
    # 纯 Python 锐化核
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # 手动卷积
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    output = np.zeros_like(img)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            for k in range(c):
                if len(img.shape) == 3:
                    patch = img[i-1:i+2, j-1:j+2, k]
                else:
                    patch = img[i-1:i+2, j-1:j+2]
                output[i, j, k] = np.sum(patch * kernel)
    
    return np.clip(output, 0, 1)

@st.cache_data
def remove_red_fog(img):
    r, g, b = np.split(img, 3, axis=-1)
    r = np.clip(r * (np.mean(g) / (np.mean(r) + 1e-6)), 0, 1)
    return np.concatenate((r, g, b), axis=-1)

@st.cache_data
def dual_band_to_hubble_safe(img):
    r, g, b = np.split(img, 3, axis=-1)

    def normalize(x):
        p1, p99 = np.percentile(x, (1, 99))
        return np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)

    ha = 0.5 * (normalize(r) + normalize(g))
    o3 = normalize(b)

    sho = np.dstack((ha * 0.7, ha * 0.3 + o3 * 0.6, o3 * 0.9))
    return np.clip(sho ** (1 / 1.8), 0, 1)

@st.cache_data
def auto_stretch(img):
    p1, p99 = np.percentile(img, (1, 99))
    return np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)

# ------------------- 主流程 -------------------
if st.button("🚀 执行全流程处理"):
    if not light_files:
        st.warning("请至少上传一组 Light 文件！")
        st.stop()

    # 加载数据
    with st.spinner("正在加载图像数据..."):
        lights = [load_img(f) for f in light_files if load_img(f) is not None]
        dark = batch_mean(dark_file)
        flat = batch_mean(flat_file)
        bias = batch_mean(bias_file)

    # 对齐
    if auto_align and len(lights) > 1:
        lights = align_images(lights)

    # 堆栈
    with st.spinner("正在进行多帧堆栈..."):
        stack = np.mean(np.array(lights), axis=0)

    # 校准
    with st.spinner("正在进行专业校准..."):
        cal = calibrate(stack, dark, flat, bias)

    # 模式处理
    with st.spinner("正在生成最终图像..."):
        if "伪哈勃" in process_mode:
            final = dual_band_to_hubble_safe(cal)
        else:
            final = cal

        # 后处理
        if remove_red and "星野" in process_mode:
            final = remove_red_fog(final)
        if sharpen:
            final = star_sharpen(final)

    # 展示结果
    st.subheader("📊 效果对比")
    c1, c2 = st.columns(2)
    with c1:
        st.image(auto_stretch(lights[0]), caption="原图", use_column_width=True)
    with c2:
        st.image(auto_stretch(final), caption="处理后", use_column_width=True)

    # 下载
    out_img = Image.fromarray((auto_stretch(final) * 255).astype(np.uint8))
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    out_img.save(buf, format="PNG")

    with open(buf.name, "rb") as f:
        st.download_button(
            "📥 下载处理结果",
            f.read(),
            file_name="StarCal_Result.png",
            mime="image/png"
        )
    os.unlink(buf.name)
    st.success("✅ 处理完成！感谢使用 StarCal！")

# ------------------- 页脚 -------------------
st.markdown("---")
st.caption("🔭 StarCal 天文摄影智能校准工具 ")
