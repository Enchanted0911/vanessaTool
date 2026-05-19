from __future__ import annotations

import numpy as np
from PIL import Image


def remove_watermark_bottom_right(
    input_path: str,
    output_path: str,
    region_ratio: float = 0.1,
) -> Image.Image:
    """
    去除图片右下角的水印，将该区域像素的 Alpha 通道设为 0（透明）。

    Parameters
    ----------
    input_path : str
        输入图片路径。
    output_path : str
        输出图片路径（建议保存为 .png 以支持透明通道）。
    region_ratio : float, optional
        右下角水印区域占图片宽/高的比例，默认 0.1（即十分之一）。
        例如 0.1 表示右下角宽度占总宽的 10%、高度占总高的 10% 的矩形区域。

    Returns
    -------
    PIL.Image.Image
        处理后的 RGBA 图片对象（同时已保存到 output_path）。

    Examples
    --------
    >>> result = remove_watermark_bottom_right("input.jpg", "output.png")
    >>> result = remove_watermark_bottom_right("input.png", "output.png", region_ratio=0.15)
    """
    # ── 1. 读入并转为 RGBA ─────────────────────────────────────────────
    pil_img = Image.open(input_path).convert("RGBA")
    img_array = np.array(pil_img, dtype=np.uint8)  # H×W×4

    h, w = img_array.shape[:2]

    # ── 2. 计算右下角区域的起始坐标 ───────────────────────────────────
    # 右下角区域：宽度 = w * region_ratio，高度 = h * region_ratio
    region_w = int(w * region_ratio)
    region_h = int(h * region_ratio)

    x_start = w - region_w  # 水印区域左边界（列起始）
    y_start = h - region_h  # 水印区域上边界（行起始）

    # ── 3. 将该区域的 Alpha 通道设为 0（完全透明）─────────────────────
    result = img_array.copy()
    result[y_start:h, x_start:w, 3] = 0  # 仅修改 Alpha 通道

    # ── 4. 保存并返回 ──────────────────────────────────────────────────
    out_img = Image.fromarray(result, mode="RGBA")
    out_img.save(output_path)
    print(
        f"[✓] 已去除右下角水印区域（{region_w}×{region_h} px），"
        f"结果已保存至: {output_path}"
    )
    return out_img


# ── 示例调用 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        src = sys.argv[1]
        dst = sys.argv[2]
        ratio = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.1
    else:
        # 默认示例路径，按需修改
        src = "cat_logo_output.png"
        dst = "origin_no_watermark.png"
        ratio = 0.1

    remove_watermark_bottom_right(src, dst, region_ratio=ratio)

