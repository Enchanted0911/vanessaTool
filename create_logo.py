"""
create_logo.py

将含有猫猫头轮廓的图片中，轮廓外的深色背景替换为透明，
猫猫头轮廓内的像素保持原样不变。

思路：
1. 将图片转为 RGBA 模式。
2. 把图片转为灰度图，对灰度图做二值化（阈值分割），
   把"深色背景"区域识别出来，其余区域视为前景（猫猫头）。
3. 利用 cv2.floodFill 从四个角出发，将连通的深色背景区域
   标记为"外部背景"，而猫猫头内部即使有暗色也不会被误删。
4. 将外部背景对应的像素 Alpha 通道设为 0（透明），其余像素不变。
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def remove_dark_background(
    input_path: str,
    output_path: str,
    dark_threshold: int = 80,
    blur_ksize: int = 3,
) -> Image.Image:
    """
    去除猫猫头轮廓外的深色背景，改为透明。

    Parameters
    ----------
    input_path : str
        输入图片路径。
    output_path : str
        输出图片路径（建议保存为 .png 以支持透明通道）。
    dark_threshold : int, optional
        灰度阈值，低于该值的像素视为"深色"，默认 80。
        可根据实际图片亮度适当调整（0-255）。
    blur_ksize : int, optional
        在二值化前对灰度图做轻微高斯模糊的核大小，
        用于减少噪点，默认 3（奇数）。

    Returns
    -------
    PIL.Image.Image
        处理后的 RGBA 图片对象（同时已保存到 output_path）。
    """
    # ── 1. 读入并转为 RGBA ─────────────────────────────────────────────
    pil_img = Image.open(input_path).convert("RGBA")
    img_rgba = np.array(pil_img, dtype=np.uint8)  # H×W×4

    # ── 2. 灰度化 + 轻微模糊，减少噪点 ────────────────────────────────
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 1:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # ── 3. 二值化：深色区域 → 0（黑），亮色区域 → 255（白）─────────────
    # 猫猫头轮廓颜色偏白/浅，背景偏深
    _, binary = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY)

    # ── 4. 形态学操作：闭运算填补轮廓内的小空洞，使轮廓更完整 ───────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ── 5. 从四个角做 floodFill，标记"外部背景" ──────────────────────
    #  floodFill 要求画布比图像大 2px（各方向 +1）
    h, w = binary_closed.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # 用 255 填充外部连通的暗色区域；先取反让"暗色=255"，再 flood 从角出发
    # 实际做法：直接对 binary_closed 取反后做 floodFill
    inv_binary = cv2.bitwise_not(binary_closed)  # 暗色→255, 亮色→0

    # 从四个角各做一次 floodFill
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for (r, c) in corners:
        if inv_binary[r, c] == 255:  # 该角是暗色才需要 flood
            cv2.floodFill(inv_binary, flood_mask, (c, r), 128)

    # inv_binary 中值为 128 的像素 = 外部背景（连通的暗色区域）
    external_bg_mask = (inv_binary == 128)

    # ── 6. 将外部背景的 Alpha 设为 0，其余像素完全不变 ─────────────────
    result = img_rgba.copy()
    result[external_bg_mask, 3] = 0  # 仅改 Alpha，RGB 保持原值

    # ── 7. 保存并返回 ──────────────────────────────────────────────────
    out_img = Image.fromarray(result, mode="RGBA")
    out_img.save(output_path)
    print(f"[✓] 已保存透明背景图片至: {output_path}")
    return out_img


# ── 示例调用 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        src = sys.argv[1]
        dst = sys.argv[2]
        threshold = int(sys.argv[3]) if len(sys.argv) >= 4 else 80
    else:
        # 默认示例路径，按需修改
        src = "origin.jpg"
        dst = "cat_logo_output.png"
        threshold = 80

    remove_dark_background(src, dst, dark_threshold=threshold)

