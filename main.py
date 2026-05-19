from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PIL import Image


def add_watermark(
    logo_path: str,
    image_dir: str,
    logo_scale: float = 0.15,
    opacity: float = 0.85,
    position: str = "top-left",
    margin: int = 20,
) -> str:
    """
    将 logo 图片作为水印叠加到指定文件夹下的所有图片上，
    输出结果保存到同级新目录，目录名格式为 with_watermark_<timestamp>_<pic_count>。

    Parameters
    ----------
    logo_path : str
        水印 logo 图片路径（支持透明通道的 PNG 效果最佳）。
    image_dir : str
        待加水印的图片文件夹路径。
    logo_scale : float, optional
        logo 相对于目标图片短边的缩放比例，默认 0.15（即 15%）。
    opacity : float, optional
        水印不透明度，范围 0.0（完全透明）~ 1.0（完全不透明），默认 0.85。
    position : str, optional
        水印位置，可选 "bottom-right"（右下）、"bottom-left"（左下）、
        "top-right"（右上）、"top-left"（左上）、"center"（居中），默认 "top-left"。
    margin : int, optional
        水印距图片边缘的像素间距，默认 20。

    Returns
    -------
    str
        输出目录的完整路径。

    Examples
    --------
    >>> out_dir = add_watermark("logo.png", "/path/to/images")
    >>> out_dir = add_watermark("logo.png", "/path/to/images", logo_scale=0.1, opacity=0.7)
    """
    # ── 1. 校验参数 ────────────────────────────────────────────────────────
    logo_path = str(logo_path)
    image_dir = str(image_dir)

    if not os.path.isfile(logo_path):
        raise FileNotFoundError(f"Logo 文件不存在: {logo_path}")
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"图片目录不存在: {image_dir}")

    # ── 2. 收集目标目录中所有图片文件 ─────────────────────────────────────
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    image_files = sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    )

    if not image_files:
        raise ValueError(f"目录 {image_dir} 中未找到支持的图片文件")

    pic_count = len(image_files)

    # ── 3. 创建输出目录 ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir_name = f"with_watermark_{timestamp}_{pic_count}"
    output_dir = os.path.join(image_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[✓] 输出目录已创建: {output_dir}")

    # ── 4. 加载 logo（转为 RGBA，应用透明度）─────────────────────────────
    logo_rgba = Image.open(logo_path).convert("RGBA")

    # ── 5. 逐张处理图片 ────────────────────────────────────────────────────
    for idx, img_path in enumerate(image_files, start=1):
        try:
            target = Image.open(img_path).convert("RGBA")
            t_w, t_h = target.size

            # 按目标图片短边等比例缩放 logo
            short_side = min(t_w, t_h)
            logo_size = int(short_side * logo_scale)
            # 保持 logo 宽高比
            lw, lh = logo_rgba.size
            ratio = logo_size / min(lw, lh)
            new_lw = max(1, int(lw * ratio))
            new_lh = max(1, int(lh * ratio))
            logo_resized = logo_rgba.resize((new_lw, new_lh), Image.LANCZOS)

            # 应用不透明度到 logo 的 Alpha 通道
            if opacity < 1.0:
                import numpy as np
                logo_arr = np.array(logo_resized, dtype=np.float32)
                logo_arr[:, :, 3] = logo_arr[:, :, 3] * opacity
                logo_resized = Image.fromarray(logo_arr.astype(np.uint8), mode="RGBA")

            # 计算水印贴合位置
            pos_map = {
                "bottom-right": (t_w - new_lw - margin, t_h - new_lh - margin),
                "bottom-left":  (margin, t_h - new_lh - margin),
                "top-right":    (t_w - new_lw - margin, margin),
                "top-left":     (margin, margin),
                "center":       ((t_w - new_lw) // 2, (t_h - new_lh) // 2),
            }
            paste_x, paste_y = pos_map.get(position, pos_map["top-right"])

            # 合成水印
            composite = target.copy()
            composite.paste(logo_resized, (paste_x, paste_y), mask=logo_resized)

            # 输出文件：jpg 转为 RGB 再保存，png 保留透明通道
            out_filename = img_path.stem + img_path.suffix.lower()
            out_path = os.path.join(output_dir, out_filename)

            if img_path.suffix.lower() in {".jpg", ".jpeg"}:
                composite.convert("RGB").save(out_path, quality=95)
            else:
                composite.save(out_path)

            print(f"  [{idx}/{pic_count}] 已处理: {img_path.name} → {out_filename}")

        except Exception as exc:
            print(f"  [!] 跳过 {img_path.name}，原因: {exc}")

    print(f"\n[✓] 全部完成，共处理 {pic_count} 张图片，结果保存至: {output_dir}")
    return output_dir


# ── 示例调用 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logo = "./origin_no_watermark.png"
    folder = "./pics"
    add_watermark(logo, folder)
