from __future__ import annotations

import os
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

from PIL import Image


# ── 核心加水印逻辑（从 main.py 内联，避免打包时路径问题） ──────────────────
def add_watermark(
    logo_path: str,
    image_dir: str,
    logo_scale: float = 0.15,
    opacity: float = 0.85,
    position: str = "top-left",
    margin: int = 20,
    progress_callback=None,
) -> str:
    logo_path = str(logo_path)
    image_dir = str(image_dir)

    if not os.path.isfile(logo_path):
        raise FileNotFoundError(f"水印图片不存在: {logo_path}")
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"图片目录不存在: {image_dir}")

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    image_files = sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    )

    if not image_files:
        raise ValueError(f"目录 {image_dir} 中未找到支持的图片文件")

    pic_count = len(image_files)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir_name = f"with_watermark_{timestamp}_{pic_count}"
    output_dir = os.path.join(image_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    logo_rgba = Image.open(logo_path).convert("RGBA")

    for idx, img_path in enumerate(image_files, start=1):
        try:
            target = Image.open(img_path).convert("RGBA")
            t_w, t_h = target.size

            short_side = min(t_w, t_h)
            logo_size = int(short_side * logo_scale)
            lw, lh = logo_rgba.size
            ratio = logo_size / min(lw, lh)
            new_lw = max(1, int(lw * ratio))
            new_lh = max(1, int(lh * ratio))
            logo_resized = logo_rgba.resize((new_lw, new_lh), Image.LANCZOS)

            if opacity < 1.0:
                import numpy as np
                logo_arr = np.array(logo_resized, dtype=np.float32)
                logo_arr[:, :, 3] = logo_arr[:, :, 3] * opacity
                logo_resized = Image.fromarray(logo_arr.astype(np.uint8), mode="RGBA")

            pos_map = {
                "bottom-right": (t_w - new_lw - margin, t_h - new_lh - margin),
                "bottom-left":  (margin, t_h - new_lh - margin),
                "top-right":    (t_w - new_lw - margin, margin),
                "top-left":     (margin, margin),
                "center":       ((t_w - new_lw) // 2, (t_h - new_lh) // 2),
            }
            paste_x, paste_y = pos_map.get(position, pos_map["top-left"])

            composite = target.copy()
            composite.paste(logo_resized, (paste_x, paste_y), mask=logo_resized)

            out_filename = img_path.stem + img_path.suffix.lower()
            out_path = os.path.join(output_dir, out_filename)

            if img_path.suffix.lower() in {".jpg", ".jpeg"}:
                composite.convert("RGB").save(out_path, quality=95)
            else:
                composite.save(out_path)

            if progress_callback:
                progress_callback(idx, pic_count, img_path.name)

        except Exception as exc:
            if progress_callback:
                progress_callback(idx, pic_count, f"[跳过] {img_path.name}: {exc}")

    return output_dir


# ── GUI 主程序 ─────────────────────────────────────────────────────────────
class WatermarkApp(tk.Tk):
    # 位置选项：显示文字 → 内部 key
    POSITION_OPTIONS = [
        ("左上角", "top-left"),
        ("右上角", "top-right"),
        ("左下角", "bottom-left"),
        ("右下角", "bottom-right"),
        ("居中",   "center"),
    ]

    def __init__(self):
        super().__init__()
        self.title("批量加水印工具")
        self.resizable(False, False)
        self._build_ui()
        self._center_window(500, 340)

    def _center_window(self, w: int, h: int):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        pad = {"padx": 16, "pady": 6}

        # ── 标题 ──
        title_lbl = tk.Label(
            self,
            text="批量加水印工具",
            font=("微软雅黑", 16, "bold"),
            fg="#2c3e50",
        )
        title_lbl.pack(pady=(18, 4))

        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill="x", padx=16, pady=(0, 8))

        # ── 表单区 ──
        form = tk.Frame(self)
        form.pack(fill="x", **pad)
        form.columnconfigure(1, weight=1)

        # 1. 水印图片
        tk.Label(form, text="水印图片：", anchor="w", width=10).grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.logo_var = tk.StringVar()
        logo_entry = tk.Entry(form, textvariable=self.logo_var, state="readonly", width=36)
        logo_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        tk.Button(form, text="选择…", width=6, command=self._pick_logo).grid(
            row=0, column=2
        )

        # 2. 图片目录
        tk.Label(form, text="图片目录：", anchor="w", width=10).grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.dir_var = tk.StringVar()
        dir_entry = tk.Entry(form, textvariable=self.dir_var, state="readonly", width=36)
        dir_entry.grid(row=1, column=1, sticky="ew", padx=(0, 6))
        tk.Button(form, text="选择…", width=6, command=self._pick_dir).grid(
            row=1, column=2
        )

        # 3. 水印位置
        tk.Label(form, text="水印位置：", anchor="w", width=10).grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.pos_var = tk.StringVar(value="左上角")
        pos_labels = [label for label, _ in self.POSITION_OPTIONS]
        pos_combo = ttk.Combobox(
            form,
            textvariable=self.pos_var,
            values=pos_labels,
            state="readonly",
            width=12,
        )
        pos_combo.grid(row=2, column=1, sticky="w")

        sep2 = ttk.Separator(self, orient="horizontal")
        sep2.pack(fill="x", padx=16, pady=(8, 0))

        # ── 加水印按钮 ──
        self.btn = tk.Button(
            self,
            text="开始加水印",
            font=("微软雅黑", 11, "bold"),
            bg="#2980b9",
            fg="white",
            activebackground="#1a6fa3",
            activeforeground="white",
            relief="flat",
            padx=24,
            pady=6,
            cursor="hand2",
            command=self._start,
        )
        self.btn.pack(pady=(14, 6))

        # ── 进度条 ──
        self.progress = ttk.Progressbar(self, length=440, mode="determinate")
        self.progress.pack(padx=16, pady=(0, 6))

        # ── 状态文字 ──
        self.status_var = tk.StringVar(value="请选择水印图片和图片目录后点击「开始加水印」")
        status_lbl = tk.Label(
            self,
            textvariable=self.status_var,
            fg="#555555",
            font=("微软雅黑", 9),
            wraplength=460,
            justify="center",
        )
        status_lbl.pack(pady=(0, 12))

    # ── 事件处理 ──────────────────────────────────────────────────────────

    def _pick_logo(self):
        path = filedialog.askopenfilename(
            title="选择水印图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.logo_var.set(path)

    def _pick_dir(self):
        path = filedialog.askdirectory(title="选择需要加水印的图片目录")
        if path:
            self.dir_var.set(path)

    def _get_position_key(self) -> str:
        label = self.pos_var.get()
        for lbl, key in self.POSITION_OPTIONS:
            if lbl == label:
                return key
        return "top-left"

    def _start(self):
        logo = self.logo_var.get().strip()
        image_dir = self.dir_var.get().strip()

        if not logo:
            messagebox.showwarning("提示", "请先选择水印图片！")
            return
        if not image_dir:
            messagebox.showwarning("提示", "请先选择图片目录！")
            return

        self.btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("处理中，请稍候…")

        position = self._get_position_key()

        def worker():
            try:
                def on_progress(idx, total, name):
                    pct = int(idx / total * 100)
                    self.progress["value"] = pct
                    self.status_var.set(f"[{idx}/{total}] 正在处理：{name}")
                    self.update_idletasks()

                out_dir = add_watermark(
                    logo_path=logo,
                    image_dir=image_dir,
                    position=position,
                    progress_callback=on_progress,
                )
                self.progress["value"] = 100
                self.status_var.set(f"✅ 全部完成！结果已保存至：\n{out_dir}")
                messagebox.showinfo("完成", f"水印添加完成！\n\n结果保存在：\n{out_dir}")
            except Exception as e:
                self.status_var.set(f"❌ 出错：{e}")
                messagebox.showerror("错误", str(e))
            finally:
                self.btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = WatermarkApp()
    app.mainloop()

