import os
from PIL import Image

# ===== 配置 =====
input_folder = r"C:/Users/29380/Downloads/first_pics/ai_modified_20260521_195703_44"   # 原图片文件夹
scale = 0.99                 # 缩放比例（默认99%）

# 输出文件夹
output_folder = os.path.join(input_folder, "resized")
os.makedirs(output_folder, exist_ok=True)

# 支持的图片格式
supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ===== 开始处理 =====
for filename in os.listdir(input_folder):

    if filename.lower().endswith(supported_formats):

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(input_path) as img:

                # 原尺寸
                width, height = img.size

                # 新尺寸
                new_width = int(width * scale)
                new_height = int(height * scale)

                # 缩放图片
                resized_img = img.resize(
                    (new_width, new_height),
                    Image.LANCZOS
                )

                # 保存
                resized_img.save(output_path)

                print(f"已处理: {filename} -> {new_width}x{new_height}")

        except Exception as e:
            print(f"处理失败 {filename}: {e}")

print("全部完成！")