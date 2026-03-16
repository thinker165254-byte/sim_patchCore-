import os
from PIL import Image
import glob


def resize_keep_aspect_ratio(input_folder, output_folder, target_long_edge=640):
    """
    仅缩放：保持原图比例缩放，不填充背景。
    适合 PatchCore 等支持任意尺寸输入的算法。

    参数:
        input_folder: 输入文件夹
        output_folder: 输出文件夹
        target_long_edge: 长边的目标尺寸 (例如 640, 512 等)
    """
    os.makedirs(output_folder, exist_ok=True)

    # 支持的扩展名
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
        files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    processed_count = 0
    print(f"开始处理: {input_folder}")
    print(f"目标: 长边缩放到 {target_long_edge}，保持长宽比")

    for image_path in files:
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                width, height = img.size

                # --- 计算缩放后的尺寸 ---
                # 确定缩放比例：根据长边来算
                if width >= height:
                    scale = target_long_edge / width
                    new_w = target_long_edge
                    new_h = int(height * scale)
                else:
                    scale = target_long_edge / height
                    new_h = target_long_edge
                    new_w = int(width * scale)

                # --- 进阶技巧：确保尺寸是 8 或 16 的倍数 (可选，但推荐) ---
                # 许多深度学习网络喜欢尺寸能被 8, 16, 32 整除
                # 这里强制高度调整为 16 的倍数，防止奇怪的报错
                align_to = 16
                new_w = round(new_w / align_to) * align_to
                new_h = round(new_h / align_to) * align_to

                # 如果算出来是0 (图太小)，保底设为16
                new_w = max(new_w, 16)
                new_h = max(new_h, 16)

                # --- 缩放 ---
                resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # --- 保存 ---
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                save_path = os.path.join(output_folder, f"{name}.jpg")
                resized_img.save(save_path, quality=95)

                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"已处理: {filename} -> {new_w}x{new_h}")

        except Exception as e:
            print(f"处理失败 {image_path}: {e}")

    print(f"\n全部完成！共 {processed_count} 张。")
    print(f"输出目录: {output_folder}")


if __name__ == "__main__":
    input_dir = "datasets/pimple_patch/002/test/lack"
    output_dir = "datasets/pimple_patch/002/test/lack"

    # 设置长边大小
    # 你的原图 2560x1440 (16:9)
    # 如果设为 512，则结果约为 512 x 288 (推荐)
    # 如果设为 640，则结果约为 640 x 360 (常用)
    # 如果设为 1024，则结果约为 1024 x 576 (高精度)
    target_long_edge = 640

    resize_keep_aspect_ratio(input_dir, output_dir, target_long_edge)