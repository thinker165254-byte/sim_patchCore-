import os

# 设置目标文件夹路径
folder_path = r"datasets/pimple_patch/001/test/spot"

# 获取文件夹内所有 jpg 文件并排序
files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
files.sort()

# 逐个重命名
for index, filename in enumerate(files):
    new_name = f"{index:03d}.jpg"  # 生成 000, 001, 002 ...
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)

print("重命名完成！")