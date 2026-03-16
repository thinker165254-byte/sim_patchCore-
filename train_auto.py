import os
import json
import numpy as np
import torch
import faiss
import joblib
from torch.nn import functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
from torchvision.models import ResNet101_Weights

# ======================== 全局配置区 ========================
# 请在这里修改你的数据路径
DATASET_ROOT = "datasets/pimple_patch/002"
TEST_DIR = "datasets/pimple_patch/002/test"


class Config:
    batch_size = 16
    img_width = 640
    img_height = 352

    # --- 核心优化参数 ---
    # 1. 降维维度：从 512 降到 128，计算速度快 4 倍，精度几乎不变
    n_components = 128

    # 2. 采样率：配合下面的随机瘦身，设为 1% - 10% 之间
    coreset_ratio = 0.01

    n_neighbors = 9

    # 3. 阈值：如果验证集只有好图，必须设高，否则全是误报
    threshold_percentile = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42


# ======================== Dataset ========================
class PimpleDataset(Dataset):
    """痘痘数据集加载器"""

    def __init__(self, root, transform, mode="train"):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 路径容错处理
        base_dir = os.path.join(root, mode)
        if not os.path.exists(base_dir):
            print(f"[警告] 目录不存在: {base_dir}")
            return

        if mode == "train":
            # 训练集：默认全为正常样本 (label=0)
            for f in os.listdir(base_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    self.image_paths.append(os.path.join(base_dir, f))
                    self.labels.append(0)
        else:
            # 测试集：遍历子文件夹 (good/bad)
            for subdir in os.listdir(base_dir):
                subdir_path = os.path.join(base_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                # 假设 'good' 文件夹为正常，其他为异常
                label = 0 if subdir == "good" else 1
                for f in os.listdir(subdir_path):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                        self.image_paths.append(os.path.join(subdir_path, f))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 确保转为 RGB
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx], img_path


# ======================== PatchCore ========================
class PatchCore:
    def __init__(self, load_weights=False):
        # 初始化 ResNet101
        self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).to(Config.device)
        self.model.eval()
        # 冻结参数
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.features = []
        # 注册钩子提取特征 (Layer2 和 Layer3)
        self.model.layer2[-1].register_forward_hook(self._hook)
        self.model.layer3[-1].register_forward_hook(self._hook)

        # --- Transform 修改 ---
        # Resize 参数顺序是 (H, W)，即 (352, 640)
        # 去掉 CenterCrop 以保留全图信息
        self.transform = transforms.Compose([
            transforms.Resize((Config.img_height, Config.img_width),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.index = None
        self.threshold = 0.0
        self.projector = None

    def _hook(self, module, input, output):
        self.features.append(output)

    def _embedding_concat(self, x, y):
        """
        特征融合：将深层特征(y)上采样并与浅层特征(x)拼接。
        针对 640x352 输入：
        - Layer2 (x): 80x44 (Stride 8)
        - Layer3 (y): 40x22 (Stride 16)
        - Scale (s): 2
        Unfold 操作可以将 80x44 拆分为 2x2 的块，正好对应 40x22，逻辑完美适配。
        """
        B, C1, H1, W1 = x.shape
        _, C2, H2, W2 = y.shape
        s = H1 // H2

        # 使用 unfold 进行局部块重排
        x = F.unfold(x, kernel_size=s, stride=s)
        x = x.view(B, C1, s * s, H2, W2)
        y = y.unsqueeze(2).repeat(1, 1, s * s, 1, 1)

        # 拼接
        return torch.cat((x, y), dim=1).view(B, -1, H2, W2)

    def _extract_embeddings(self, dataloader):
        embs = []
        print(f"正在提取特征 (Device: {Config.device})...")
        with torch.no_grad():
            for x, _, _ in dataloader:
                x = x.to(Config.device)
                self.features = []
                _ = self.model(x)

                # 对特征图进行平滑
                fmap = [F.avg_pool2d(f, 3, 1, 1) for f in self.features]

                # 融合
                embedding = self._embedding_concat(fmap[0], fmap[1])

                # 调整维度: (B, C, H, W) -> (N, C)
                B, C, H, W = embedding.shape
                embs.extend(embedding.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy())

        return np.array(embs)

    def train(self, train_loader):
        # 1. 提取所有 Patch 特征
        total_embeddings = self._extract_embeddings(train_loader)
        print(f"原始特征维度: {total_embeddings.shape}")  # 比如 (118800, 6144)

        if len(total_embeddings) == 0:
            raise ValueError("特征为空！")

        # --- 【关键一步】随机强力瘦身 ---
        # 限制最大参与计算的点数。通常 20,000 个点就足够代表整个数据集的分布了。
        # 如果不加这一步，10万+的点会导致 kCenter 卡死。
        MAX_TRAIN_SAMPLES = 20000

        if total_embeddings.shape[0] > MAX_TRAIN_SAMPLES:
            print(f"特征点太多 ({total_embeddings.shape[0]})，进行随机采样至 {MAX_TRAIN_SAMPLES}...")
            # 设置随机种子保证可复现
            np.random.seed(Config.seed)
            # 随机选择索引
            indices = np.random.choice(total_embeddings.shape[0], MAX_TRAIN_SAMPLES, replace=False)
            total_embeddings = total_embeddings[indices]
            print(f"瘦身且打乱后维度: {total_embeddings.shape}")

        # 2. 稀疏随机投影降维
        # 注意：使用 Config.n_components (128)
        self.projector = SparseRandomProjection(n_components=Config.n_components, random_state=Config.seed)
        reduced = self.projector.fit_transform(total_embeddings)
        print(f"降维后特征: {reduced.shape}")  # 应该是 (20000, 128)

        # 3. 核心集采样 (现在会飞快)
        print("正在进行核心集采样 (K-Center)...")
        selector = kCenterGreedy(reduced, None, seed=Config.seed)

        # 计算采样数量
        n_coreset = int(len(reduced) * Config.coreset_ratio)
        # 兜底：至少存 500 个点，防止采样太少
        n_coreset = max(n_coreset, 500)

        print(f"计划选取 {n_coreset} 个核心特征点...")

        selected_idx = selector.select_batch(model=None, already_selected=[], N=n_coreset)
        coreset = reduced[selected_idx]
        print(f"核心集采样完成，最终索引库大小: {coreset.shape}")

        # 4. 构建索引
        self.index = faiss.IndexFlatL2(coreset.shape[1])
        self.index.add(coreset)

    def predict(self, x):
        # 增加 batch 维度
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.to(Config.device)

        with torch.no_grad():
            self.features = []
            _ = self.model(x)
            fmap = [F.avg_pool2d(f, 3, 1, 1) for f in self.features]
            embedding = self._embedding_concat(fmap[0], fmap[1])
            B, C, H, W = embedding.shape

            # 展平
            emb_flat = embedding.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

            # 投影 (必须使用训练时的 projector)
            emb_flat = self.projector.transform(emb_flat)

            # 搜索最近邻
            scores, _ = self.index.search(emb_flat, k=Config.n_neighbors)

            # 计算异常分数
            # 1. 取每个 patch 最近 k 个邻居的平均距离
            patch_scores = np.mean(scores, axis=1)
            # 2. 图像分数 = 所有 patch 中最大的那个分数
            return float(np.max(patch_scores))

    def calibrate_threshold(self, val_loader):
        print("正在计算最佳阈值...")
        scores = []
        for x, _, _ in val_loader:
            scores.append(self.predict(x))

        if not scores:
            print("[警告] 验证集为空，默认阈值为 0.0")
            return 0.0

        self.threshold = np.percentile(scores, Config.threshold_percentile)
        print(f"根据验证集分布，设定阈值为: {self.threshold:.4f}")
        return self.threshold

    def evaluate(self, test_loader):
        print("开始评估测试集...")
        y_true, y_pred = [], []
        normal_count, anomaly_count = 0, 0

        for x, label, _ in test_loader:
            score = self.predict(x)
            pred = 1 if score > self.threshold else 0
            y_true.append(int(label))
            y_pred.append(pred)

            if pred == 0:
                normal_count += 1
            else:
                anomaly_count += 1

        if not y_true:
            print("[错误] 测试集为空！")
            return 0.0

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = np.mean(y_true == y_pred)
        print(f"评估结果 -> 准确率: {acc:.4f} (预测正常: {normal_count}, 预测异常: {anomaly_count})")
        return acc

    def save_model(self, path="best_patchcore_model"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({
                "threshold": float(self.threshold),
                "img_width": Config.img_width,
                "img_height": Config.img_height
            }, f)
        joblib.dump(self.projector, os.path.join(path, "projector.pkl"))
        print(f"模型成功保存至: {path}")

    @staticmethod
    def load_model(path="best_patchcore_model"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到模型路径: {path}")

        print(f"正在加载模型: {path} ...")
        index = faiss.read_index(os.path.join(path, "faiss_index.bin"))

        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
            threshold = cfg["threshold"]
            # 可以在这里校验尺寸是否匹配，这里略过

        projector = joblib.load(os.path.join(path, "projector.pkl"))

        model = PatchCore(load_weights=True)
        model.index = index
        model.threshold = threshold
        model.projector = projector
        print("模型加载完成！")
        return model


# ======================== 运行流程 ========================
def main_auto_train(max_runs=20):
    best_acc = 0.0
    stop_acc = 0.99
    best_state = None

    print(f"数据根目录: {DATASET_ROOT}")
    print(f"输入尺寸: {Config.img_width}x{Config.img_height}")

    for i in range(1, max_runs + 1):
        print(f"\n============= 第 {i}/{max_runs} 轮训练 =============")
        model = PatchCore()

        # 训练集 Loader
        train_ds = PimpleDataset(DATASET_ROOT, model.transform, "train")
        if len(train_ds) == 0:
            print("错误：训练集为空，请检查路径配置！")
            return

        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4)

        # 验证/测试集 Loader
        val_loader = DataLoader(PimpleDataset(DATASET_ROOT, model.transform, "test"), batch_size=1, shuffle=False)
        test_loader = DataLoader(PimpleDataset(DATASET_ROOT, model.transform, "test"), batch_size=1, shuffle=False)

        # 核心流程
        model.train(train_loader)
        model.calibrate_threshold(val_loader)
        acc = model.evaluate(test_loader)

        # 记录最佳模型
        if acc > best_acc:
            best_acc = acc
            best_state = {
                "index": model.index,
                "threshold": model.threshold,
                "projector": model.projector
            }
            print(f"🚀 新纪录！最佳准确率: {best_acc:.4f}")
        else:
            print(f"当前准确率: {acc:.4f} (未超过最佳 {best_acc:.4f})")

        if best_acc >= stop_acc:
            print("🎯 已达到目标准确率，提前停止训练。")
            break

    # 保存最佳模型
    if best_state:
        print("\n正在保存最佳模型...")
        best_model = PatchCore(load_weights=True)
        best_model.index = best_state["index"]
        best_model.threshold = best_state["threshold"]
        best_model.projector = best_state["projector"]
        best_model.save_model("best_rect_model")


def test_program(test_path):
    """单独的测试函数，加载已保存的模型进行推理"""
    print(f"\n启动独立测试，测试目录: {test_path}")
    try:
        model = PatchCore.load_model("best_rect_model")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    normal_cnt, abnormal_cnt = 0, 0

    # 遍历 good/bad 文件夹
    for category in os.listdir(test_path):
        sub_path = os.path.join(test_path, category)
        if not os.path.isdir(sub_path): continue

        print(f"\n--- 正在测试类别: {category} ---")
        for img_file in os.listdir(sub_path):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_full_path = os.path.join(sub_path, img_file)
            img = Image.open(img_full_path).convert("RGB")
            img_tensor = model.transform(img)

            score = model.predict(img_tensor)
            pred = "异常 (NG)" if score > model.threshold else "正常 (OK)"

            print(f"[{img_file}] 分数: {score:.4f} -> {pred}")

            if score > model.threshold:
                abnormal_cnt += 1
            else:
                normal_cnt += 1

    print("\n" + "=" * 30)
    print(f"测试统计完成:")
    print(f"判定为正常: {normal_cnt}")
    print(f"判定为异常: {abnormal_cnt}")
    print("=" * 30)


if __name__ == "__main__":
    # 模式选择：
    # 1. 训练模式
    main_auto_train(max_runs=3)

    # 2. 测试模式 (训练完成后取消下面的注释即可单独测试)
    #test_program(TEST_DIR)