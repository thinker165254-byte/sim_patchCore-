import os
import time
import json
import numpy as np
import torch
import faiss
import joblib
import argparse  # <-- 引入 argparse
from torch.nn import functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
from torchvision.models import Wide_ResNet50_2_Weights, ResNet101_Weights


# ======================== Dataset ========================
class PimpleDataset(Dataset):
    """痘痘数据集"""

    def __init__(self, root, transform, mode="train"):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        base_dir = os.path.join(root, mode)
        if mode == "train":
            for f in os.listdir(base_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(base_dir, f))
                    self.labels.append(0)
        else:
            for subdir in os.listdir(base_dir):
                subdir_path = os.path.join(base_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                label = 0 if subdir == "good" else 1
                for f in os.listdir(subdir_path):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(subdir_path, f))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx], img_path


# ======================== Config ========================
class Config:
    # --- 可调参数 ---
    batch_size = 16
    load_size = 256
    input_size = 224
    coreset_ratio = 0.04  # 核心集采样比例
    n_neighbors = 9  # kNN 邻居数量
    threshold_percentile = 20  # 阈值校准百分位数
    # ----------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== PatchCore ========================
class PatchCore:
    """简化版 PatchCore (仅二分类)"""

    def __init__(self, load_weights=False):
        # 使用 ResNet101
        self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).to(Config.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.features = []
        self.model.layer2[-1].register_forward_hook(self._hook)
        self.model.layer3[-1].register_forward_hook(self._hook)

        # transform
        self.transform = transforms.Compose([
            transforms.Resize((Config.load_size, Config.load_size)),
            transforms.CenterCrop(Config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.index = None
        self.threshold = 0.0
        self.coreset_data = None
        self.projector = None

    def _hook(self, module, input, output):
        self.features.append(output)

    def _embedding_concat(self, x, y):
        B, C1, H1, W1 = x.shape
        _, C2, H2, W2 = y.shape
        s = H1 // H2
        x = F.unfold(x, kernel_size=s, stride=s)
        x = x.view(B, C1, s * s, H2, W2)
        y = y.unsqueeze(2).repeat(1, 1, s * s, 1, 1)
        return torch.cat((x, y), dim=1).view(B, -1, H2, W2)

    def _extract_embeddings(self, dataloader):
        embs = []
        with torch.no_grad():
            for x, _, _ in dataloader:
                x = x.to(Config.device)
                self.features = []
                _ = self.model(x)
                fmap = [F.avg_pool2d(f, 3, 1, 1) for f in self.features]
                embedding = self._embedding_concat(fmap[0], fmap[1])
                B, C, H, W = embedding.shape
                embs.extend(embedding.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy())
        return np.array(embs)

    def train(self, train_loader):
        print("提取训练特征中...")
        total_embeddings = self._extract_embeddings(train_loader)
        print(f"特征数量: {total_embeddings.shape}")

        # 降维
        self.projector = SparseRandomProjection(n_components=550, random_state=42)
        reduced = self.projector.fit_transform(total_embeddings)

        # 核心集选择
        selector = kCenterGreedy(reduced, None, seed=42)
        n_coreset = int(len(reduced) * Config.coreset_ratio)
        selected_idx = selector.select_batch(model=None, already_selected=[], N=n_coreset)
        coreset = reduced[selected_idx]
        self.coreset_data = coreset
        print(f"核心集数量: {len(coreset)}")

        # 构建索引
        self.index = faiss.IndexFlatL2(coreset.shape[1])
        self.index.add(coreset)
        print("训练完成，Faiss 索引构建完毕。")

    def predict(self, x):
        """预测单张图像"""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.to(Config.device)

        with torch.no_grad():
            self.features = []
            _ = self.model(x)
            fmap = [F.avg_pool2d(f, 3, 1, 1) for f in self.features]
            embedding = self._embedding_concat(fmap[0], fmap[1])
            B, C, H, W = embedding.shape
            emb_flat = embedding.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

            emb_flat = self.projector.transform(emb_flat)
            scores, _ = self.index.search(emb_flat, k=Config.n_neighbors)
            patch_scores = np.mean(scores, axis=1)
            return float(np.max(patch_scores))

    def calibrate_threshold(self, val_loader):
        print("校准阈值中...")
        scores = [self.predict(x) for x, _, _ in val_loader]
        self.threshold = np.percentile(scores, Config.threshold_percentile)
        print(f"阈值设为 {self.threshold:.4f} (基于 {Config.threshold_percentile} 百分位数)")
        return self.threshold

    def evaluate(self, test_loader):
        print("评估模型准确率中...")
        y_true, y_pred = [], []
        for x, label, _ in test_loader:
            score = self.predict(x)
            pred = 1 if score > self.threshold else 0
            y_true.append(int(label))
            y_pred.append(pred)
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"测试准确率: {acc:.4f}")
        return acc

    def save_model(self, model_save_path):  # <-- 路径作为参数传入
        os.makedirs(model_save_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(model_save_path, "faiss_index.bin"))
        with open(os.path.join(model_save_path, "threshold.json"), "w") as f:
            json.dump({"threshold": float(self.threshold)}, f)
        joblib.dump(self.projector, os.path.join(model_save_path, "projector.pkl"))
        print(f"模型已保存到 {model_save_path}")

    @staticmethod
    def load_model(model_save_path):  # <-- 路径作为参数传入
        index = faiss.read_index(os.path.join(model_save_path, "faiss_index.bin"))
        with open(os.path.join(model_save_path, "threshold.json")) as f:
            threshold = json.load(f)["threshold"]
        projector = joblib.load(os.path.join(model_save_path, "projector.pkl"))
        model = PatchCore(load_weights=True)
        model.index = index
        model.threshold = threshold
        model.projector = projector
        print(f"模型从 {model_save_path} 加载成功")
        return model


# ======================== Train (Manual) ========================
def main_train(dataset_root, category):  # <-- 接收参数
    """手动运行一次 PatchCore 训练、校准和评估的流程"""

    # 使用类别名生成唯一的模型保存路径
    model_save_path = f"patchcore_model/patchcore_model_{category}"

    print(f"\n--- PatchCore 单次训练开始 (类别: {category}) ---")
    model = PatchCore()

    # 1. 准备数据加载器
    train_loader = DataLoader(PimpleDataset(dataset_root, model.transform, "train"),
                              batch_size=Config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(PimpleDataset(dataset_root, model.transform, "test"),
                            batch_size=1, shuffle=False)
    test_loader = DataLoader(PimpleDataset(dataset_root, model.transform, "test"),
                             batch_size=1, shuffle=False)

    # 2. 训练/构建核心集
    model.train(train_loader)

    # 3. 校准阈值
    model.calibrate_threshold(val_loader)

    # 4. 评估
    acc = model.evaluate(test_loader)

    # 5. 保存模型
    model.save_model(model_save_path)

    print(f"\n--- 流程结束，最终准确率: {acc:.4f} ---")


# ======================== Test ========================
def test_program(dataset_root, category):  # <-- 接收参数
    """加载模型并对测试集进行预测"""
    model_load_path = f"patchcore_model/patchcore_model_{category}"
    test_path = os.path.join(dataset_root, "test")

    print(f"\n--- 模型测试开始 (类别: {category}) ---")
    try:
        model = PatchCore.load_model(model_load_path)  # <-- 使用类别路径加载
    except Exception as e:
        print(f"无法加载模型 '{model_load_path}'。请先运行 main_train() 训练模型。错误: {e}")
        return

    normal, abnormal = 0, 0
    # 遍历 good/bad 目录
    for sub_category in os.listdir(test_path):
        sub_path = os.path.join(test_path, sub_category)
        if not os.path.isdir(sub_path):
            continue
        print(f"\n-> 正在测试子目录: {sub_category}")
        for img_file in os.listdir(sub_path):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(sub_path, img_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = model.transform(img)
            score = model.predict(img_tensor)
            pred = "异常 (Bad)" if score > model.threshold else "正常 (Good)"
            print(f"  [{img_file}] 分数={score:.4f} 预测={pred}")

            if sub_category == "good":
                if pred == "正常 (Good)":
                    normal += 1
                else:
                    abnormal += 1
            else:
                if pred == "异常 (Bad)":
                    abnormal += 1
                else:
                    normal += 1

    print("\n--- 统计结果 ---")
    print(f"预测正常样本总数: {normal}")
    print(f"预测异常样本总数: {abnormal}")


# ======================== Argument Parsing ========================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PatchCore 异常检测模型训练与测试。")
    parser.add_argument('mode', type=str, choices=['train', 'test'],default='train',
                        help="运行模式: 'train' (训练并保存模型) 或 'test' (加载并测试模型)。")
    parser.add_argument('--root_path', type=str,default='datasets_pro_500/pimple_patch/001',
                        help="数据集根目录 (例如: datasets_pro_500/pimple_patch/002)。")
    parser.add_argument('--category', type=str,default='001',
                        help="模型类别名称，用于命名保存的模型文件 (例如: 002)。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'train':
        # 命令行示例: python your_script.py train --root_path datasets_pro_500/pimple_patch/002 --category 002
        main_train(args.root_path, args.category)
    elif args.mode == 'test':
        # 命令行示例: python your_script.py test --root_path datasets_pro_500/pimple_patch/002 --category 002
        test_program(args.root_path, args.category)