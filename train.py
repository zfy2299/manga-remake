import os
import random
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# 导入单独的模型类
from ResNetUNet import ResNetUNet

# ====================== 1. 全局配置（核心可配置参数） ======================
# 基础路径配置（已修正为dataset）
ROOT_DIR = "dataset"  # 修正：从met_dataset改为dataset
DATA_DIR = os.path.join(ROOT_DIR, "data")  # 原始数据目录
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
VAL_DIR = os.path.join(ROOT_DIR, "val")
TEST_DIR = os.path.join(ROOT_DIR, "test")
TEST_OUT_DIR = os.path.join(ROOT_DIR, "test_results")
WEIGHT_DIR = "weight"  # 权重保存目录

# 训练/测试模式控制（train=训练，test=测试）
MODE = "train"
# 测试模式下指定的权重文件路径
TEST_WEIGHT_PATH = "weight/20251227_174219_best_loss_0.0242_fp16.pth"  # 替换为你的权重路径

# 训练参数
BATCH_SIZE = 12
EPOCHS = 50
LEARNING_RATE = 2e-4
TARGET_SIZE = (512, 512)  # 训练时统一的目标尺寸
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_THRESHOLD = 127  # 掩码二值化阈值
RANDOM_SEED = 42  # 随机种子（保证划分结果可复现）


# ====================== 2. 工具函数 ======================
class ResizeWithPad:
    """保持图片比例的resize，不足部分填充黑色"""

    def __init__(self, target_size):
        self.target_size = target_size  # (width, height)

    def __call__(self, img):
        original_w, original_h = img.size
        target_w, target_h = self.target_size

        scale = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        pad_right = target_w - new_w - pad_left
        pad_bottom = target_h - new_h - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        img = transforms.Pad(padding, fill=0)(img)

        return img


def create_dirs():
    """创建所需目录"""
    dirs = [
        TRAIN_DIR, VAL_DIR, TEST_DIR, WEIGHT_DIR,
        os.path.join(TRAIN_DIR, "images"), os.path.join(TRAIN_DIR, "masks"),
        os.path.join(VAL_DIR, "images"), os.path.join(VAL_DIR, "masks"),
        os.path.join(TEST_DIR, "images"), os.path.join(TEST_DIR, "masks")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def split_dataset_from_data():
    """
    从data目录划分数据集：
    1. 先清空val/test目录并将文件移回data（如果存在）
    2. 从data中随机划分70%train/20%val/10%test
    """
    # 步骤1：将val/test目录下的文件移回data目录
    for split in [VAL_DIR, TEST_DIR]:
        for sub_dir in ["images", "masks"]:
            src_dir = os.path.join(split, sub_dir)
            dst_dir = os.path.join(DATA_DIR, sub_dir)
            if os.path.exists(src_dir):
                for file in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, file)
                    dst_path = os.path.join(dst_dir, file)
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    shutil.move(src_path, dst_path)
                # 清空子目录
                shutil.rmtree(src_dir)
    # exit()
    # 步骤2：获取data目录下的所有图片-掩码配对
    img_dir = os.path.join(DATA_DIR, "images")
    mask_dir = os.path.join(DATA_DIR, "masks")

    img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    img_names = [f for f in os.listdir(img_dir) if any(f.lower().endswith(ext) for ext in img_extensions)]
    random.seed(RANDOM_SEED)
    random.shuffle(img_names)

    # 步骤3：划分数据集（70%train, 20%val, 10%test）
    # 先分train+val(90%) 和 test(10%)
    train_val_imgs, test_imgs = train_test_split(img_names, test_size=0.1, random_state=RANDOM_SEED)
    # 再从train+val中分train(70%) 和 val(20%)
    train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=2 / 9, random_state=RANDOM_SEED)

    # 步骤4：复制文件到对应目录
    def copy_files(img_list, split_dir):
        img_dst = os.path.join(split_dir, "images")
        mask_dst = os.path.join(split_dir, "masks")
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(mask_dst, exist_ok=True)

        for img_name in img_list:
            # 复制图片
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(img_dst, img_name)
            shutil.copy(src_img, dst_img)

            # 复制掩码（支持_mask.png/_mask2.png）
            img_base = os.path.splitext(img_name)[0]
            mask_candidates = [f"{img_base}_mask.png", f"{img_base}_mask2.png"]
            for mask_name in mask_candidates:
                src_mask = os.path.join(mask_dir, mask_name)
                if os.path.exists(src_mask):
                    dst_mask = os.path.join(mask_dst, mask_name)
                    shutil.copy(src_mask, dst_mask)
                    break

    copy_files(train_imgs, TRAIN_DIR)
    copy_files(val_imgs, VAL_DIR)
    copy_files(test_imgs, TEST_DIR)

    print(f"✅ 数据集划分完成！")
    print(f"训练集：{len(train_imgs)} 张 | 验证集：{len(val_imgs)} 张 | 测试集：{len(test_imgs)} 张")


def save_weight_with_time(model, val_loss, optimizer=None):
    """保存float16半精度权重，大幅减小体积"""
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_str = f"{val_loss:.4f}"
    weight_name = f"{timestamp}_best_loss_{loss_str}_fp16.pth"
    weight_path = os.path.join(WEIGHT_DIR, weight_name)

    # 将模型参数转为float16
    model_fp16_state_dict = {k: v.half() for k, v in model.state_dict().items()}
    torch.save(model_fp16_state_dict, weight_path, _use_new_zipfile_serialization=True)

    print(f"💾 Float16半精度权重已保存至：{weight_path}")
    return weight_path


# ====================== 3. 自定义数据集类 ======================
class MetSegDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform
        self.split_dir = {
            "train": TRAIN_DIR,
            "val": VAL_DIR,
            "test": TEST_DIR
        }[split]

        self.img_dir = os.path.join(self.split_dir, "images")
        self.mask_dir = os.path.join(self.split_dir, "masks")
        self.pairs = self._build_pairs()

    def _build_pairs(self):
        """匹配原图和掩码"""
        pairs = []
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        for img_file in os.listdir(self.img_dir):
            if not any(img_file.lower().endswith(ext) for ext in img_extensions):
                continue

            img_base = os.path.splitext(img_file)[0]
            mask_file = None
            for suffix in ["_mask.png", "_mask2.png"]:
                candidate = os.path.join(self.mask_dir, img_base + suffix)
                if os.path.exists(candidate):
                    mask_file = candidate
                    break

            if mask_file:
                pairs.append((
                    os.path.join(self.img_dir, img_file),
                    mask_file,
                    img_base
                ))

        return pairs

    def _process_mask(self, mask_path):
        """处理透明通道/黑白掩码"""
        mask = Image.open(mask_path)
        if mask.mode == "RGBA":
            mask = mask.split()[-1]  # 提取Alpha通道
        elif mask.mode == "RGB":
            mask = mask.convert("L")
        elif mask.mode != "L":
            mask = mask.convert("L")

        mask_np = np.array(mask)
        mask_np = (mask_np > MASK_THRESHOLD).astype(np.uint8)
        return Image.fromarray(mask_np)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, img_base = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        mask = self._process_mask(mask_path)

        if self.transform:
            img = self.transform(img)
            mask_transform = transforms.Compose([
                ResizeWithPad(TARGET_SIZE),
                transforms.ToTensor()
            ])
            mask = mask_transform(mask)
            mask = (mask > 0).float()

        return {
            "image": img,
            "mask": mask,
            "img_base": img_base,
            "original_size": original_size,
            "img_path": img_path
        }


# ====================== 4. 训练/验证/测试函数 ======================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="训练")

    for batch in pbar:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc="验证")

    with torch.no_grad():
        for batch in pbar:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def infer_single_image(img_path, model, save_dir, device=DEVICE, threshold=0.5):
    """单图推理并保存掩码"""
    # 加载原图
    img = Image.open(img_path).convert("RGB")
    original_size = img.size
    img_base = os.path.splitext(os.path.basename(img_path))[0]

    # 预处理
    transform = transforms.Compose([
        ResizeWithPad(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = (output > threshold).float().squeeze(0).squeeze(0).cpu().numpy()
        pred_mask = (pred_mask * 255).astype(np.uint8)

    # 还原原图尺寸
    pred_mask_img = Image.fromarray(pred_mask)
    target_w, target_h = TARGET_SIZE
    original_w, original_h = original_size
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pred_mask_img = pred_mask_img.crop((pad_left, pad_top, pad_left + new_w, pad_top + new_h))

    pred_mask_img = pred_mask_img.resize(original_size, Image.NEAREST)

    # 保存掩码
    save_path = os.path.join(save_dir, f"{img_base}_pred_mask.png")
    pred_mask_img.save(save_path)
    return save_path


def test_model(model, weight_path, test_loader, device=DEVICE):
    """测试模式：加载权重，对测试集所有图片推理"""
    # 加载权重
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    print(f"✅ 加载权重完成：{weight_path}")
    if os.path.exists(TEST_OUT_DIR):
        shutil.rmtree(TEST_OUT_DIR)
        print('以往测试结果已删除')
    # 创建测试结果保存目录
    os.makedirs(TEST_OUT_DIR, exist_ok=True)

    # 对测试集逐图推理
    print("\n📝 开始对测试集推理...")
    pbar = tqdm(test_loader, desc="测试推理")
    for batch in pbar:
        img_paths = batch["img_path"]
        for img_path in img_paths:
            infer_single_image(img_path, model, TEST_OUT_DIR, device)

    print(f"✅ 测试完成！所有掩码已保存至：{TEST_OUT_DIR}")


# ====================== 5. 主流程 ======================
if __name__ == "__main__":
    # 初始化目录
    create_dirs()

    # ====================== 训练模式 ======================
    if MODE == "train":
        # 1. 划分数据集（从data目录拆分train/val/test）
        split_dataset_from_data()

        # 2. 创建数据集和加载器
        train_transform = transforms.Compose([
            ResizeWithPad(TARGET_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_test_transform = transforms.Compose([
            ResizeWithPad(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = MetSegDataset(split="train", transform=train_transform)
        val_dataset = MetSegDataset(split="val", transform=val_test_transform)
        test_dataset = MetSegDataset(split="test", transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"\n📊 数据集统计：")
        print(f"训练集：{len(train_dataset)} 样本 | 验证集：{len(val_dataset)} 样本 | 测试集：{len(test_dataset)} 样本")

        # 3. 初始化模型和优化器
        model = ResNetUNet(n_channels=3, n_classes=1).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # 4. 开始训练（仅保存损失更小的权重）
        best_val_loss = float('inf')  # 初始化最优损失为无穷大
        print(f"\n🚀 开始训练ResNet-Unet（设备：{DEVICE}）")
        for epoch in range(EPOCHS):
            print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss = validate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_loss)
            print(f"📈 训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f}")

            # 仅当当前验证损失小于历史最优损失时，才保存权重
            if val_loss < best_val_loss and epoch >= 12:
                best_val_loss = val_loss  # 更新最优损失
                save_weight_with_time(model, best_val_loss, optimizer)  # 保存最优权重

        print("\n🎉 训练完成！仅保存了验证损失最小的最优模型至weight目录")

    # ====================== 测试模式 ======================
    elif MODE == "test":
        # 从test模块导入并运行测试
        from utils import run_test

        run_test(
            weight_path=TEST_WEIGHT_PATH,
            test_out_dir=TEST_OUT_DIR,
            batch_size=BATCH_SIZE
        )

    else:
        print(f"❌ 无效的mode值：{MODE}，请设置为'train'或'test'")
