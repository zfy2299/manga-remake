from pathlib import Path
import photoshop.api as ps

# 1. 设定路径
SRC_DIR = Path(r"F:\JHenTai_data\待翻新\[赤城あさひと]")  # 原始图片目录
DEST_NAME_sub = 'EN_赤城 ノーまん'  # 输出名字的前缀
DEST_DIR = Path(r"../dataset/psd")  # 输出 PSD 目录
DEST_DIR.mkdir(exist_ok=True)

# 2. 支持的图片后缀（Photoshop 能打开即可）
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}

# 3. 启动 Photoshop
app = ps.Application()

# 4. 遍历图片
for img_path in SRC_DIR.iterdir():
    if img_path.suffix.lower() not in IMG_EXT:
        continue

    # 4-1. 打开图片
    doc = app.open(str(img_path.resolve()))
    if doc.mode == 6:
        doc.changeMode(1)  # 1是灰度，2是RGB，3是CMYK，6是索引
    # 4-2. 构造输出路径
    if DEST_NAME_sub:
        psd_path = DEST_DIR / f"{DEST_NAME_sub}_{img_path.stem}.psd"
    else:
        psd_path = DEST_DIR / f"{img_path.stem}.psd"

    # 4-3. 以 PSD 格式保存
    psd_options = ps.PhotoshopSaveOptions()
    doc.saveAs(str(psd_path.resolve()), psd_options)

    # 4-4. 关闭文档（不保存更改）
    doc.close(ps.SaveOptions.DoNotSaveChanges)

print("全部转换完成！")
