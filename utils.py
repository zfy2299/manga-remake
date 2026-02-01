from pathlib import Path

import cv2
import photoshop.api as ps
from photoshop import Session
import shutil
import numpy as np
from PIL import Image
import pillow_avif
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import os

from train import ResizeWithPad, MetSegDataset, TARGET_SIZE, DEVICE


def isGrayMap(img, threshold=10, debug=False):
    """
    入参：
    img：PIL读入的图像
    threshold：判断阈值，图片3个通道间差的方差均值小于阈值则判断为灰度图。
    阈值设置的越小，容忍出现彩色面积越小；设置的越大，那么就可以容忍出现一定面积的彩色，例如微博截图。
    如果阈值设置的过小，某些灰度图片会被漏检，这是因为某些黑白照片存在偏色，例如发黄的黑白老照片、
    噪声干扰导致灰度图不同通道间值出现偏差（理论上真正的灰度图是RGB三个通道的值完全相等或者只有一个通道，
    然而实际上各通道间像素值略微有偏差看起来仍是灰度图）
    出参：
    bool值
    """
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if debug:
        print(f"差异值为：{diff_sum}")
    if diff_sum <= threshold:
        return True
    else:
        return False


def infer_single_image(img_path, model, save_dir='', device=DEVICE, threshold=0.5):
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


def test_model(model, weight_path, test_loader, test_out_dir, device=DEVICE):
    """测试模式：加载权重，对测试集所有图片推理"""
    # 加载权重
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    print(f"✅ 加载权重完成：{weight_path}")

    # 清理并创建测试结果目录
    if os.path.exists(test_out_dir):
        shutil.rmtree(test_out_dir)
        print('以往测试结果已删除')
    os.makedirs(test_out_dir, exist_ok=True)

    # 对测试集逐图推理
    print("\n📝 开始对测试集推理...")
    pbar = tqdm(test_loader, desc="测试推理")
    for batch in pbar:
        img_paths = batch["img_path"]
        for img_path in img_paths:
            infer_single_image(img_path, model, test_out_dir, device)

    print(f"✅ 测试完成！所有掩码已保存至：{test_out_dir}")


def run_test(weight_path, test_out_dir, batch_size=12):
    """运行测试的主函数"""
    # 定义测试集变换
    val_test_transform = transforms.Compose([
        ResizeWithPad(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建测试数据集和加载器
    test_dataset = MetSegDataset(split="test", transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型并测试
    from ResNetUNet import ResNetUNet
    model = ResNetUNet(n_channels=3, n_classes=1).to(DEVICE)
    test_model(model, weight_path, test_loader, test_out_dir, DEVICE)


def ps_auto_composite_layers(bg_img_path, top_img_path, mask_img_path, save_psd_path, auto_gray=False, cv2_align=True,
                             color_level=None, filter_blur=None, filter_sharp=None, do_action=None):
    """

    :param cv2_align:
    :param bg_img_path:
    :param top_img_path:
    :param mask_img_path:
    :param auto_gray:
    :param color_level: 色阶的参数，比如：黑场12、白场230、灰场0.8 -> {'black': 12, 'white': 230, 'gray': 0.8}
    :param filter_blur: 表面模糊的参数，推荐：半径3、阈值8 -> {'radius': 3, 'threshold': 8}
    :param filter_sharp: USM锐化的参数，推荐：数量65、半径1、阈值8 -> {'quantity': 65, 'radius': 1, 'threshold': 8}
    :param do_action: 关闭前要运行的动作，比如['动作分组名', '动作名']
    :param save_psd_path:
    :return:
    """
    # ========== 相对路径 → PS支持的绝对路径（必做） ==========
    bg_img_path = os.path.abspath(os.path.normpath(bg_img_path))
    top_img_path = os.path.abspath(os.path.normpath(top_img_path))
    if mask_img_path is not None:
        mask_img_path = os.path.abspath(os.path.normpath(mask_img_path))
    save_psd_path = os.path.abspath(os.path.normpath(save_psd_path))

    # ========== 文件有效性校验 ==========
    file_check = [(bg_img_path, "底图"), (top_img_path, "上层图")]
    if mask_img_path is not None:
        file_check.append((mask_img_path, "MASK图"))
    for path, name in file_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ {name}文件不存在 → {path}")
    app = ps.Application()
    doc = app.open(bg_img_path)
    if doc.mode == 6:
        doc.changeMode(1)  # 1是灰度，2是RGB，3是CMYK，6是索引
    bg_layer = doc.artLayers[0]
    bg_layer.name = "背景图层"
    # ========== 灰度图判定 ==========
    is_gray = isGrayMap(Image.open(bg_img_path))
    if auto_gray and doc.channels.length > 1 and is_gray:
        app.doJavaScript("app.activeDocument.changeMode(ChangeMode.GRAYSCALE);")
    # 导入图层并对齐
    if cv2_align:
        top_img_path = align_images(bg_img_path, top_img_path)
    with Session() as ps_:
        desc = ps_.ActionDescriptor
        desc.putPath(ps_.app.charIDToTypeID("null"), top_img_path)
        ps_.app.executeAction(ps_.app.charIDToTypeID("Plc "), desc)
    doc.activeLayer.rasterize(5)
    up_layer = doc.artLayers[0]
    up_layer.name = "上层图层"
    if not cv2_align:
        stdlib_js = open('stdlib.js', encoding='utf-8').read()
        stdlib_js += "Stdlib.loadSelection(doc, doc.artLayers.getByName('背景图层'), 'Trsp');Stdlib.crop(doc);"
        app.doJavaScript(stdlib_js)
    bg_layer.isBackgroundLayer = True
    # 色阶
    if color_level and auto_gray and is_gray:
        app.doJavaScript(f"""
            var desc284 = new ActionDescriptor();
            var idpresetKind = stringIDToTypeID( "presetKind" );
            var idpresetKindType = stringIDToTypeID( "presetKindType" );
            var idpresetKindCustom = stringIDToTypeID( "presetKindCustom" );
            desc284.putEnumerated( idpresetKind, idpresetKindType, idpresetKindCustom );
            var list4 = new ActionList();
            var desc285 = new ActionDescriptor();
            var idChnl = charIDToTypeID( "Chnl" );
            var ref3 = new ActionReference();
            var idChnl = charIDToTypeID( "Chnl" );
            ref3.putEnumerated( idChnl, charIDToTypeID( "Ordn" ), charIDToTypeID( "Trgt" ) );
            desc285.putReference( idChnl, ref3 );
            var list5 = new ActionList();
            list5.putInteger( {color_level['black']} );
            list5.putInteger( {color_level['white']} );
            desc285.putList( charIDToTypeID( "Inpt" ), list5 );
            desc285.putDouble( charIDToTypeID( "Gmm " ), {color_level['gray']} );
            list4.putObject( charIDToTypeID( "LvlA" ), desc285 );
            desc284.putList( charIDToTypeID( "Adjs" ), list4 );
            executeAction( charIDToTypeID( "Lvls" ), desc284, DialogModes.NO );
        """)
    # 表面模糊
    if filter_blur:
        app.doJavaScript(f"""
            var desc227 = new ActionDescriptor();
            desc227.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), {filter_blur['radius']});
            desc227.putInteger(charIDToTypeID("Thsh"), {filter_blur['threshold']} );
            executeAction(stringIDToTypeID("surfaceBlur"), desc227, DialogModes.NO );
        """)
    # USM锐化
    if filter_sharp:
        app.doJavaScript(f"""
            var desc256 = new ActionDescriptor();
            desc256.putUnitDouble(charIDToTypeID("Amnt"), charIDToTypeID("#Prc"), {filter_sharp['quantity']});
            desc256.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), {filter_sharp['radius']});
            desc256.putInteger(charIDToTypeID("Thsh"), {filter_sharp['threshold']});
            executeAction(idUnsM = charIDToTypeID("UnsM"), desc256, DialogModes.NO);
        """)
    if mask_img_path is not None:
        with Session() as ps_:
            desc = ps_.ActionDescriptor
            desc.putPath(ps_.app.charIDToTypeID("null"), mask_img_path)
            ps_.app.executeAction(ps_.app.charIDToTypeID("Plc "), desc)
        mask_layer = doc.artLayers[0]
        mask_layer.rasterize(5)
        mask_layer.name = "mask"
        # 将白色像素载入选区
        app.doJavaScript(r"""
            var desc = new ActionDescriptor();
            var ref = new ActionReference();
            ref.putProperty(stringIDToTypeID("channel"), stringIDToTypeID("selection"));
            desc.putReference(charIDToTypeID("null"), ref);
            desc.putInteger(charIDToTypeID("fzns"), 0); 
            desc.putDouble(stringIDToTypeID("H"), 0); 
            desc.putDouble(stringIDToTypeID("H_1"), 0); 
            desc.putEnumerated(stringIDToTypeID("sample"), stringIDToTypeID("sampleFrom"), stringIDToTypeID("currentLayer"));
            executeAction(stringIDToTypeID("colorRange"), desc, DialogModes.NO);
        """)
        mask_layer.visible = False
    doc.activeLayer = up_layer
    # 将选区应用为蒙版
    app.doJavaScript(r"""
        try {
            var hasSelection = app.activeDocument.selection.bounds;
            var desc220 = new ActionDescriptor();
            var idChnl = charIDToTypeID( "Chnl" );
            desc220.putClass( charIDToTypeID( "Nw  " ), idChnl );
            var ref1 = new ActionReference();
            ref1.putEnumerated( idChnl, idChnl, charIDToTypeID( "Msk " ) );
            desc220.putReference( charIDToTypeID( "At  " ), ref1 );
            desc220.putEnumerated( charIDToTypeID( "Usng" ), charIDToTypeID( "UsrM" ), charIDToTypeID( "RvlS" ) );
            executeAction( charIDToTypeID( "Mk  " ), desc220, DialogModes.NO );
        } catch(e) {
            var desc219 = new ActionDescriptor();
            var idChnl = charIDToTypeID( "Chnl" );
            desc219.putClass( charIDToTypeID( "Nw  " ), idChnl );
            var idAt = charIDToTypeID( "At  " );
            var ref1 = new ActionReference();
            ref1.putEnumerated( idChnl, idChnl, charIDToTypeID( "Msk " ) );
            desc219.putReference( idAt, ref1 );
            desc219.putEnumerated( charIDToTypeID( "Usng" ), charIDToTypeID( "UsrM" ), charIDToTypeID( "RvlA" ) );
            executeAction( charIDToTypeID( "Mk  " ), desc219, DialogModes.NO );
            var desc226 = new ActionDescriptor();
            var idClr = charIDToTypeID( "Clr " );
            desc226.putEnumerated( charIDToTypeID( "Usng" ), charIDToTypeID( "FlCn" ), idClr );
            var desc227 = new ActionDescriptor();
            desc227.putUnitDouble( charIDToTypeID( "H   " ), charIDToTypeID( "#Ang" ), 299.992676 );
            desc227.putDouble( charIDToTypeID( "Strt" ), 0.000000 );
            desc227.putDouble( charIDToTypeID( "Brgh" ), 0.000000 );
            desc226.putObject( idClr, charIDToTypeID( "HSBC" ), desc227 );
            desc226.putUnitDouble( charIDToTypeID( "Opct" ), charIDToTypeID( "#Prc" ), 100.000000 );
            desc226.putEnumerated( charIDToTypeID( "Md  " ), charIDToTypeID( "BlnM" ), charIDToTypeID( "Nrml" ) );
            executeAction( charIDToTypeID( "Fl  " ), desc226, DialogModes.NO );
        }
    """)
    if do_action:
        app.doAction(do_action[1], do_action[0])
    # 72dpi
    app.doJavaScript("""
        var desc1 = new ActionDescriptor();
        desc1.putUnitDouble(charIDToTypeID('Rslt'), charIDToTypeID('#Rsl'), 72);
        executeAction(stringIDToTypeID('imageSize'), desc1, DialogModes.NO);
    """)
    doc.saveAs(save_psd_path, ps.PhotoshopSaveOptions())
    doc.close(ps.SaveOptions.DoNotSaveChanges)


def match_comics_2(folder_a, folder_b, match_from_son=False):
    r_ = '**/*' if match_from_son else '*'
    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载预训练模型
    if os.path.exists('pth/resnet50-11ad3fa6.pth'):
        model = models.resnet50(weights=None)
        state_dict = torch.load('pth/resnet50-11ad3fa6.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    # 将模型移动到 GPU 上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义函数来提取图像特征
    def extract_features(image_path):
        img = Image.open(image_path)
        # 确保图像是 RGB 格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_t = preprocess(img)
        img_t = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_t)
        return features

    # 定义函数来计算两个特征向量之间的余弦相似度
    def cosine_similarity(feat1, feat2):
        return F.cosine_similarity(feat1, feat2)

    # 获取文件夹 A 和 B 中的图片路径
    support_images = ('.png', '.jpg', '.jpeg', '.webp', '.avif')
    images_a = [
        str(img_path.absolute())
        for img_path in Path(folder_a).glob('*')
        if img_path.is_file() and img_path.suffix.lower() in support_images
    ]
    images_b = [
        str(img_path.absolute())
        for img_path in Path(folder_b).glob(r_)
        if img_path.is_file() and img_path.suffix.lower() in support_images
    ]

    # 为文件夹 B 中的每张图片提取特征
    features_b = [extract_features(img_path) for img_path in images_b]

    # 初始化匹配字典
    match_dict = []

    # 遍历文件夹 A 中的每张图片，找到与之相似度最高的图片
    for img_path_a in images_a:
        features_a = extract_features(img_path_a)
        similarities = [cosine_similarity(features_a, features_b[i]).item() for i in range(len(features_b))]
        max_similarity = max(similarities)
        max_similarity_index = similarities.index(max_similarity)
        most_similar_img_path = images_b[max_similarity_index]
        match_dict.append({
            'raw': os.path.basename(img_path_a),
            'rawPath': img_path_a,
            'match': os.path.basename(most_similar_img_path),
            'matchPath': most_similar_img_path,
            'matchRatio': max_similarity
        })
    return {'match_result': match_dict, 'a_num': len(images_a), 'b_num': len(images_b)}


def split_image(img_dir, match_from_son=False):
    r_ = '**/*' if match_from_son else '*'
    images_ = [
        p for p in Path(img_dir).glob(r_)
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'] and p.is_file()
    ]
    for img_path in images_:
        out_dir = img_path.parent
        try:
            with Image.open(img_path) as im:
                width, height = im.size
                if width > height:
                    mid = width // 2
                    left = im.crop((0, 0, mid, height))
                    right = im.crop((mid, 0, width, height))
                    base = img_path.stem
                    ext = img_path.suffix.lower()
                    left_path = out_dir / f"{base}_2{ext}"
                    right_path = out_dir / f"{base}_1{ext}"
                    save_kwargs = {}
                    if ext in {'.jpg', '.jpeg'} and im.mode in ('RGBA', 'LA', 'P'):
                        left = left.convert('RGB')
                        right = right.convert('RGB')
                    if ext == '.png':
                        save_kwargs['compress_level'] = im.info.get('compress_level', 9)
                    icc = im.info.get('icc_profile')
                    if icc:
                        save_kwargs['icc_profile'] = icc
                    left.save(left_path, format=Image.EXTENSION[ext], **save_kwargs)
                    right.save(right_path, format=Image.EXTENSION[ext], **save_kwargs)
                    print(f"拆分了图片：f{img_path}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")


def align_images(
        ref_path,
        img_path,
        output_dir="temp_align",
        min_good_matches=100,
        fill_color=(255, 255, 255),  # 新增：填充颜色，默认白色
        print_log=False
):
    """
    将 img_path 的图片对齐到 ref_path 的图片空间，并保存对齐后的结果。
    支持自定义 warpPerspective 的填充颜色（例如白色 (255,255,255)）。

    参数:
        ref_path: 目标参考图片路径（en.png 风格）
        img_path: 需要对齐的图片路径（zh.jpg 风格）
        output_dir: 输出目录，默认为当前目录
        min_good_matches: 最低有效匹配点数阈值，低于此值视为失败
        fill_color: 填充区域的颜色 (B, G, R)，默认 (0,0,0) 黑色

    返回:
        str | None: 对齐成功时返回保存的文件路径，失败时返回 None
    """
    # 读取图片
    ref = cv2.imread(ref_path)
    img = cv2.imread(img_path)

    if ref is None or img is None:
        if print_log:
            print(f"读取图片失败：ref={ref_path}, img={img_path}")
        return None

    # 转灰度
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # AKAZE 特征检测（对漫画线条友好）
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(ref_gray, None)
    kp2, des2 = detector.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        if print_log:
            print(f"特征检测失败，无法提取描述子：ref={ref_path}, img={img_path}")
        return None

    # 匹配 + 比率测试
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if print_log:
        print(f"有效匹配点数：{len(good)}")

    if len(good) < min_good_matches:
        if print_log:
            print(f"匹配点不足（{len(good)} < {min_good_matches}），对齐失败：ref={ref_path}, img={img_path}")
        return None

    # 提取匹配点坐标
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # 计算单应矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inlier_ratio = mask.sum() / len(good) if len(good) > 0 else 0
    if print_log:
        print(f"内点比例：{inlier_ratio:.2%}")

    if inlier_ratio < 0.5:
        if print_log:
            print(f"内点比例过低，对齐不可靠：ref={ref_path}, img={img_path}")
        return None

    # 进行透视变换对齐，支持自定义填充颜色
    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(
        img,
        H,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_color  # 这里控制填充颜色
    )

    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_aligned.png")

    # 保存
    success = cv2.imwrite(output_path, aligned)
    if not success:
        if print_log:
            print(f"保存失败：{output_path}：ref={ref_path}, img={img_path}")
        return None
    if print_log:
        print(f"对齐完成，已保存至：{output_path}")
    return os.path.abspath(os.path.normpath(output_path))


if __name__ == "__main__":
    # # 测试：检测是否黑白图
    # test_img_gray = Image.open(r"F:\CH1 Visiting Home (COMIC X-Eros #52) (02).png")
    # print(isGrayMap(test_img_gray, debug=True))

    # 测试：使图片B向图片A对齐
    aligned_path_white = align_images(ref_path=r"F:\[赤城あさひと] 反り (あま❤ナマ)\09_01.png",
                                      img_path=r"F:\[赤城あさひと] 反り (あま❤ナマ)\汉化\STARS_18057_132.jpg", )
    if aligned_path_white:
        print(f"成功生成：{aligned_path_white}")
    else:
        print("对齐失败")
