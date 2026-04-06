from pathlib import Path

import cv2
import photoshop.api as ps
from photoshop import Session
import shutil
import numpy as np
from PIL import Image
import pillow_avif
from photoshop.api import PNGSaveOptions, SaveOptions, JPEGSaveOptions
from skimage import exposure
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


def cv2_imread_unicode(file_path, flags=cv2.IMREAD_COLOR):
    """
    解决 cv2.imread 无法读取中文路径的问题

    参数:
        file_path: 图片文件路径（支持中文）
        flags: cv2 读取标志，默认为 cv2.IMREAD_COLOR

    返回:
        numpy.ndarray 或 None
    """
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
        return img
    except Exception as e:
        print(f"读取图片失败 {file_path}: {e}")
        return None


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


def load_white_to_selection(ps_app):
    # 将白色像素载入选区
    ps_app.doJavaScript(r"""
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


def ps_auto_composite_layers(bg_img_path, top_img_path, mask_img_path, save_psd_path, auto_gray=False, cv2_align=True,
                             color_align=True, color_level=None, filter_blur=None, filter_sharp=None,
                             filter_only_selection=False, selection_contract=0, selection_feather=0, do_action=None):
    """

    :param filter_only_selection:
    :param selection_feather:
    :param selection_contract:
    :param color_align:
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
    align_res = None
    if cv2_align:
        align_res = align_images(bg_img_path, top_img_path)
        if align_res is not None:
            top_img_path = align_res
    if color_align:
        temp_align_res = align_image_color_v2(top_img_path, bg_img_path)
        if temp_align_res:
            top_img_path = temp_align_res
    with Session() as ps_:
        desc = ps_.ActionDescriptor
        desc.putPath(ps_.app.charIDToTypeID("null"), top_img_path)
        ps_.app.executeAction(ps_.app.charIDToTypeID("Plc "), desc)
    doc.activeLayer.rasterize(5)
    up_layer = doc.artLayers[0]
    up_layer.name = "上层图层"
    if align_res is not None or not cv2_align:
        stdlib_js = open('stdlib.js', encoding='utf-8').read()
        stdlib_js += "Stdlib.loadSelection(doc, doc.artLayers.getByName('背景图层'), 'Trsp');Stdlib.crop(" \
                     "doc);app.activeDocument.selection.deselect(); "
        app.doJavaScript(stdlib_js)
    bg_layer.isBackgroundLayer = True
    mask_layer = None
    if mask_img_path is not None:
        with Session() as ps_:
            desc = ps_.ActionDescriptor
            desc.putPath(ps_.app.charIDToTypeID("null"), mask_img_path)
            ps_.app.executeAction(ps_.app.charIDToTypeID("Plc "), desc)
        mask_layer = doc.artLayers[0]
        mask_layer.visible = True
        mask_layer.rasterize(5)
        mask_layer.name = "mask"
    # 各种滤镜
    selection = None
    if filter_only_selection:
        load_white_to_selection(app)
        try:
            selection = app.activeDocument.selection
            t_ = selection.bounds
            selection.invert()
        except:
            pass
    doc.activeLayer = up_layer
    # 表面模糊
    if filter_blur:
        app.doJavaScript(f"""
            var desc227 = new ActionDescriptor();
            desc227.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), {filter_blur['radius']});
            desc227.putInteger(charIDToTypeID("Thsh"), {filter_blur['threshold']} );
            executeAction(stringIDToTypeID("surfaceBlur"), desc227, DialogModes.NO );
        """)
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
    # USM锐化
    if filter_sharp:
        app.doJavaScript(f"""
            var desc256 = new ActionDescriptor();
            desc256.putUnitDouble(charIDToTypeID("Amnt"), charIDToTypeID("#Prc"), {filter_sharp['quantity']});
            desc256.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), {filter_sharp['radius']});
            desc256.putInteger(charIDToTypeID("Thsh"), {filter_sharp['threshold']});
            executeAction(idUnsM = charIDToTypeID("UnsM"), desc256, DialogModes.NO);
        """)
    if mask_img_path is not None and mask_layer is not None:
        # 先取消选区
        if selection is not None:
            try:
                t_ = selection.bounds
                selection.deselect()
            except:
                pass
        doc.activeLayer = mask_layer
        load_white_to_selection(app)
        mask_layer.visible = False
        try:
            selection = app.activeDocument.selection
            t_ = selection.bounds
            if selection_contract:
                selection.contract(selection_contract)  # 收缩选区
            if selection_feather:
                selection.feather(selection_feather)  # 羽化
        except:
            pass
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
            desc227.putUnitDouble( charIDToTypeID( "H   " ), charIDToTypeID( "#Ang" ), 300 );
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


def ps_auto_white(img_path, mask_img_path, save_psd_path, auto_gray=False, white_opacity=100,
                  color_level=None, filter_blur=None, filter_sharp=None,
                  selection_contract=0, selection_feather=0, do_action=None):
    """
    仅涂白
    :param img_path:
    :param white_opacity:
    :param selection_feather:
    :param selection_contract:
    :param mask_img_path:
    :param auto_gray:
    :param color_level: 色阶的参数，比如：黑场12、白场230、灰场0.8 -> {'black': 12, 'white': 230, 'gray': 0.8}
    :param filter_blur: 表面模糊的参数，推荐：半径3、阈值8 -> {'radius': 3, 'threshold': 8}
    :param filter_sharp: USM锐化的参数，推荐：数量65、半径1、阈值8 -> {'quantity': 65, 'radius': 1, 'threshold': 8}
    :param do_action: 关闭前要运行的动作，比如['动作分组名', '动作名']
    :param save_psd_path:
    :return:
    """
    if mask_img_path is not None:
        mask_img_path = os.path.abspath(os.path.normpath(mask_img_path))
    # ========== 文件有效性校验 ==========
    file_check = [(img_path, "底图")]
    print(mask_img_path)
    if mask_img_path is not None:
        file_check.append((mask_img_path, "MASK图"))
    for path, name in file_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ {name}文件不存在 → {path}")
    app = ps.Application()
    doc = app.open(img_path)
    if doc.mode == 6:
        doc.changeMode(1)  # 1是灰度，2是RGB，3是CMYK，6是索引
    bg_layer = doc.artLayers[0]
    bg_layer.name = "背景图层"
    # ========== 灰度图判定 ==========
    is_gray = isGrayMap(Image.open(img_path))
    if auto_gray and doc.channels.length > 1 and is_gray:
        app.doJavaScript("app.activeDocument.changeMode(ChangeMode.GRAYSCALE);")
    # 各种滤镜
    # 表面模糊
    if filter_blur:
        app.doJavaScript(f"""
            var desc227 = new ActionDescriptor();
            desc227.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), {filter_blur['radius']});
            desc227.putInteger(charIDToTypeID("Thsh"), {filter_blur['threshold']} );
            executeAction(stringIDToTypeID("surfaceBlur"), desc227, DialogModes.NO );
        """)
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
    # USM锐化
    if filter_sharp:
        app.doJavaScript(f"""
            var desc256 = new ActionDescriptor();
            desc256.putUnitDouble(charIDToTypeID("Amnt"), charIDToTypeID("#Prc"), {filter_sharp['quantity']});
            desc256.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), {filter_sharp['radius']});
            desc256.putInteger(charIDToTypeID("Thsh"), {filter_sharp['threshold']});
            executeAction(idUnsM = charIDToTypeID("UnsM"), desc256, DialogModes.NO);
        """)
    # 导入mask图层
    bg_layer.isBackgroundLayer = True
    mask_layer = None
    if mask_img_path is not None:
        with Session() as ps_:
            desc = ps_.ActionDescriptor
            desc.putPath(ps_.app.charIDToTypeID("null"), mask_img_path)
            ps_.app.executeAction(ps_.app.charIDToTypeID("Plc "), desc)
        mask_layer = doc.artLayers[0]
        mask_layer.visible = True
        mask_layer.rasterize(5)
        mask_layer.name = "mask"
    if mask_img_path is not None and mask_layer is not None:
        # 根据蒙版建立选区
        doc.activeLayer = mask_layer
        load_white_to_selection(app)
        mask_layer.visible = False
        try:
            selection = app.activeDocument.selection
            t_ = selection.bounds
            if selection_contract:
                selection.contract(selection_contract)  # 收缩选区
            if selection_feather:
                selection.feather(selection_feather)  # 羽化
            # 涂白
            selection = app.activeDocument.selection
            t_ = selection.bounds
            white_layer = doc.artLayers.add()
            white_layer.name = "涂白"
            fill_color = ps.SolidColor()
            fill_color.rgb.red = 255
            fill_color.rgb.green = 255
            fill_color.rgb.blue = 255
            doc.selection.fill(fill_color)
            white_layer.opacity = white_opacity
        except:
            pass
        mask_layer.remove()
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


def match_comics_2(folder_a, folder_b, match_from_son=False, match_twice=False, match_twice_point=100,
                   match_twice_start=0.5):
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
    support_images = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif')
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
        final_ratio = max_similarity
        if match_twice and final_ratio >= match_twice_start:
            is_good, _, _, _ = find_good_matches(
                ref_gray=cv2_imread_unicode(most_similar_img_path, cv2.COLOR_BGR2GRAY),
                img_gray=cv2_imread_unicode(img_path_a, cv2.COLOR_BGR2GRAY),
                min_good_matches=100,
                print_log=False
            )
            if not is_good:
                final_ratio = 0.0
        match_dict.append({
            'raw': os.path.basename(img_path_a),
            'rawPath': img_path_a,
            'match': os.path.basename(most_similar_img_path),
            'matchPath': most_similar_img_path,
            'matchRatio': final_ratio
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


def find_good_matches(
        ref_gray: np.ndarray,
        img_gray: np.ndarray,
        min_good_matches: int = 100,
        ratio_thresh: float = 0.75,
        print_log: bool = False):
    """
    检测并筛选足够的优质匹配点

    返回:
        Tuple[
            bool: 是否找到足够好的匹配点,
            List[cv2.DMatch] | None: 优质匹配点列表（成功时有值）,
            src_pts: 来自img的匹配点坐标（成功时有值）,
            dst_pts: 来自ref的匹配点坐标（成功时有值）
        ]
    """
    if ref_gray is None or ref_gray.size == 0:
        if print_log:
            print("参考图为空，无法提取特征")
        return False, None, None, None

    if img_gray is None or img_gray.size == 0:
        if print_log:
            print("待匹配图为空，无法提取特征")
        return False, None, None, None

    # AKAZE 特征检测
    detector = cv2.AKAZE_create()
    kp_ref, des_ref = detector.detectAndCompute(ref_gray, None)
    kp_img, des_img = detector.detectAndCompute(img_gray, None)

    if des_ref is None or des_ref.shape[0] == 0:
        if print_log:
            print("参考图没有任何特征点")
        return False, None, None, None

    if des_img is None or des_img.shape[0] == 0:
        if print_log:
            print("目标图没有任何特征点")
        return False, None, None, None

    if len(kp_ref) < 30 or len(kp_img) < 30:
        if print_log:
            print(f"特征点太少：ref={len(kp_ref)}, img={len(kp_img)}")
        return False, None, None, None

    # ──────────────── 改用 FLANN 匹配器 ────────────────
    # LSH 算法的 index 类型数值就是 6（FLANN_INDEX_LSH = 6）
    index_params = dict(
        algorithm=6,  # ← 这里直接写 6 代替 cv2.FLANN_INDEX_LSH
        table_number=12,  # 推荐 6~12
        key_size=20,  # 推荐 10~20
        multi_probe_level=2  # 推荐 1~2
    )
    search_params = dict(checks=50)  # 50~100，越大越准但越慢

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if print_log:
        print(f"参考图描述子数量: {des_ref.shape[0]}")
        print(f"目标图描述子数量: {des_img.shape[0]}")
        print("开始 FLANN knnMatch...")

    try:
        matches = flann.knnMatch(des_img, des_ref, k=2)
    except cv2.error as e:
        if print_log:
            print(f"FLANN knnMatch 失败: {e}")
        return False, None, None, None

    for match in matches:
        if len(match) != 2:
            if print_log:
                print(f"{ref_gray} 和 {img_gray} 匹配项长度不足")
            return False, None, None, None

    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if print_log:
        print(f"有效匹配点数：{len(good)}")

    if len(good) < min_good_matches:
        if print_log:
            print(f"匹配点不足（{len(good)} < {min_good_matches}）")
        return False, None, None, None

    # 提取坐标
    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good])  # img
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good])  # ref

    return True, good, src_pts, dst_pts


def align_images(
        ref_path: str,
        img_path: str,
        output_dir: str = "temp_align",
        min_good_matches: int = 100,
        fill_color: tuple = (255, 255, 255),
        print_log: bool = False):
    """
    将 img_path 的图片对齐到 ref_path 的图片空间，并保存对齐后的结果。
    """
    ref = cv2_imread_unicode(ref_path)
    img = cv2_imread_unicode(img_path)

    if ref is None or img is None:
        if print_log:
            print(f"读取图片失败：ref={ref_path}, img={img_path}")
        return None

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 抽取出来的匹配判断 ────────────────────────────────
    success, good_matches, src_pts, dst_pts = find_good_matches(
        ref_gray, img_gray,
        min_good_matches=min_good_matches,
        print_log=print_log
    )

    if not success:
        if print_log:
            print(f"匹配失败：ref={ref_path}, img={img_path}")
        return None

    # 计算单应矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    inlier_ratio = mask.sum() / len(good_matches) if len(good_matches) > 0 else 0
    if print_log:
        print(f"内点比例：{inlier_ratio:.2%}")

    if inlier_ratio < 0.5:
        if print_log:
            print(f"内点比例过低，对齐不可靠")
        return None

    # 透视变换
    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(
        img, H, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_color
    )

    # 保存
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_aligned.png")

    success = cv2.imwrite(output_path, aligned)
    if not success:
        if print_log:
            print(f"保存失败：{output_path}")
        return None

    if print_log:
        print(f"对齐完成：{output_path}")
    return os.path.abspath(os.path.normpath(output_path))


def align_image_color(source_path, reference_path, output_dir="temp_align", ):
    """
    将 source 图像的色彩/色调对齐到 reference 图像。

    :param source_path: 待转换的图像路径
    :param reference_path: 参考的标准图像路径
    :param output_dir:
    """
    # 1. 加载图像
    # 对于黑白漫画，建议直接以灰度模式加载，效果最稳定
    src_img = cv2_imread_unicode(source_path, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2_imread_unicode(reference_path, cv2.IMREAD_GRAYSCALE)

    if src_img is None or ref_img is None:
        print("错误：请检查路径，无法读取图片。")
        return

    # 2. 直方图匹配 (Histogram Matching)
    # 这步是核心：它会将 src 的像素分布映射得跟 ref 一模一样
    matched = exposure.match_histograms(src_img, ref_img)

    # 3. 转换为 8-bit 无符号整型（防止 skimage 输出 float）
    matched = matched.astype(np.uint8)

    # 4. (可选) 细节微调：如果需要极致的蒙版效果，可以确保背景是纯白
    # 比如将 250 以上的像素全部强制转为 255
    # _, matched = cv2.threshold(matched, 250, 255, cv2.THRESH_TRUNC)
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(source_path))[0] + '_colored.png')
    # 5. 保存结果
    cv2.imwrite(output_path, matched)
    return output_path


def align_image_color_v2(source_path, reference_path, output_dir="temp_align"):
    """
    兼容彩色和黑白漫画的色彩/亮度对齐函数
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 加载原图 (不指定灰度，保留原始通道)
    src_img = cv2_imread_unicode(source_path)
    ref_img = cv2_imread_unicode(reference_path)

    if src_img is None or ref_img is None:
        print("错误：无法读取图片。")
        return None

    # 2. 判断是否需要彩色处理
    # 如果图片本身是 3 通道的且不是纯灰度，则进入彩色模式
    is_gray = isGrayMap(Image.open(source_path))

    if not is_gray:
        # --- 彩色模式：使用 LAB 空间防止色偏 ---
        # 将 BGR 转换为 LAB
        src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)
        ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)

        # 对三个通道分别进行直方图匹配
        # L: 亮度, A: 绿-红, B: 蓝-黄
        matched_lab = np.zeros_like(src_lab)
        for i in range(3):
            matched_lab[:, :, i] = exposure.match_histograms(
                src_lab[:, :, i], ref_lab[:, :, i]
            )

        # 转回 BGR
        matched = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    else:
        # --- 黑白模式：直接匹配 ---
        # 确保转为单通道处理
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        matched = exposure.match_histograms(src_gray, ref_gray)

    # 3. 数据类型转换
    matched = np.clip(matched, 0, 255).astype(np.uint8)

    # 4. 生成输出路径
    file_name = os.path.splitext(os.path.basename(source_path))[0] + '_colored.png'
    output_path = os.path.join(output_dir, file_name)

    # 5. 保存结果
    cv2.imwrite(output_path, matched)
    return os.path.abspath(os.path.normpath(output_path))


def psd_to_png(psd_path, out__dir='PNG', compression=9):
    """
    处理PSD文件，根据图层数量导出PNG
    :param compression:
    :param out__dir:
    :param psd_path: PSD文件的完整路径（支持字符串或Path对象）
    """
    psd_path = Path(psd_path)
    output_png = (Path(psd_path).parent / out__dir / psd_path.with_suffix(".png").name).absolute()

    def get_min_compress_png_options():
        png_options = PNGSaveOptions()
        png_options.compression = compression
        png_options.interlaced = False
        png_options.optimizedColorPalette = False
        return png_options

    if not psd_path.exists():
        print(f"文件不存在：{psd_path}")
        return

    with Session() as ps:
        # 手动设置PS界面不可见（兼容所有版本）
        ps.app.visible = False
        doc = ps.app.open(str(psd_path))
        doc.saveAs(str(output_png), get_min_compress_png_options(), True)
        doc.close(SaveOptions.DoNotSaveChanges)


def psd_to_jpg(psd_path, out_dir='PNG', quality=12):
    """
    处理PSD文件，导出JPG格式
    :param psd_path: PSD文件的完整路径（支持字符串或Path对象）
    :param out_dir: 输出文件夹名称，默认 JPG
    :param quality: JPG质量 0-12，默认12（最高质量）
    """
    psd_path = Path(psd_path)
    output_jpg = (Path(psd_path).parent / out_dir / psd_path.with_suffix(".jpg").name).absolute()

    def get_jpg_save_options():
        jpg_options = JPEGSaveOptions()
        jpg_options.quality = quality
        jpg_options.embedColorProfile = True
        jpg_options.formatOptions = 1
        jpg_options.scans = 3
        jpg_options.matte = 1
        return jpg_options

    if not psd_path.exists():
        print(f"文件不存在：{psd_path}")
        return
    output_jpg.parent.mkdir(exist_ok=True)
    with Session() as ps:
        ps.app.visible = False
        doc = ps.app.open(str(psd_path))
        # 保存 JPG
        doc.saveAs(str(output_jpg), get_jpg_save_options(), True)
        doc.close(SaveOptions.DoNotSaveChanges)


if __name__ == "__main__":
    # # 测试：检测是否黑白图
    # test_img_gray = Image.open(r"F:\CH1 Visiting Home (COMIC X-Eros #52) (02).png")
    # print(isGrayMap(test_img_gray, debug=True))

    # 测试：使图片B向图片A对齐
    aligned_path_white = align_images(
        ref_path=r"F:\JHenTai_data\待翻新\[陰謀の帝国 (印度カリー)] 婚約者の妹は顔SSR、性格最悪地獄のエロダンス女。\CN\43_C106_044.jpg",
        img_path=r"F:\JHenTai_data\待翻新\[陰謀の帝国 (印度カリー)] 婚約者の妹は顔SSR、性格最悪地獄のエロダンス女。\043.jpg", )
    # aligned_path_white = align_images(ref_path=r"F:\JHenTai_data\待翻新\[陰謀の帝国 (印度カリー)] 婚約者の妹は顔SSR、性格最悪地獄のエロダンス女。\044.jpg",
    #                                   img_path=r"F:\JHenTai_data\待翻新\[陰謀の帝国 (印度カリー)] 婚約者の妹は顔SSR、性格最悪地獄のエロダンス女。\CN\44_C106_045.jpg")
    if aligned_path_white:
        print(f"成功生成：{aligned_path_white}")
    else:
        print("对齐失败")

    # # 测试：使图片B向图片A对齐，且对齐颜色
    # aligned_path_white = align_images(ref_path=r"F:\JHenTai_data\[いーむす・アキ] きもちいーむすめ\TEST\E022.png",
    #                                   img_path=r"F:\JHenTai_data\[いーむす・アキ] きもちいーむすめ\TEST\020_MJK_18_D1350_017.png")
    # if aligned_path_white:
    #     print(f"成功生成：{aligned_path_white}")
    #     aligned_color_img = align_image_color_v2(source_path=aligned_path_white,
    #                                              reference_path=r"F:\JHenTai_data\[いーむす・アキ] きもちいーむすめ\TEST\E022.png")
    #     print(aligned_color_img)

    # # 测试：使图片B向图片A对齐，且对齐颜色
    # is_good, _, _, _ = find_good_matches(
    #     ref_gray=cv2_imread_unicode(r"F:\JHenTai_data\待翻新\[きづかかずき] エロ漫研とかにようこそ! [DL版]\日文\082.jpg", cv2.COLOR_BGR2GRAY),
    #     img_gray=cv2_imread_unicode(r"F:\JHenTai_data\待翻新\[きづかかずき] エロ漫研とかにようこそ! [DL版]\[きづかかずき] エロ漫研とかにようこそ\084.jpg", cv2.COLOR_BGR2GRAY),
    #     min_good_matches=100,
    #     print_log=True
    # )
