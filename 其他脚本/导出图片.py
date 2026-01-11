import os
from pathlib import Path
from photoshop import Session
from photoshop.api import SaveOptions, PNGSaveOptions, SolidColor

psd_file_path = Path(r"../dataset/psd_with_mask")
out__dir = '../dataset/data'
out_png_dir = 'images'
out_mask_dir = 'masks'


def process_psd(psd_path, out_dir, mask_dir):
    """
    处理PSD文件，根据图层数量导出原图和mask图（纯黑填充）
    :param psd_path: PSD文件的完整路径（支持字符串或Path对象）
    :param out_dir: 输出图片的保存路径
    :param mask_dir: 输出mask的保存路径
    """
    psd_path = Path(psd_path)
    # 生成输出文件路径
    output_png = (Path(out_dir) / psd_path.with_suffix(".png").name).absolute()
    output_mask_png = (Path(mask_dir) / psd_path.with_name(f"{psd_path.stem}_mask.png").name).absolute()

    # 配置PNG最小压缩选项（关键：压缩级别设为9，关闭透明度优化等）
    def get_min_compress_png_options():
        png_options = PNGSaveOptions()
        png_options.compression = 9
        png_options.interlaced = False
        png_options.optimizedColorPalette = False
        return png_options

    # 图层填充黑色，确保图层未锁定
    def fill_layer_black(layer):
        doc.activeLayer = layer
        doc.selection.selectAll()
        black_color = SolidColor()
        black_color.rgb.red = 0
        black_color.rgb.green = 0
        black_color.rgb.blue = 0
        doc.selection.fill(black_color)
        doc.selection.deselect()

    if not psd_path.exists():
        print(f"文件不存在：{psd_path}")
        return

    with Session() as ps:
        # 手动设置PS界面不可见（兼容所有版本）
        ps.app.visible = False
        doc = ps.app.open(str(psd_path))
        layer_count = doc.layers.count
        if doc.mode == 6:
            doc.changeMode(1)  # 1是灰度，2是RGB，3是CMYK，6是索引
        if layer_count > 2:
            print(f"只支持1~2个图层的PSD：{psd_path.name}")
            doc.close(SaveOptions.DoNotSaveChanges)
            return

        if layer_count == 2:
            # 导出底图
            doc.layers[0].visible = False
            doc.saveAs(str(output_png), get_min_compress_png_options(), True)
            doc.layers[0].visible = True
            # 导出mask
            doc.layers[-1].isBackgroundLayer = False
            fill_layer_black(doc.layers[-1])
            doc.layers[-1].isBackgroundLayer = True
            doc.saveAs(str(output_mask_png), get_min_compress_png_options(), True)
        else:
            doc.saveAs(str(output_png), get_min_compress_png_options(), True)
            doc.layers[-1].isBackgroundLayer = False
            fill_layer_black(doc.layers[-1])
            doc.layers[-1].isBackgroundLayer = True
            doc.saveAs(str(output_mask_png), get_min_compress_png_options(), True)
        # 关闭psd
        doc.close(SaveOptions.DoNotSaveChanges)


# 测试调用示例
if __name__ == "__main__":
    out_png_dir = os.path.join(out__dir, out_png_dir)
    out_mask_dir = os.path.join(out__dir, out_mask_dir)
    if not os.path.exists(out_png_dir):
        os.makedirs(out_png_dir, exist_ok=True)
    if not os.path.exists(out_mask_dir):
        os.makedirs(out_mask_dir, exist_ok=True)
    png_list = os.listdir(out_png_dir)
    for psd in Path(psd_file_path).rglob('*.PSD'):
        if psd.with_suffix('.png').name in png_list:
            continue
        process_psd(psd.absolute(), out_png_dir, out_mask_dir)
