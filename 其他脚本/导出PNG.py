import os
from pathlib import Path
from photoshop import Session
from photoshop.api import SaveOptions, PNGSaveOptions, SolidColor

psd_file_path = Path(input("请输入文件夹路径:\n"))
out__dir = 'PNG'


def process_psd(psd_path):
    """
    处理PSD文件，根据图层数量导出原图和mask图（纯黑填充）
    :param psd_path: PSD文件的完整路径（支持字符串或Path对象）
    """
    psd_path = Path(psd_path)
    # 生成输出文件路径
    output_png = (Path(psd_path).parent / out__dir / psd_path.with_suffix(".png").name).absolute()

    # 配置PNG最小压缩选项（关键：压缩级别设为9，关闭透明度优化等）
    def get_min_compress_png_options():
        png_options = PNGSaveOptions()
        png_options.compression = 9
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


# 测试调用示例
if __name__ == "__main__":
    if not os.path.exists(Path(psd_file_path) / out__dir):
        os.makedirs(Path(psd_file_path) / out__dir, exist_ok=True)
    for psd in Path(psd_file_path).rglob('*.PSD'):
        process_psd(psd.absolute())
