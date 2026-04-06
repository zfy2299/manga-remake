import io
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import torch
from PIL import Image
from flask import Flask, send_from_directory, send_file, request
import pythoncom
from photoshop import Session

from ResNetUNet import ResNetUNet
from utils import infer_single_image, ps_auto_composite_layers, match_comics_2, split_image, ps_auto_white, psd_to_jpg, \
    psd_to_png

# 关键配置：static_folder 指定静态文件根目录为 web
# Flask 自动将 / 映射为 web/index.html，且支持访问目录内所有文件

app = Flask(__name__, static_folder='web/asset')

executor = ThreadPoolExecutor(max_workers=1)
task_ids = []
image_extensions = [".jpg", ".jpeg", ".webp", ".png", ".bmp", '.avif']


@app.route('/')
def index():
    # send_from_directory：从指定目录返回指定文件
    # 参数1：静态文件所在目录（web）；参数2：要返回的文件名（index.html）
    return send_from_directory('web', 'index.html')


@app.route('/images/<path:max_size>/<path:filepath>')
def serve_compress_img(filepath, max_size):
    filepath = unquote(filepath)
    filepath = filepath.replace('/', '\\')
    if not os.path.exists(filepath):
        print(filepath)
        return "图片不存在", 404
    max_temp = max_size.split('x')
    with Image.open(filepath) as img:
        if img.mode == 'P':
            img = img.convert('RGBA', dither=None)
        img.thumbnail((int(max_temp[0]), int(max_temp[1])), Image.Resampling.LANCZOS)
        img_byte = io.BytesIO()
        img.save(img_byte, format='webp', quality=75, optimize=True)
        img_byte.seek(0)
    return send_file(img_byte, mimetype='image/webp')


@app.route('/api/img_match', methods=['POST'])
def img_match():
    data = request.get_json(force=True)
    match_from_dir = data.get("match_from_dir", '未知路径')
    match_to_dir = data.get("match_to_dir", '未知路径')
    match_from_son = data.get("match_from_son", False)
    match_twice = data.get("match_twice", False)
    match_twice_point = data.get("match_twice_point", 100)
    match_twice_start = data.get("match_twice_start", 0.5)
    split_wide = data.get("split_wide", False)
    if not os.path.exists(match_to_dir) or not os.path.exists(match_from_dir):
        return {
            'code': 400,
            'msg': "路径不存在"
        }
    if split_wide:
        split_image(match_from_dir, match_from_son)
    # final_result = match_comics(
    #     match_to_dir, match_from_dir,
    #     similarity_threshold=similar_threshold,
    #     debug_mode=False,
    #     max_workers=5,
    #     use_align=True
    # )
    final_result = match_comics_2(
        match_to_dir, match_from_dir, match_from_son=match_from_son, match_twice=match_twice,
        match_twice_point=match_twice_point, match_twice_start=match_twice_start
    )
    r_ = '**/*' if match_from_son else '*'
    return {
        'code': 200,
        'data': {
            'match_result': final_result['match_result'],
            'match_from_num': final_result['a_num'],
            'match_dir_num': final_result['b_num'],
            'match_from_list': [
                {
                    "name": img_path.name,
                    "path": str(img_path)
                }
                for img_path in Path(match_from_dir).glob(r_)
                if img_path.is_file() and img_path.suffix.lower() in image_extensions
            ]
        }
    }


@app.route('/api/img_match_name', methods=['POST'])
def img_match_name():
    def natural_sort_key(filename):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', filename)]

    def get_image_files(dir_path):
        files = []
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            # 只保留文件 + 后缀在允许列表内
            if os.path.isfile(file_path) and os.path.splitext(f)[1].lower() in image_extensions:
                files.append(f)
        return files

    data = request.get_json(force=True)
    match_from_dir = data.get("match_from_dir", '未知路径')
    match_to_dir = data.get("match_to_dir", '未知路径')
    if not os.path.exists(match_to_dir) or not os.path.exists(match_from_dir):
        return {
            'code': 400,
            'msg': "路径不存在"
        }
    from_files = get_image_files(match_from_dir)
    to_files = get_image_files(match_to_dir)
    # 2. 自然排序（按文件名中的数字排序）
    from_files.sort(key=natural_sort_key)
    to_files.sort(key=natural_sort_key)
    # 3. 按顺序一一配对生成结果
    result = []
    min_len = min(len(from_files), len(to_files))
    for i in range(min_len):
        match_name = from_files[i]
        raw_name = to_files[i]
        raw_path = os.path.join(match_to_dir, raw_name).replace('\\', '/')
        match_path = os.path.join(match_from_dir, match_name).replace('\\', '/')
        result.append({
            "raw": raw_name,
            "rawPath": raw_path,
            "match": match_name,
            "matchPath": match_path,
            "matchRatio": 1
        })
    return {
        'code': 200,
        'data': {
            'match_result': result,
            'match_from_list': [
                {
                    "name": img_name,
                    "path": os.path.join(match_from_dir, img_name)
                }
                for img_name in from_files
            ]
        }
    }


def start_ps_task(param, task_id):
    try:
        task_list = param['match_list']
        if not task_list:
            print('任务队列为空')
            return
        config = param['config']
        if 'no_mask' in param:
            no_mask = param['no_mask']
        else:
            no_mask = False
        models = os.listdir('weight')
        if len(models) == 0:
            print(f"weight目录下未找到模型文件")
            return
        use_model = os.path.join('weight', models[-1])
        if config['maskUseCPU']:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetUNet(n_channels=3, n_classes=1, local_model=True).to(device)
        model.load_state_dict(torch.load(use_model, weights_only=True, map_location=device))
        model.eval()
        temp_mask_dir = 'temp_mask'
        if config['maskTempDir']:
            temp_mask_dir = config['maskTempDir']
        os.makedirs(temp_mask_dir, exist_ok=True)
        if config['colorLv']:
            color_level = {'black': config['colorLvBlack'], 'white': config['colorLvWhite'],
                           'gray': config['colorLvGray']}
        else:
            color_level = None
        if config['blurFilter']:
            filter_blur = {'radius': config['blurFilterRadius'], 'threshold': config['blurFilterThreshold']}
        else:
            filter_blur = None
        if config['USMFilter']:
            filter_sharp = {'quantity': config['USMFilterQuantity'], 'radius': config['USMFilterRadius'],
                            'threshold': config['USMFilterThreshold']}
        else:
            filter_sharp = None
        if config['UseAction']:
            do_action = [config['UseActionGroup'], config['UseActionName']]
        else:
            do_action = None
        cv2_align = False
        color_align = False
        if config['cv2Align']:
            cv2_align = config['cv2Align']
        if config['autoColor']:
            color_align = config['autoColor']
        if config['selectionOperate']:
            selection_contract = config['selectionContract']
            selection_feather = config['selectionFeather']
        else:
            selection_contract = 0
            selection_feather = 0
        print(f'启动队列：{task_id}')
        pythoncom.CoInitialize()
        for item in task_list:
            bottom_img_path = item['rawPath']
            up_img_path = item['matchPath']
            if not os.path.exists(bottom_img_path):
                print(f"路径不存在：{bottom_img_path}")
            if not os.path.exists(up_img_path):
                print(f"路径不存在：{up_img_path}")
            save_psd_path = os.path.join(os.path.dirname(bottom_img_path), 'auto_PSD', item['raw'])
            os.makedirs(os.path.dirname(save_psd_path), exist_ok=True)
            if no_mask:
                mask_path = None
            else:
                mask_path = infer_single_image(bottom_img_path, model, save_dir=temp_mask_dir, device=device)
            ps_auto_composite_layers(bottom_img_path, up_img_path, mask_path,
                                     color_level=color_level,
                                     filter_blur=filter_blur,
                                     cv2_align=cv2_align,
                                     color_align=color_align,
                                     filter_sharp=filter_sharp,
                                     selection_contract=selection_contract,
                                     selection_feather=selection_feather,
                                     do_action=do_action, filter_only_selection=config['filterOnlySelection'],
                                     auto_gray=config['autoGray'], save_psd_path=save_psd_path)
            # print(f"PSD已保存：{save_psd_path}")
        pythoncom.CoUninitialize()
        print(f"队列已完成...")
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        task_ids.remove(task_id)


def start_ps_white_task(param, task_id):
    try:
        config = param['config']
        task_path = config['match_to_dir']
        models = os.listdir('weight')
        if len(models) == 0:
            print(f"weight目录下未找到模型文件")
            return
        use_model = os.path.join('weight', models[-1])
        if config['maskUseCPU']:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetUNet(n_channels=3, n_classes=1, local_model=True).to(device)
        model.load_state_dict(torch.load(use_model, weights_only=True, map_location=device))
        model.eval()
        temp_mask_dir = 'temp_mask'
        if config['maskTempDir']:
            temp_mask_dir = config['maskTempDir']
        os.makedirs(temp_mask_dir, exist_ok=True)
        if config['colorLv']:
            color_level = {'black': config['colorLvBlack'], 'white': config['colorLvWhite'],
                           'gray': config['colorLvGray']}
        else:
            color_level = None
        if config['blurFilter']:
            filter_blur = {'radius': config['blurFilterRadius'], 'threshold': config['blurFilterThreshold']}
        else:
            filter_blur = None
        if config['USMFilter']:
            filter_sharp = {'quantity': config['USMFilterQuantity'], 'radius': config['USMFilterRadius'],
                            'threshold': config['USMFilterThreshold']}
        else:
            filter_sharp = None
        if config['UseAction']:
            do_action = [config['UseActionGroup'], config['UseActionName']]
        else:
            do_action = None
        if config['selectionOperate']:
            selection_contract = config['selectionContract']
            selection_feather = config['selectionFeather']
        else:
            selection_contract = 0
            selection_feather = 0
        print(f'启动队列：{task_id}')
        pythoncom.CoInitialize()
        images_ = [
            p for p in Path(task_path).glob('*')
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'] and p.is_file()
        ]
        for item in images_:
            if not os.path.exists(item):
                print(f"路径不存在：{item}")
            save_psd_path = os.path.join(task_path, 'PSD', item.with_suffix('.psd').name)
            os.makedirs(os.path.dirname(save_psd_path), exist_ok=True)
            item_path = str(item.absolute())
            mask_path = infer_single_image(item_path, model, save_dir=temp_mask_dir, device=device)
            ps_auto_white(item_path, mask_path, color_level=color_level,
                          filter_blur=filter_blur,
                          filter_sharp=filter_sharp,
                          selection_contract=selection_contract,
                          selection_feather=selection_feather,
                          do_action=do_action, white_opacity=config['white_opacity'],
                          auto_gray=config['autoGray'], save_psd_path=save_psd_path)
        pythoncom.CoUninitialize()
        print(f"队列已完成...")
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        task_ids.remove(task_id)


def start_ps_output_task(param, task_id):
    try:
        config = param['config']
        task_path = config['psd_dir']
        out__dir = config['psd_out_dir']
        if not os.path.exists(Path(task_path) / out__dir):
            os.makedirs(Path(task_path) / out__dir, exist_ok=True)
        pythoncom.CoInitialize()
        print(f'启动队列：{task_id}')
        for psd in Path(task_path).rglob('*.PSD'):
            psd_path = psd
            if config['color_jpg']:
                with Session() as ps:
                    ps.app.visible = False
                    doc = ps.app.open(str(psd_path))
                    layer_count = len(doc.channels)
                    if layer_count > 1:
                        psd_to_jpg(psd_path, out__dir, config['jpg_quality'])
                    else:
                        psd_to_png(psd_path, out__dir, config['png_compression'])
            else:
                psd_to_png(psd_path, out__dir, config['png_compression'])
        pythoncom.CoUninitialize()
        print(f"队列已完成...")
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        task_ids.remove(task_id)


@app.route('/api/start_ps', methods=['POST'])
def start_ps():
    data = request.get_json(force=True)
    p1 = os.path.dirname(data['match_list'][0]['matchPath'])
    p2 = os.path.dirname(data['match_list'][0]['rawPath'])
    task_id = f"{p1}:{p2}"
    if task_id in task_ids:
        return {
            'code': 400,
            'msg': '此任务正在执行！'
        }
    task_ids.append(task_id)
    executor.submit(start_ps_task, data, task_id)
    return {
        'code': 200
    }


@app.route('/api/start_ps_white_only', methods=['POST'])
def start_ps_white_only():
    data = request.get_json(force=True)
    p = data['config']['match_to_dir']
    if not os.path.exists(p):
        return {
            'code': 400,
            'msg': '输入路径不存在!'
        }
    task_id = p
    if task_id in task_ids:
        return {
            'code': 400,
            'msg': '此任务正在执行！'
        }
    task_ids.append(task_id)
    executor.submit(start_ps_white_task, data, task_id)
    return {
        'code': 200
    }


@app.route('/api/start_ps_output', methods=['POST'])
def start_ps_output():
    data = request.get_json(force=True)
    p = data['config']['psd_dir']
    if not os.path.exists(p):
        return {
            'code': 400,
            'msg': '输入路径不存在!'
        }
    task_id = p
    if task_id in task_ids:
        return {
            'code': 400,
            'msg': '此任务正在执行！'
        }
    task_ids.append(task_id)
    executor.submit(start_ps_output_task, data, task_id)
    return {
        'code': 200
    }


@app.route('/api/start_rename', methods=['POST'])
def start_rename():
    data = request.get_json(force=True)
    rename_copy = data['config']['rename_copy']
    match_list = data['match_list']
    raw_path = os.path.dirname(data['match_list'][0]['rawPath'])
    save_path = os.path.join(raw_path, os.path.basename(raw_path))
    os.makedirs(save_path, exist_ok=True)
    raw_list = os.listdir(raw_path)
    over_list = []
    if data['config']['remain_ext']:
        for item in match_list:
            source_path = item['matchPath']
            raw_name = item['raw']
            ext = os.path.splitext(source_path)[1]
            base_name = os.path.splitext(raw_name)[0]
            new_filename = base_name + ext
            dest_path = os.path.join(save_path, new_filename)
            shutil.copy2(source_path, dest_path)
            over_list.append(item['raw'])
    else:
        for item in match_list:
            shutil.copy2(item['matchPath'], os.path.join(save_path, item['raw']))
            over_list.append(item['raw'])
    if rename_copy:
        for item in raw_list:
            img_path = Path(os.path.join(raw_path, item))
            if item in over_list or img_path.is_dir() or img_path.suffix.lower() not in image_extensions:
                continue
            shutil.copy2(img_path, os.path.join(save_path, item))
    return {
        'code': 200
    }


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
