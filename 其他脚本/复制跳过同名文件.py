import os
import shutil


def get_folder_path(prompt_info: str) -> str:
    """获取用户输入的文件夹路径，做基础合法性校验"""
    while True:
        folder_path = input(prompt_info).strip()
        # 路径为空则重新输入
        if not folder_path:
            print("❌ 路径不能为空，请重新输入！")
            continue
        # 路径存在且是文件夹则返回，否则提示
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            return folder_path
        else:
            print(f"❌ 路径【{folder_path}】不存在或不是文件夹，请重新输入！")


def copy_files_by_name_prefix(src_dir: str, dst_dir: str):
    """
    将源文件夹A的文件复制到目标文件夹B
    规则：文件名（不含扩展名）重复则跳过，自动创建目标文件夹
    """
    # 1. 自动创建目标文件夹（不存在则创建，存在则无操作）
    os.makedirs(dst_dir, exist_ok=True)

    # 2. 提取目标文件夹B中所有文件的【无扩展名前缀】，存入集合（查询效率O(1)）
    dst_file_prefixes = set()
    for item in os.listdir(dst_dir):
        item_full_path = os.path.join(dst_dir, item)
        # 仅处理文件，跳过子文件夹
        if os.path.isfile(item_full_path):
            # 分离文件名和扩展名，取前缀
            file_name, _ = os.path.splitext(item)
            dst_file_prefixes.add(file_name)

    # 3. 遍历源文件夹A，执行复制逻辑
    copied_count = 0  # 统计成功复制的文件数
    skipped_count = 0  # 统计跳过的文件数
    for file_name in os.listdir(src_dir):
        src_file_path = os.path.join(src_dir, file_name)
        # 跳过源文件夹中的子文件夹，仅处理文件
        if not os.path.isfile(src_file_path):
            continue

        # 分离源文件的【前缀名】和【扩展名】
        file_prefix, file_ext = os.path.splitext(file_name)

        # 核心判定：前缀名已存在则跳过
        if file_prefix in dst_file_prefixes:
            # print(f"⏭️  跳过【{file_name}】→ 前缀名「{file_prefix}」已存在于目标文件夹")
            skipped_count += 1
            continue

        # 前缀名不存在，执行复制操作
        dst_file_path = os.path.join(dst_dir, file_name)
        try:
            shutil.copy2(src_file_path, dst_file_path)
            print(f"✅ 成功复制【{file_name}】→ {dst_file_path}")
            copied_count += 1
        except Exception as e:
            print(f"❌ 复制【{file_name}】失败：{str(e)}")

    # 4. 输出最终统计结果
    print("-" * 50)
    print(f"📊 复制完成 | 成功：{copied_count}个 | 跳过：{skipped_count}个")


if __name__ == "__main__":
    print("===== 文件复制工具（同名前缀跳过版）=====\n")
    # 获取源文件夹A、目标文件夹B路径
    src_folder = get_folder_path("请输入【源文件夹A】的完整路径：")
    dst_folder = get_folder_path("请输入【目标文件夹B】的完整路径：")
    print("\n开始执行复制操作...\n")
    # 执行核心复制逻辑
    copy_files_by_name_prefix(src_folder, dst_folder)