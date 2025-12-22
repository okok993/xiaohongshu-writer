# fix_imports.py - 自动修复导入问题
import os
import shutil


def fix_project_structure():
    print("🔧 修复项目结构...")

    # 1. 创建必要的文件夹
    for folder in ['llm', 'prompts']:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ 创建文件夹: {folder}")

        # 创建 __init__.py
        init_file = os.path.join(folder, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Python package\n')
            print(f"✅ 创建: {folder}/__init__.py")

    # 2. 移动文件
    files_to_fix = [
        ('llm.client.py', 'llm/client.py'),
        ('xiaohongshu_template.py', 'prompts/xiaohongshu_template.py')
    ]

    for old_name, new_name in files_to_fix:
        if os.path.exists(old_name):
            if os.path.exists(new_name):
                print(f"⚠️  文件已存在: {new_name}")
            else:
                shutil.move(old_name, new_name)
                print(f"✅ 移动: {old_name} -> {new_name}")
        elif os.path.exists(new_name):
            print(f"✅ 文件已在正确位置: {new_name}")
        else:
            print(f"❌ 找不到文件: {old_name}")

    # 3. 检查app.py是否需要修改
    app_file = 'app.py'
    if os.path.exists(app_file):
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查导入语句
        if 'from llm.client import' in content and 'from prompts.xiaohongshu_template import' in content:
            print("✅ app.py导入语句正确")
        else:
            print("⚠️  app.py可能需要更新导入语句")

    print("\n🎯 修复完成！")
    print("\n当前项目结构:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            if not file.startswith('.') and file not in ['__pycache__']:
                print(f'{subindent}{file}')


if __name__ == '__main__':
    fix_project_structure()