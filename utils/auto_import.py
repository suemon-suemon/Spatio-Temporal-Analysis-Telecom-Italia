import importlib
import os

def auto_import_modules_from(directory, package_prefix):
    """
    自动从某个目录导入所有 Python 模块，触发注册器。
    Args:
        directory (str): 文件夹路径，如 './models' 或 './datasets'
        package_prefix (str): 包前缀，如 'models' 或 'datasets'
    """
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py":

            if "Mstgcn" in filename: continue  # 跳过会导入 mxnet 的模型

            module_name = filename[:-3]  # 去掉 .py 后缀
            full_module_name = f"{package_prefix}.{module_name}"
            importlib.import_module(full_module_name)