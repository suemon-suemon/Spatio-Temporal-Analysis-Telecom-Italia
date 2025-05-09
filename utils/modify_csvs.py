import os
import pandas as pd
import shutil

# 原数据文件所在目录
data_dir = '/data/scratch/jiayin'
# 新数据存储目录
new_data_dir = '/data/scratch/jiayin/milan'

# 创建新目录（如果不存在的话）
os.makedirs(new_data_dir, exist_ok=True)

# 获取所有csv文件
file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# 遍历每个文件
for file_name in file_list:
    # 读取CSV文件
    path = os.path.join(data_dir, file_name)
    try:
        data = pd.read_csv(path, header=0, index_col=0)

        # 检查是否有 'time' 列
        if 'time' not in data.columns:
            print(f"警告: 文件 {file_name} 没有 'time' 列，跳过该文件。")
            continue  # 如果没有 'time' 列，跳过当前文件

        # 转换为 datetime 类型，并处理无效时间
        data['time'] = pd.to_datetime(data['time'], unit='ms', errors='coerce')  # 'coerce' 会将无效的日期转换为 NaT

        # 检查是否存在有效的时间戳
        if data['time'].isna().all():
            print(f"警告: 文件 {file_name} 中所有时间戳无效。跳过该文件。")
            continue  # 如果所有时间戳无效，跳过当前文件

        # 获取文件中最晚的时间（如果有 NaT，会忽略 NaT）
        latest_time = data['time'].dropna().max()  # 使用 dropna() 去除 NaT 值

        # 生成新的文件名：根据最晚时间的日期来命名
        new_file_name = f'sms-call-internet-mi-{latest_time.strftime("%Y-%m-%d")}.csv'
        new_file_path = os.path.join(new_data_dir, new_file_name)

        # 复制文件到新目录，并重命名
        shutil.copy(path, new_file_path)

        print(f"文件 {file_name} 已处理，重命名为 {new_file_name}")

    except Exception as e:
        print(f"读取文件 {file_name} 时发生错误: {e}，跳过该文件。")
        continue  # 如果遇到任何其他错误，跳过当前文件

##
