import json
import h5py
import numpy as np
from shapely.geometry import Polygon
import pandas as pd

# 读取GeoJSON文件
def load_geojson(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# 计算每个cell的矩形中心点
def calculate_cell_centroids(features):
    centroids = []
    cell_vertices = []

    for feature in features:
        # 获取每个cell的多边形顶点
        coordinates = feature['geometry']['coordinates'][0]  # 取出每个Polygon的坐标
        cell_vertices.append(np.array(coordinates))

        # 创建Polygon对象，计算中心点
        polygon = Polygon(coordinates)
        centroid = polygon.centroid.coords[0]
        centroids.append(np.array(centroid))

    return np.array(cell_vertices), np.array(centroids)


# 将数据保存到HDF5文件中
def save_to_hdf5(cell_vertices, centroids, output_file):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('cell_vertices', data=cell_vertices)
        f.create_dataset('centroids', data=centroids)
        print(f"Data saved to {output_file}")


def main():
    # 设置输入输出路径
    geojson_file = '/data/scratch/jiayin/milan/milano-grid.geojson'  # 你可以根据需要修改路径
    output_file = '/data/scratch/jiayin/milan/milano-grid-coordinates.h5'  # 目标HDF5文件

    # 读取GeoJSON文件
    data = load_geojson(geojson_file)

    # 提取每个cell的顶点坐标和中心点坐标
    cell_vertices, centroids = calculate_cell_centroids(data['features'])

    # 保存数据到HDF5文件
    save_to_hdf5(cell_vertices, centroids, output_file)

def rewrite_crawled_feature():
    # 1. 加载 crawled_feature.csv
    csv_path = '/data/scratch/jiayin/milan/crawled_feature.csv'
    df = pd.read_csv(csv_path)

    # 2. 加载 milano-grid-data.h5 文件
    h5_file_path = '/data/scratch/jiayin/milan/milano-grid-coordinates.h5'
    with h5py.File(h5_file_path, 'r') as f:
        # 获取 cell 的中心点坐标
        centroids = f['centroids'][:]
        # centroids shape should be (10000, 2), where each row is the [x, y] center of each cell

    # 3. 将 centroids 合并到 DataFrame 中
    # 假设 cell_ids 的顺序与 centroids 中的顺序一致
    df['centroid_x'] = centroids[:, 0]
    df['centroid_y'] = centroids[:, 1]

    # 4. 保存新的 CSV 文件
    new_csv_path = '/data/scratch/jiayin/milan/crawled_feature_with_centroids.csv'
    df.to_csv(new_csv_path, index=False)

    print(f"New CSV file with centroids saved to: {new_csv_path}")


if __name__ == "__main__":
    main()
    # rewrite_crawled_feature()