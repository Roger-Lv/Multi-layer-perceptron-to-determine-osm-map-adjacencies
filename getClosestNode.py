import json
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.neighbors import BallTree
from geopy.distance import geodesic


# 加载两个Json文件，这里假设文件名分别为 "file_1.json" and "file_2.json"


def get_closet_node(file_name1, file_name2):
    with open(file_name1) as file1, open(file_name2) as file2:
        data_1 = json.load(file1)
        data_2 = json.load(file2)
    print("closet nodes:")
    # 将经纬度组合成numpy数组
    points_1 = np.array([[data['lat'], data['lon']] for data in data_1.values()])
    points_2 = np.array([[data['lat'], data['lon']] for data in data_2.values()])

    # 使用球树算法创建空间索引
    tree = BallTree(np.radians(points_2))

    # 查询最近点
    distances, indices = tree.query(np.radians(points_1), k=1)

    # 找到最小距离及其索引
    min_index = np.argmin(distances)
    closest_node_1_key = list(data_1.keys())[min_index]
    closest_node_2_key = list(data_2.keys())[indices[min_index][0]]

    # 获取并打印详细信息
    node_1_info = data_1[closest_node_1_key]
    node_2_info = data_2[closest_node_2_key]

    print('两个最近的节点分别为：', closest_node_1_key, '和', closest_node_2_key)
    print('节点1的经纬度：', (node_1_info['lat'], node_1_info['lon']))
    print('节点2的经纬度：', (node_2_info['lat'], node_2_info['lon']))
    print('他们的距离：', geodesic((node_1_info['lat'], node_1_info['lon']),
                                  (node_2_info['lat'], node_2_info['lon'])).kilometers * 1000, '米')
    print('elevation差值：', abs(node_1_info['elevation'] - node_2_info['elevation']), '米')
    distance1 = 0
    elevation1 = 0
    distance1 = geodesic((node_1_info['lat'], node_1_info['lon']),
                         (node_2_info['lat'], node_2_info['lon'])).kilometers * 1000
    elevation1 = abs(node_1_info['elevation'] - node_2_info['elevation'])
    return distance1, elevation1


def get_closet_pair_node(file_name1, file_name2):
    with open(file_name1) as file1, open(file_name2) as file2:
        data_1 = json.load(file1)
        data_2 = json.load(file2)

    points_1 = np.array([[data['lat'], data['lon']] for data in data_1.values()])
    points_2 = np.array([[data['lat'], data['lon']] for data in data_2.values()])

    tree = BallTree(np.radians(points_2))

    distances, indices = tree.query(np.radians(points_1), k=len(points_2))

    close_pairs = {}

    min_distance_indices = np.argpartition(distances.flatten(), 2)[:2]

    for index in min_distance_indices:
        i, j = np.unravel_index(index, distances.shape)

        node_1_key = list(data_1.keys())[i]
        node_2_key = list(data_2.keys())[indices[i][j]]
        close_pairs[(node_1_key, node_2_key)] = distances[i][j]

    distance_and_elevation = []
    pairs_info = []
    print("closet pair nodes:")
    for pair, distance in close_pairs.items():
        node_1_info = data_1[pair[0]]
        node_2_info = data_2[pair[1]]
        print(f'两个最近的节点分别为：{pair[0]} 和 {pair[1]}')
        print(f'节点1的经纬度：{(node_1_info["lat"], node_1_info["lon"])}')
        print(f'节点2的经纬度：{(node_2_info["lat"], node_2_info["lon"])}')
        print(
            f'他们的距离：{geodesic((node_1_info["lat"], node_1_info["lon"]), (node_2_info["lat"], node_2_info["lon"])).kilometers * 1000} 米')
        print(f'elevation差值：{abs(node_1_info["elevation"] - node_2_info["elevation"])} 米\n')

        pairs_info.append(((node_1_info["lat"], node_1_info["lon"]), (node_2_info["lat"], node_2_info["lon"])))

        distance_and_elevation.append((geodesic((node_1_info["lat"], node_1_info["lon"]),
                                                (node_2_info["lat"], node_2_info["lon"])).kilometers * 1000,
                                       abs(node_1_info["elevation"] - node_2_info["elevation"])))

    if len(distance_and_elevation) < 2:
        raise ValueError('Less than two pairs of points found.')

    def get_area(points):
        R = 6371e3  # 地球半径，单位：m
        lat1, lon1 = np.radians(points[0])
        lat2, lon2 = np.radians(points[1])
        lat3, lon3 = np.radians(points[2])
        lat4, lon4 = np.radians(points[3])

        # 计算每两点之间的经度、纬度差值
        delta_phi1 = abs(lat2 - lat1)
        delta_lambda1 = abs(lon2 - lon1)
        delta_phi2 = abs(lat4 - lat3)
        delta_lambda2 = abs(lon4 - lon3)

        # 估算中心纬度
        phi_avg1 = 0.5 * (lat1 + lat2)
        phi_avg2 = 0.5 * (lat3 + lat4)

        # 计算面积
        area1 = R ** 2 * delta_phi1 * delta_lambda1 * np.cos(phi_avg1)
        area2 = R ** 2 * delta_phi2 * delta_lambda2 * np.cos(phi_avg2)

        return area1 + area2

    area = get_area(
        [pairs_info[0][0], pairs_info[0][1], pairs_info[1][0], pairs_info[1][1]]
    )
    print("他们的面积为:{}".format(area))
    return distance_and_elevation[0][0], distance_and_elevation[0][1], distance_and_elevation[1][0], \
        distance_and_elevation[1][1], area


if __name__ == '__main__':
    parentDir = "jsonDir"
    # print("node")
    cellId1 = "1168926"
    cellId2 = "1173814"
    # file_1 = "{}/{}_node.json".format(parentDir, cellId1)
    # file_2 = "{}/{}_node.json".format(parentDir, cellId2)
    # get_closet_node(file_1, file_2)
    print("rc")
    file_1 = "{}/{}_rc.json".format(parentDir, cellId1)
    file_2 = "{}/{}_rc.json".format(parentDir, cellId2)
    get_closet_node(file_1, file_2)
    print("lc")
    file_1 = "{}/{}_lc.json".format(parentDir, cellId1)
    file_2 = "{}/{}_lc.json".format(parentDir, cellId2)
    get_closet_node(file_1, file_2)
    print("joint")
    file_1 = "{}/{}_joint.json".format(parentDir, cellId1)
    file_2 = "{}/{}_joint.json".format(parentDir, cellId2)
    get_closet_pair_node(file_1, file_2)
