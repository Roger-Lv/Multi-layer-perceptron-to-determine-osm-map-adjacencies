import sys
import time
import logging.handlers
import datetime
import requests
import hashlib
import os
from collections import OrderedDict

# 2022/10/17
# download-latest

logger = logging.getLogger('mylogger')
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rf_handler = logging.handlers.TimedRotatingFileHandler('download-latestVersion.log', when='midnight', interval=1,
                                                       backupCount=8,
                                                       atTime=datetime.time(0, 0, 0, 0))
rf_handler.setFormatter(logFormatter)
logger.addHandler(rf_handler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# 设置正确的环境(dev,dev,prod之一)及分支名称, 以及需要下载的cellid
envType = "prod"

output_latest_version_url = "/cellversion/latest-version"
download_latest_url = "/cellversion/download-latest"


def do_latestVersion(cell_id: str, branch: str, base_url: str):
    """ 返回resp字段 """
    fullUrl = base_url + output_latest_version_url
    query_params = {'cellId': cell_id, "branch": branch}
    resp = requests.get(url=fullUrl, params=query_params)
    return resp.text


def do_neighbor(cellId: str, branch, base_url: str):
    fullUrl = base_url + "/cellversion/neighbouring"
    query_params = {'cellId': cellId, "branch": branch}
    resp = requests.get(url=fullUrl, params=query_params)
    return resp


def do_download_latestVersion(cell_id: str, branch: str, base_url: str, dest_dir: str):
    # 发送请求
    fullUrl = base_url + download_latest_url
    query_params = {'cellId': cell_id, "branch": branch}

    down_res = requests.get(url=fullUrl, params=query_params)

    # 存入数据
    # full_file_path = dest_dir + "/" + cell_id + "@@" + branch + "@@" + version + ".osm"
    full_file_path = dest_dir + "/" + cell_id + ".osm"
    with open(full_file_path, "wb") as code:
        code.write(down_res.content)
        logger.info("full file path:{}".format(full_file_path))

    return down_res


if __name__ == '__main__':
    count = 0
    while count < 10:
        count += 1
        # base_url = the url for downloading osm files.
        parentDir = "osmDir"
        cell_id_list = []
        for root, dirs, files in os.walk(parentDir):
            for file in files:
                if file.endswith(".osm"):
                    # 提取文件名前面的数字
                    cellId = str(file.split('.')[0])
                    # 把数字添加到 list 中
                    cell_id_list.append(cellId)
        neighboring_set = set()
        intersects_set = set()

        cellId_set = set(cell_id_list)
        print(cellId_set)
        print("length_of_cellId_set:{}".format(len(cellId_set)))

        branch = "main"
        for cellId in cellId_set:
            res = do_neighbor(cellId, branch, base_url)
            data = res.json()
            for part in data["in"]:
                neighbor_cellId = part["id"]
                neighboring_set.add(neighbor_cellId)
            for part in data["out"]:
                neighbor_cellId = part["id"]
                neighboring_set.add(neighbor_cellId)
            for part in data["intersects"]:
                neighbor_cellId = part["id"]
                intersects_set.add(neighbor_cellId)

        print(neighboring_set)
        print("length_of_neighboring_set_before:{}".format(len(neighboring_set)))
        print(intersects_set)
        print("length_of_intersects_set_before:{}".format(len(intersects_set)))
        print(len(intersects_set))
        new_neighboring_set = neighboring_set - cellId_set
        print(print("length_of_neighboring_set_after:{}".format(len(new_neighboring_set))))
        new_intersects_set = intersects_set - cellId_set
        print(print("length_of_intersects_set_after:{}".format(len(new_intersects_set))))
        logger.info("当前环境环境:{}, 分支:{}, 域名:{}".format(envType, branch, base_url))

        # timeStr = time.strftime("%m%d-%H%M%S")
        # target_file = envType + "-" + branch + "-" + timeStr
        # download_dest_dir = os.getcwd() + "/" + target_file
        # os.mkdir(download_dest_dir)

        # 验证导出数据数量
        download_dest_dir = "./osmDir"
        cell_num = 0
        cell_num_found = 0
        count_in_list = len(cell_id_list)
        cell_id_set = set(cell_id_list)
        count_in_set = len(cell_id_set)
        logger.info("count_in_list{}个 ; count_in_set{}个".format(count_in_list, count_in_set))
        len_neighboring = len(new_neighboring_set)
        index = 0
        for cellId in new_neighboring_set:
            index += 1
            print("neighboring,index:all={}/{}".format(index, len_neighboring))
            res = do_download_latestVersion(cellId, branch, base_url, download_dest_dir)
            assert res.status_code == 200
        index = 0
        len_intersects = len(new_intersects_set)
        for cellId in new_intersects_set:
            index += 1
            print("intersects,index:all={}/{}".format(index, len_intersects))
            res = do_download_latestVersion(cellId, branch, base_url, download_dest_dir)
            assert res.status_code == 200

        logger.info("导出文件夹位置：{}".format(download_dest_dir))
