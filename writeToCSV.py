import csv
import os
from getNodeJson import get_node_json
from downloadLatestByNeighboring import do_download_latestVersion, do_neighbor
from getClosestNode import get_closet_node, get_closet_pair_node

if __name__ == '__main__':
    # base_url = 
    osmDir = "osmDir"
    jsonDir = "jsonDir"
    data_csv = "dataset_0821.csv"

    cellId_list = []
    cellId_exist_json_list = []
    branch = "main"

    for root, dirs, files in os.walk(osmDir):
        for file in files:
            if file.endswith(".osm"):
                cellId = str(file.split('.')[0])
                cellId_list.append(cellId)
    for root, dirs, files in os.walk(jsonDir):
        for file in files:
            if file.endswith("nodes.json"):
                cellId = str(file.split('_')[0])
                cellId_exist_json_list.append(cellId)

    fields = ['cellId1', 'cellId2', 'distance_joint_1', 'distance_ele_1', 'distance_joint_2',
              'distance_ele_2', 'area', 'rc_distance1', 'rc_ele1', 'rc_distance2', 'rc_ele2', 'area_rc', 'lc_distance1',
              'lc_ele1', 'lc_distance2', 'lc_ele2', 'area_lc', "boundary_dis", "boundary_ele" 'label']

    # cellId_list = cellId_list[:100]
    with open(data_csv, 'a', newline='') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow(fields)

        index = 0
        length = len(cellId_exist_json_list)
        cellId_exist_json_list.reverse()
        for cellId in cellId_exist_json_list:
            index += 1
            print("index:all = {}/{}".format(index, length))

            # if cellId not in cellId_exist_json_list:
            #     continue

            neighboring_set = set()
            intersects_set = set()
            res = do_neighbor(cellId, branch, base_url)
            assert res.status_code == 200
            data = res.json()

            for part in data["in"]:
                neighbor_cellId = part["id"]
                neighboring_set.add(neighbor_cellId)
            for part in data["out"]:
                neighbor_cellId = part["id"]
                neighboring_set.add(neighbor_cellId)
            for part in data["intersects"]:
                cellId_intersects = part["id"]
                intersects_set.add(cellId_intersects)

            # Your code to calculate and append each row data...
            # Write the row data immediately into CSV

            for cellId_neighboring in neighboring_set:
                if cellId_neighboring in cellId_exist_json_list:
                    file_1 = "{}/{}_joint.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_joint.json".format(jsonDir, cellId_neighboring)
                    file_1 = "{}/{}_joint.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_joint.json".format(jsonDir, cellId_neighboring)
                    distance_joint_1, distance_ele_1, distance_joint_2, distance_ele_2, area = get_closet_pair_node(
                        file_1,
                        file_2)
                    file_1 = "{}/{}_rc.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_rc.json".format(jsonDir, cellId_neighboring)
                    rc_distance1, rc_ele1, rc_distance2, rc_ele2, area_rc = get_closet_pair_node(file_1, file_2)
                    file_1 = "{}/{}_lc.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_lc.json".format(jsonDir, cellId_neighboring)
                    lc_distance1, lc_ele1, lc_distance2, lc_ele2, area_lc = get_closet_pair_node(file_1, file_2)

                    file_1 = "{}/{}_boundary_nodes.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_boundary_nodes.json".format(jsonDir, cellId_neighboring)
                    boundary_dis1, boundary_ele1, boundary_dis2, boundary_ele2, area_boundary = get_closet_pair_node(file_1, file_2)
                    label = 1
                    data = [cellId, cellId_neighboring, distance_joint_1, distance_ele_1, distance_joint_2,
                            distance_ele_2, area, rc_distance1, rc_ele1, rc_distance2, rc_ele2, area_rc, lc_distance1,
                            lc_ele1, lc_distance2, lc_ele2, area_lc, boundary_dis1, boundary_ele1, boundary_dis2, boundary_ele2, area_boundary, label]

                    writer.writerow(data)

            for cellId_intersects in intersects_set:
                if cellId_intersects in cellId_exist_json_list:
                    file_1 = "{}/{}_joint.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_joint.json".format(jsonDir, cellId_intersects)
                    distance_joint_1, distance_ele_1, distance_joint_2, distance_ele_2, area = get_closet_pair_node(
                        file_1,
                        file_2)
                    file_1 = "{}/{}_rc.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_rc.json".format(jsonDir, cellId_intersects)
                    rc_distance1, rc_ele1, rc_distance2, rc_ele2, area_rc = get_closet_pair_node(file_1, file_2)
                    file_1 = "{}/{}_lc.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_lc.json".format(jsonDir, cellId_intersects)
                    lc_distance1, lc_ele1, lc_distance2, lc_ele2, area_lc = get_closet_pair_node(file_1, file_2)

                    file_1 = "{}/{}_boundary_nodes.json".format(jsonDir, cellId)
                    file_2 = "{}/{}_boundary_nodes.json".format(jsonDir, cellId_intersects)
                    boundary_dis1, boundary_ele1, boundary_dis2, boundary_ele2, area_boundary = get_closet_pair_node(
                        file_1, file_2)
                    label = 0
                    data = [cellId, cellId_neighboring, distance_joint_1, distance_ele_1, distance_joint_2,
                            distance_ele_2, area, rc_distance1, rc_ele1, rc_distance2, rc_ele2, area_rc, lc_distance1,
                            lc_ele1, lc_distance2, lc_ele2, area_lc, boundary_dis1, boundary_ele1, boundary_dis2,
                            boundary_ele2, area_boundary, label]

                    writer.writerow(data)
