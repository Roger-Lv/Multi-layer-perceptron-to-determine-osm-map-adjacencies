import json
import os
import xml.dom.minidom


def get_node_json(cellId,parentDir):
    file_name = '{}/{}.osm'.format(parentDir, cellId)
    dom = xml.dom.minidom.parse(file_name)
    root = dom.documentElement
    nodelist = root.getElementsByTagName('node')
    waylist = root.getElementsByTagName('way')

    node_dic = {}
    node_dic_joint = {}
    node_dic_rc = {}
    node_dic_lc = {}

    # 统计记录所有node
    for node in nodelist:
        node_id = node.getAttribute('id')
        node_lat = float(node.getAttribute('lat'))
        node_lon = float(node.getAttribute('lon'))
        tags = node.getElementsByTagName('tag')
        elevation = None  # 初始设为None，表示未知
        for tag in tags:
            if tag.getAttribute('k') == 'ELEVATION':
                elevation = float(tag.getAttribute('v'))  # 若找到'ELEVATION'，则更新对应的值

        node_dic[node_id] = {'lat': node_lat, 'lon': node_lon, 'elevation': elevation}
        node_dic_joint[node_id] = {'lat': node_lat, 'lon': node_lon, 'elevation': elevation}
        node_dic_lc[node_id] = {'lat': node_lat, 'lon': node_lon, 'elevation': elevation}
        node_dic_rc[node_id] = {'lat': node_lat, 'lon': node_lon, 'elevation': elevation}
    rc_lc_ways = {}
    all_start_end_nodes = []
    # 排除非路node
    for way in waylist:
        taglist = way.getElementsByTagName('tag')
        road_flag = False
        rc_flag = False
        lc_flag = False
        joint_flag = False
        rc_or_lc_flag = False
        for tag in taglist:
            print(tag.getAttribute('v'))
            if tag.getAttribute('v') == 'RoadCenter' or tag.getAttribute('v') == 'LaneCenter':
                road_flag = True
                rc_or_lc_flag = True
            if tag.getAttribute('v') == 'RoadCenter':
                rc_flag = True
            if tag.getAttribute('v') == 'LaneCenter':
                lc_flag = True
            if tag.getAttribute('v') == 'CellJoint':
                joint_flag = True

        if not road_flag:
            ndlist = way.getElementsByTagName('nd')
            for nd in ndlist:
                nd_id = nd.getAttribute('ref')
                if nd_id in node_dic:
                    node_dic.pop(nd_id)
                    print("pop nd in all")
        if not rc_flag:
            ndlist = way.getElementsByTagName('nd')
            for nd in ndlist:
                nd_id = nd.getAttribute('ref')
                if nd_id in node_dic_rc:
                    node_dic_rc.pop(nd_id)
                    print("pop nd in rc")
        if not lc_flag:
            ndlist = way.getElementsByTagName('nd')
            for nd in ndlist:
                nd_id = nd.getAttribute('ref')
                if nd_id in node_dic_lc:
                    node_dic_lc.pop(nd_id)
                    print("pop nd in lc")
        if not joint_flag:
            ndlist = way.getElementsByTagName('nd')
            for nd in ndlist:
                nd_id = nd.getAttribute('ref')
                if nd_id in node_dic_joint:
                    node_dic_joint.pop(nd_id)
                    print("pop nd in joint")
        if rc_or_lc_flag:
            ndlist = way.getElementsByTagName('nd')
            way_id = way.getAttribute('id')
            start_node = ndlist[0].getAttribute('ref')
            end_node = ndlist[-1].getAttribute('ref')
            rc_lc_ways[way_id] = {'start_node': start_node, 'end_node': end_node}
            all_start_end_nodes.extend([start_node, end_node])
    print(len(node_dic))
    print(len(node_dic_rc))
    print(len(node_dic_lc))
    print(len(node_dic_joint))
    boundary_nodes = []
    for way in rc_lc_ways.values():
        if all_start_end_nodes.count(way['start_node']) == 1:
            boundary_nodes.append(way['start_node'])
        if all_start_end_nodes.count(way['end_node']) == 1:
            boundary_nodes.append(way['end_node'])

    # with open('./jsonDir/{}_node.json'.format(cellId), 'w') as fout:
    #     json.dump(node_dic, fout)
    with open('./jsonDir/{}_rc.json'.format(cellId), 'w') as fout:
        json.dump(node_dic_rc, fout)
    with open('./jsonDir/{}_lc.json'.format(cellId), 'w') as fout:
        json.dump(node_dic_lc, fout)
    with open('./jsonDir/{}_joint.json'.format(cellId), 'w') as fout:
        json.dump(node_dic_joint, fout)
    # Make a dictionary with only the boundary nodes
    boundary_dict = {node: data for node, data in node_dic.items() if node in boundary_nodes}

    # Save the result to a json file
    with open(f'./jsonDir/{cellId}_boundary_nodes.json', 'w') as fout:
        json.dump(boundary_dict, fout)


if __name__ == '__main__':
    parentDir = "osmDir"
    jsonDir = "jsonDir"
    cellId_list = []
    cellId_exist_json_list = []
    for root, dirs, files in os.walk(jsonDir):
        for file in files:
            if file.endswith(".json"):
                # 提取文件名前面的数字
                cellId = str(file.split('_')[0])
                # 把数字添加到 list 中
                cellId_exist_json_list.append(cellId)
    for root, dirs, files in os.walk(parentDir):
        for file in files:
            if file.endswith(".osm"):
                # 提取文件名前面的数字
                cellId = str(file.split('.')[0])
                # 把数字添加到 list 中
                cellId_list.append(cellId)

    print(cellId_list)
    index = 0
    length = len(cellId_list)
    for cellId in cellId_list:
        index += 1
        if cellId in cellId_exist_json_list:
            print("cellId_json_file:{} exist.".format(cellId))
            continue
        print("index:all = {}/{}".format(index, length))
        get_node_json(cellId, parentDir)

