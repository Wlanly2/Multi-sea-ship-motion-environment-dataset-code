import math
import csv
import shutil
import os
import numpy as np
import re
import shutil
import random


R = 6371000


def haversine_distance(lon1, lat1, lon2, lat2):

    # 将十进制度数转换为弧度
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    # 计算距离
    distance = R * c
    return distance


#这个是用来计算航速、航向角是否正确的
path3= r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\12-速度航向修正"
def calculate_new_position(lat1, lon1, speed, azimuth):
    """
    根据雷达位置信息和目标相对于雷达的方位角和距离，计算目标的经纬度。

    Args:
    lat1 (float): 雷达位置的纬度，单位为度数。
    lon1 (float): 雷达位置的经度，单位为度数。
    distance (float): 目标与雷达的直线距离，单位为米。
    azimuth (float): 目标相对于雷达的方位角，以北方向为基准，顺时针旋转角度为正，单位为度数。

    Returns:
    tuple: 目标的经度和纬度，单位为度数。
    """
    distance = speed*1.852*10/3.6

    # 将经纬度信息转换为弧度制
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    azimuth_rad = math.radians(azimuth)

    # 计算目标的纬度
    lat2_rad = math.asin(
        math.sin(lat1_rad) * math.cos(distance / R) + math.cos(lat1_rad) * math.sin(distance / R) * math.cos(
            azimuth_rad))

    # 计算目标的经度
    lon2_rad = lon1_rad + math.atan2(math.sin(azimuth_rad) * math.sin(distance / R) * math.cos(lat1_rad),
                math.cos(distance / R) - math.sin(lat1_rad) * math.sin(lat2_rad))

    # 将经纬度信息转换为度数制
    lat2 = math.degrees(lat2_rad)
    lon2 = math.degrees(lon2_rad)

    return lat2, lon2


path1 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\11-10秒差值"
path2 = r"C:\Users\wuyuegao\Desktop\newfolder"

def recreat(path1,path2):
    csvlist = os.listdir(path1)
    for sv in csvlist:
        csvpath = os.path.join(path1, sv)
        newpath = os.path.join(path2, sv)
        filelists = os.listdir(csvpath)
        for file in filelists:
            filepath = os.path.join(csvpath,file)
            newfilepath = os.path.join(newpath,file)
            readerdata = np.loadtxt(filepath, delimiter=',')
            a, b = readerdata.shape
            number = int(b / 9)

            datalist = []
            for i in range(number):
                latlist = []
                lonlist = []
                dislist = []

                True_lat = readerdata[1:, (i * 9) + 2]
                True_lon = readerdata[1:, (i * +9) + 3]
                pre_lat = readerdata[0, (i * 9) + 2]
                pre_lon = readerdata[0, (i * +9) + 3]
                velo = readerdata[:, (i * 9) + 4].reshape(-1, 1)
                angle = readerdata[:, (i * 9) + 5].reshape(-1, 1)

                for j in range(a-1):
                    ve = velo[j]
                    an = angle[j]
                    Tr_lat = True_lat[j]
                    Tr_lon = True_lon[j]

                    pre_lat, pre_lon = calculate_new_position(pre_lat, pre_lon, ve, an)
                    distance = haversine_distance(Tr_lon, Tr_lat, pre_lon, pre_lat)

                    latlist.append(pre_lat)
                    lonlist.append(pre_lon)
                    dislist.append(distance)

                latlist = np.array(latlist).reshape(-1, 1)
                lonlist = np.array(lonlist).reshape(-1, 1)
                dislist = np.array(dislist).reshape(-1, 1)

                shipdata = np.hstack((latlist, lonlist, dislist))
                datalist.append(shipdata)

            datalist = np.concatenate(datalist, axis=1)

            with open(newfilepath, 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(datalist)


recreat(path1,path2)



