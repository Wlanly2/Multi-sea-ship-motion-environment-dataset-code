import math
import csv
import shutil
import os
import numpy as np
import re
import shutil
import random



R = 6371000
def calculate_angle(lat1, lon1, lat2, lon2):
    """
    计算从 (lat1, lon1) 到 (lat2, lon2) 的方位角，以北为基准，顺时针旋转角度为正。
    """
    # 将纬度和经度转换为弧度
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # 计算经度差值
    delta_lon = lon2 - lon1

    # 计算方位角（以弧度为单位）
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    bearing_rad = np.arctan2(y, x)

    # 将方位角从弧度转换为度数，并在负角度上加 360 度
    bearing_deg = np.degrees(bearing_rad)
    bearing_deg[bearing_deg < 0] += 360

    return bearing_deg

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


path1 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\11-10秒差值"
path2 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\12-速度航向修正"

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
            print(a)
            number = int(b / 9)
            datalist = []
            for i in range(number):
                lat_list = readerdata[:, (i * 9) + 2].reshape(-1, 1)
                lon_list = readerdata[:, (i * +9) + 3].reshape(-1, 1)
                type = readerdata[:, (i * 9) + 6].reshape(-1, 1)
                length = readerdata[:, (i * 9) + 7].reshape(-1, 1)
                width = readerdata[:, (i * 9) + 8].reshape(-1, 1)
                pre_lat = lat_list[0]
                pre_lon = lon_list[0]
                speed_list = []
                angle_list = []
                for j in range(a - 1):
                    new_lat = lat_list[j + 1]
                    new_lon = lon_list[j + 1]
                    distance = haversine_distance(pre_lon, pre_lat, new_lon, new_lat)
                    angle = calculate_angle(pre_lat, pre_lon, new_lat, new_lon)
                    speed = (distance / 10) * 3.6 / 1.852

                    speed_list.append(speed)
                    angle_list.append(angle)

                    pre_lat = new_lat
                    pre_lon = new_lon

                speed_element = speed_list[-1]
                angle_element = angle_list[-1]

                speed_list.append(speed_element)
                angle_list.append(angle_element)

                speed_list = np.array(speed_list).reshape(-1, 1)
                angle_list = np.array(angle_list).reshape(-1, 1)
                shipdata = np.hstack((lat_list, lon_list, speed_list, angle_list, type, length, width))
                datalist.append(shipdata)
            datalist = np.concatenate(datalist, axis=1)

            with open(newfilepath, 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(datalist)


# recreat(path1,path2)





#检查是否存在插值处理后存在的一些错误的数据

path4 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\12-速度航向修正"
def sortmaxmin(path):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        csvfiles = os.listdir(filepath)
        for csvfile in csvfiles:
            csvpath = os.path.join(filepath,csvfile)
            csvdata = np.loadtxt(csvpath, delimiter=',')
            h, w = csvdata.shape
            number = w // 7
            for i in range(number):
                data = csvdata[:, (i * 7) + 2].reshape(-1, 1)
                width = csvdata[:, (i * 7) + 6].reshape(-1, 1)

                # 获取最大值
                max_value = data.max()
                max_width = width.max()
                # 获取最小值
                min_value = data.min()

                if max_value > 35 or min_value < 1 or max_width >45:
                    os.remove(csvpath)
                    print(csvpath)
                    break
                else:
                    continue

# sortmaxmin(path4)





#将原本的航向角转为余弦和正弦方向的角度，更为平滑  同时将船舶与目的地之间的距离加入特征
path5 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\12-速度航向修正"
path6 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\13-添加新特征"

# 角度值转换为正弦和余弦值
def angle_to_sin_cos(angle):
    # 将角度映射到[0, 2*pi]范围内
    angle_rad = np.deg2rad(angle)
    # 计算正弦和余弦值
    sin_value = np.sin(angle_rad)
    cos_value = np.cos(angle_rad)
    total_value = np.hstack([sin_value,cos_value])
    return total_value

#sin、cos值反推计算角度
def sin_cos_to_angle(sin_value, cos_value):
    # 计算角度的弧度值
    angle_rad = np.arctan2(sin_value, cos_value)
    # 将弧度值转换为角度
    angle_deg = np.rad2deg(angle_rad)
    # 将角度调整到[0, 360)的范围内
    angle_deg = angle_deg % 360
    return angle_deg

#距离目的地距离
def normalize_distance(data):
    # 获取最后一行数据作为参考经纬度
    # 计算每一行数据与参考经纬度之间的欧氏距离
    distances = np.linalg.norm(data - data[-1], axis=1).reshape(-1,1)
    # 对距离进行归一化处理
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    normalized_distances = (distances - min_distance) / (max_distance - min_distance)

    return normalized_distances

def processdouble(path1,path2):
    files = os.listdir(path1)
    for file in files:
        filepath = os.path.join(path1,file)
        newfilepath = os.path.join(path2,file)
        csvlists = os.listdir(filepath)
        for csvlist in csvlists:
            csvpath = os.path.join(filepath,csvlist)
            newcsvpath = os.path.join(newfilepath,csvlist)
            csvdata = np.loadtxt(csvpath, delimiter=',')
            a, b = csvdata.shape
            number = b // 7
            scene_data = []
            for i in range(number):
                data = csvdata[:, 7 * i:7 * (i + 1)]
                angledata = data[:, 3].reshape(-1, 1)
                angle_pre = angle_to_sin_cos(angledata)
                motiondata = data[:, :3]
                typedata = data[:, 4:]
                posdata = data[:, :2]
                posdata = normalize_distance(posdata)
                data = np.hstack([motiondata, angle_pre, typedata, posdata])
                scene_data.append(data)
            scene_data = np.hstack(scene_data)
            with open(newcsvpath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(scene_data)


# processdouble(path5,path6)





def changeposition(path):
    files = os.listdir(path)
    c = 1
    for file in files:
        filepath = os.path.join(path,file)
        newfilepath = os.path.join(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\14-切割区分", file)
        csvlists = os.listdir(filepath)
        for csvlist in csvlists:
            csvpath = os.path.join(filepath,csvlist)
            csvdata = np.loadtxt(csvpath, delimiter=',')
            a, b = csvdata.shape
            newdata = (csvdata.reshape(-1, 9)[:, :-1]).reshape(a, -1)
            number = b // 9

            for i in range(number):
                singledata1 = csvdata[:, i * 9:(i + 1) * 9]
                if np.all(singledata1[:, 2] > 6):
                    singledata2 = newdata[:, i * 8:(i + 1) * 8]
                    singledata3 = np.delete(newdata, slice(i * 8, (i + 1) * 8), axis=1)
                    totaldata = np.hstack((singledata1, singledata2, singledata3))
                    e = '{}.csv'.format(c)
                    newcwvpath = os.path.join(newfilepath,e)
                    with open(newcwvpath, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(totaldata)
                    c += 1

                else:
                    continue


# changeposition(path6)




#这个是用来计算数据中每个特征的最大最小值的，
# 一方面方便归一化，一方面在前期筛选位置速度后，还是会存在错误的速度值，还是需要进一步删选
#这种错误还是来源于船舶AIS数据只有位置信息是相对准确的，其他的都呃呃

path7= r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\12-速度航向修正"
def sortmaxmin2(path):
    files = os.listdir(path)
    mindata = np.array([1000, 1000, 1000, 1000,1000,1000,1000,]).reshape(1, 7)
    maxdata = np.array([-1000, -1000, -1000, -1000,-1000,-1000,-1000]).reshape(1, 7)
    for file in files:
        filepath = os.path.join(path, file)
        csvlists = os.listdir(filepath)
        for csvlist in csvlists:
            csvpath = os.path.join(filepath,csvlist)
            csvdata = np.loadtxt(csvpath, delimiter=',')
            h, w = csvdata.shape
            number = w // 7
            for i in range(number):
                data = csvdata[:, i * 7:(i + 1) * 7]
                # 计算每个特征的最大值和最小值
                min_vals = np.min(data, axis=0).reshape(1, 7)  # 沿着列方向计算最小值
                mindata = np.minimum(mindata, min_vals).reshape(1, 7)

                max_vals = np.max(data, axis=0).reshape(1, 7)  # 沿着列方向计算最大值
                maxdata = np.maximum(maxdata, max_vals).reshape(1, 7)


    print(mindata)
    print(maxdata)
    print(mindata[0][4])
    print(mindata[0][5])
    print(mindata[0][6])

    print(maxdata[0][4])
    print(maxdata[0][5])
    print(maxdata[0][6])

# sortmaxmin2(path7)




#过采样和欠采样完美执行的程序
def process_scenes(path, samplesize):
    files = os.listdir(path)
    c = 0
    for file in files:
        filepath = os.path.join(path, file)
        csvlists = os.listdir(filepath)
        b = 0

        # 随机打乱文件列表
        random.shuffle(csvlists)
        while b < samplesize:
            for csvlist in csvlists:

                if b >= samplesize:
                    break      # 如果b已经达到或超过samplesize，退出for循环
                b += 1
                c += 1
                csvpath = os.path.join(filepath, csvlist)
                newcsvpath = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\11-挑选文件\{}.csv".format(c)
                csvdata = np.loadtxt(csvpath, delimiter=',')
                with open(newcsvpath, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(csvdata)
                print(b)

# process_scenes(path4, 500)






#再一次的根据条件的不同而进行文件挑选,这个不是
def random_select_csv(path):

        # 获取所有 CSV 文件名
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

        # 挑选 num_files 个随机文件
        selected_files = random.sample(csv_files, 800)

        # 移动文件到目标文件夹
        for filename in selected_files:
            shutil.copyfile(os.path.join(path, filename),
    os.path.join(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\15-挑选文件",filename))

# random_select_csv(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\14-切割区分\1")





path8 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\15-挑选文件"
# 计算每个特征的最大值和最小值
min_vals = np.array([48.555,-122.92,0,-1,-1,0,0,0]).reshape(1,8)
max_vals = np.array([48.61,-122.84,36,1,1,99,154,40]).reshape(1,8)

def normal (datas):
    # 对每个特征进行最大最小归一化
    # 对每艘船的特征进行归一化
    normalized_features = (datas - min_vals) / (max_vals - min_vals)
    # 将归一化后的特征放回对应的位置
    return normalized_features

def choose_normal(path):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        newfilepath = os.path.join(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\16-归一化数据",file)

        data = np.loadtxt(filepath, delimiter=',')
        a,b = data.shape

        singledata1 = data[:,8].reshape(a,1)


        singledata2 = data[:,:8]
        normaldata2 = normal(singledata2)

        singledata3 = data[:,9:].reshape(-1,8)
        normaldata3 = normal(singledata3).reshape(a,-1)

        modified_array = np.hstack((normaldata2,singledata1,normaldata3))


        with open( newfilepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(modified_array)

# choose_normal(path8)






path9 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\16-归一化数据"
path10 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\17-填充数据"

def recreate(path1,path2):
    files = os.listdir(path1)
    for file in files:
        csvpath = os.path.join(path1, file)
        newpath = os.path.join(path2,file)
        stacked_arr = np.loadtxt(csvpath, delimiter=',')
        a, b = stacked_arr.shape
        number = (b-9)//8
        if number > 9:
            print('wocao')
            print(csvpath)

        else:
            array_padding = np.zeros((a, (9-number)*8))
            stacked_arr = np.hstack((stacked_arr, array_padding))

        weight_arr = np.zeros((a, 9))
        weight_arr[:, :number] = 1

        stacked_arr = np.hstack((stacked_arr, weight_arr))

        with open(newpath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(stacked_arr)

# recreate(path9,path10)



path11 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\16-归一化数据"
path12 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\17-填充数据2"

def recreate2(path1,path2):
    files = os.listdir(path1)
    for file in files:
        csvpath = os.path.join(path1, file)
        newpath = os.path.join(path2,file)
        stacked_arr = np.loadtxt(csvpath, delimiter=',')
        oneshipdata = stacked_arr[:,:9]
        stacked_arr = np.delete(stacked_arr, slice(0, 17), axis=1)
        a, b = stacked_arr.shape
        number = b//8
        if number < 1:
            array_padding2 = np.zeros((a, 72))
            stacked = np.hstack((oneshipdata, array_padding2))

        else:
            array_padding1 = np.zeros((a,number, 1))
            stacked_arr = stacked_arr.reshape(a,number,8)
            stacked_arr = np.concatenate((stacked_arr,array_padding1),axis=2)
            stacked_arr = stacked_arr.reshape(a,-1)

            array_padding2 = np.zeros((a, (8-number)*9))
            stacked= np.hstack((oneshipdata, stacked_arr, array_padding2))

        weight_arr = np.zeros((a, 9))
        weight_arr[:, :number+1] = 1

        stacked = np.hstack((stacked, weight_arr))

        with open(newpath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(stacked)

# recreate2(path11,path12)










def velocity(path):
    v = []
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        csvdata = np.loadtxt(filepath, delimiter=',')
        h, w = csvdata.shape
        number = w // 4
        for i in range(number):
            data = csvdata[:, (i * 4) + 2:(i * 4) + 3]
            v.append(data)
    stacked_array = np.vstack(v)
    with open(r"C:\Users\wuyuegao\Desktop\论文投递文件\new-velocity.csv",
              'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(stacked_array)

# velocity(path)


def counttime(path):
    v = []
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        csvdata = np.loadtxt(filepath, delimiter=',')
        h, w = csvdata.shape
        v.append(h)
    stacked_array = np.vstack(v)
    with open(r"C:\Users\wuyuegao\Desktop\论文投递文件\new-time.csv",
              'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(stacked_array)

# counttime(path)



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



def test(path):
    files = os.listdir(path)
    for file in files:
        csvpath = os.path.join(path, file)
        readerdata = np.loadtxt(csvpath, delimiter=',')
        a, b = readerdata.shape
        number = int((b) /7)

        for i in range(number):
            mpcdata = []
            origndata = readerdata[:, (i*7):(i*7) + 2]
            motiondata = readerdata[:-1, (i*7) + 2:(i*7) + 4]
            previous_lat = readerdata[0,(i*7)]
            previous_lon = readerdata[0,(i*7)+1]

            mpcdata.append(previous_lat)
            mpcdata.append(previous_lon)
            for motion in motiondata:
                new_lat, new_lon = calculate_new_position(previous_lat, previous_lon, motion[0], motion[1])
                previous_lat = new_lat
                previous_lon = new_lon
                mpcdata.append(previous_lat)
                mpcdata.append(previous_lon)
            real_lat = origndata[-1, 0]
            real_lon = origndata[-1, 1]
            fina_lat = mpcdata[-2]
            fina_lon = mpcdata[-1]
            distanceloss = haversine_distance(real_lon, real_lat, fina_lon, fina_lat)
            if distanceloss >5:
                print('错误')
                print(csvpath)

# test(path2)



















































































