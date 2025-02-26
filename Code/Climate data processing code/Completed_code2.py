import csv
import datetime
import shutil
import numpy as np
import os
from scipy.interpolate import interp1d
import pandas as pd
import time
import math


#去除其中特征值为空白的行
path1=r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\201231-211231.csv"


def ct(a):# 将带有T的字符串转为时间格式,且直接用datetime.time模块内置的方法，得到年、月、日、时、分、秒
    return datetime.datetime.strptime(a, "%Y-%m-%dT%H:%M:%S")

def time_to_seconds(time_str):
    time_struct = time.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
    return int(time.mktime(time_struct))

def tc(timestamp):
    # 将时间戳转换为datetime对象
    dt = datetime.datetime.fromtimestamp(timestamp)
    # 格式化datetime对象为指定格式的时间字符串
    time_string = dt.strftime("%Y-%m-%dT%H:%M:%S")
    return time_string


#将从网站上下载的AIS原文件按照呼号分类
def choose(path):


    # 提取目录路径（即去掉文件名部分）
    folderpath = os.path.dirname(path)
    foldername = os.path.basename(folderpath)

    folderpath2 = os.path.join(folderpath,"船舶独立数据-1")
    if not os.path.exists(folderpath2):
        os.makedirs(folderpath2)
        print(f'Created directory: {folderpath2}')

    with open(path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]
    aname = []
    b = 0
    for i, a in enumerate(rows):
        alist = []
        value = a[0]

        if value not in aname and float(a[4])>1 :
            b = b + 1

            alist.append((a[1], a[0], float(a[2]), float(a[3]), float(a[4]), float(a[5]), float(a[6]), a[10], a[12], a[13]))
            for a2 in rows[i + 1:]:
                if value == a2[0] and float(a2[4])>1:

                    alist.append((a2[1], a2[0], float(a2[2]), float(a2[3]), float(a2[4]), float(a2[5]),float(a2[6]), a2[10], a2[12], a2[13]))
            headers = ['TIME','MMSI','LAT', 'LON', 'SOG', 'COG','HEADING','TYPE','LENGTH','WIDTH']
            print(b)
            with open(r"{}\{}.csv".format(folderpath2,b), 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(alist)
            aname.append(value)
    print(aname)
    print(len(aname))



    # 第二步，检查csv文件中的缺失情况,对齐进行填充
    files2 = os.listdir(folderpath2)
    folderpath3 = os.path.join(folderpath, "补充缺失值-2")

    if not os.path.exists(folderpath3):
        os.makedirs(folderpath3)
        print(f'Created directory: {folderpath3}')


    for file2 in files2:
        filepath2 = os.path.join(folderpath2, file2)
        outputfilepath3 = os.path.join(folderpath3, file2)

        # 读取CSV文件，假设有表头
        df = pd.read_csv(filepath2, header=0)

        # 检查列数，确保访问的列索引存在
        total_columns = len(df.columns)
        print(f"文件 {file} 有 {total_columns} 列")

        for column_index in range(3, total_columns):  # 注意范围，3是起始列
            # 检查该列是否全是空值
            if df.iloc[:, column_index].isnull().all():
                print('缺失全部')
                print(filepath2)
                df.iloc[:, column_index] = df.iloc[:, column_index].fillna(0)  # 使用指定值替换空值
            elif df.iloc[:, column_index].isnull().any():
                # 查找该列的最大值
                max_value = df.iloc[:, column_index].max()
                df.iloc[:, column_index] = df.iloc[:, column_index].fillna(max_value)
                print('不全')
                print(filepath2)
            else:
                continue

        # 保存到新文件
        df.to_csv(outputfilepath3, index=False, header=True)  # 保留表头



    # 第三步 直接分类处理的AIS分类数据中有的数据时间顺序混乱，不能直接使用，需要重新进行排序。
    files3 = os.listdir(folderpath3)

    folderpath4 = os.path.join(folderpath, "船舶时间顺序排列-3")
    if not os.path.exists(folderpath4):
        os.makedirs(folderpath4)
        print(f'Created directory: {folderpath4}')


    for file3 in files3:
        filepath3 = os.path.join(folderpath3, file3)
        outputfilepath4 = os.path.join(folderpath4, file3)

        # 读取CSV文件
        with open(filepath3, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # 读取标题行
            rows = list(reader)  # 读取所有数据行

        # 定义时间列的索引（假设时间列是第一列）
        time_index = 0

        # 将时间字符串转换为 datetime 对象
        for row in rows:
            row[time_index] = datetime.datetime.strptime(row[time_index], "%Y-%m-%dT%H:%M:%S")

        # 按照时间列排序
        rows.sort(key=lambda row: row[time_index])

        # 将 datetime 对象转换回字符串格式
        for row in rows:
            row[time_index] = row[time_index].strftime("%Y-%m-%dT%H:%M:%S")

        # 写入排序后的数据到新的 CSV 文件
        with open(outputfilepath4, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # 写入标题行
            writer.writerows(rows)  # 写入排序后的数据行


# choose(path1)



#第四步：由于这是两个月内的数据，难免有重复的船舶经过同一区域，
# 需要分析AIS数据一般的时间间隔是多少，确定划分的时间间隔。
#分析一下到底该采用什么样的AIS数据间隔，看一下大数据分布
path2=r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\船舶时间顺序排列-3"
def Ansystime(path):
    files = os.listdir(path)
    with open(r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\Sanjuanisland-AIS时间间隔.csv"
            , 'w', newline='') as f:
        writer = csv.writer(f)
        for file in files:
            csvfile = os.path.join(path, file)
            print(csvfile)
            with open(csvfile, 'r') as f:
                csvlist = csv.reader(f)
                csvlist = list(csvlist)

            csvlist = csvlist[1:]  # 跳过第一行（列名称）
            a = len(csvlist)
            for i in range(a - 1):
                time1 = ct(csvlist[i][0])
                time2 = ct(csvlist[i + 1][0])
                value = (time2-time1).total_seconds()  # 计算时间间隔（以秒为单位）
                writer.writerow([value])


    # 第五步：删除其中大于XXX的数据,XX秒的时间间隔对于AIS数据的时间间隔来说时间太长
    df = pd.read_csv(r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\Sanjuanisland-AIS时间间隔.csv", header=None)  # header=None表示没有表头

    filtered_df = df[df[0] <= 1000]

    filtered_df.to_csv(r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\时间阈值.csv"
                       , index=False, header=False)

# Ansystime(path2)




#分析后、可以看出设置为最大阈值设置为200秒进行选择是比较最好的
#由于这是xx月内的数据，难免有重复的船舶经过同一区域，
#根据已经分析得到的时间，将AIS文件切割成不同文件,同时除去AIS中重复的数据

path3=r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\船舶时间顺序排列-3"

def split(path):

    # 提取目录路径（即去掉文件名部分）
    folderpath = os.path.dirname(path)
    foldername = os.path.basename(folderpath)

    folderpath2 = os.path.join(folderpath, "船舶按时间分割-4")

    if not os.path.exists(folderpath2):
        os.makedirs(folderpath2)
        print(f'Created directory: {folderpath2}')

    files = os.listdir(path)
    for a,file in enumerate(files):
        csvfile = os.path.join(path, file)
        with open(csvfile, 'r') as f:
            csvlist=csv.reader(f)
            headers = next(csvlist)  # 读取标题行,同时指指针跳过第一行
            csvlist=list(csvlist)

        data=[]
        b=1
        for i in csvlist:
            if len(data) == 0:
                data.append(i)
            else:
                if time_to_seconds(data[-1][0]) != time_to_seconds(i[0]):#时间不同才读入
                    if time_to_seconds(i[0])-time_to_seconds(data[-1][0]) >0 and time_to_seconds(i[0])-time_to_seconds(data[-1][0])<=150:
                        data.append(i)
                    else:
                        with open(r"{}\{}.{}.csv".format(folderpath2,a+1, b), 'w',  newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(headers)  # 写入标题行
                            writer.writerows(data)
                        print(b)
                        b = b + 1
                        data = []
                        data.append(i)
                else:#时间相同不读入
                    continue
        with open(r"{}\{}.{}.csv".format(folderpath2,a+1,b), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # 写入标题行
            writer.writerows(data)


    #检查一下原始文件中的船舶速度分布
    files = os.listdir(folderpath2)
    with open(r"{}\Sanjuanisland-速度分布.csv".format(folderpath)
            , 'w', newline='') as f:
        writer = csv.writer(f)
        for file in files:
            csvfile = os.path.join(folderpath2, file)
            with open(csvfile, 'r') as f:
                csvlist = csv.reader(f)
                headers = next(csvlist)
                csvlist = list(csvlist)
            for i in csvlist:
                writer.writerow([i[4]])


# split(path3)


path4=r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\船舶按时间分割-4"
def Ansystype(path):

    # 提取目录路径（即去掉文件名部分）
    folderpath = os.path.dirname(path)
    foldername = os.path.basename(folderpath)

    # 按照AIS定义更精确地映射类型
    vessel_types = {
        'Unknow':0,
        'WIG (20-29)': 0,
        'Fishing (30)': 0,
        'Port operation ship (31-35)': 0,
        'Sailing Vessel (36-37)': 0,
        'Reserved for future use (38-39)': 0,
        'Port service vessels(50-59)':0,
        'Passenger Vessels(60-69)':0,
        'Cargo Vessels(70-79)':0,
        'Liquid cargo ship(80-89)':0,
        'Other Vessels(90-100)':0
    }

    files = os.listdir(path)
    for file in files:
        csvfile = os.path.join(path, file)
        print(csvfile)

        with open(csvfile, 'r') as f:
            csvlist = csv.reader(f)
            headers = next(csvlist)  # 跳过标题行
            csvlist = list(csvlist)

            if csvlist:
                # 提取第一条AIS记录的类型编号
                vessel_type_code = int(float(csvlist[0][7]))

                if vessel_type_code == 0:  # 跳过未知类型
                    vessel_types['Unknow'] += 1
                if 20 <= vessel_type_code <= 29:
                    vessel_types['WIG (20-29)'] += 1
                elif vessel_type_code == 30:
                    vessel_types['Fishing (30)'] += 1
                elif 31 <= vessel_type_code <= 35:
                    vessel_types['Port operation ship (31-35)'] += 1
                elif 36 <= vessel_type_code <= 37:
                    vessel_types['Sailing Vessel (36-37)'] += 1
                elif 38 <= vessel_type_code <= 39:
                    vessel_types['Reserved for future use (38-39)'] += 1
                elif 50 <= vessel_type_code <= 59:
                    vessel_types['Port service vessels(50-59)'] += 1
                elif 60 <= vessel_type_code <= 69:
                    vessel_types['Passenger Vessels(60-69)'] += 1
                elif 70 <= vessel_type_code <= 79:
                    vessel_types['Cargo Vessels(70-79)'] += 1
                elif 80 <= vessel_type_code <= 89:
                    vessel_types['Liquid cargo ship(80-89)'] += 1
                else:
                    vessel_types['Other Vessels(90-100)'] += 1

        # 删除数据中少于4行的数据,三次插值至少四个点，之后的处理中在删除速度过快和过慢的数据，在上一步分析中删除速度小于2节和大于30的数据
        if len(csvlist) < 4:
            os.remove(csvfile)
            print('已删除{}'.format(csvfile))
        else:
            continue

    # 将统计结果写入CSV文件
    output_file = r"{}\{}-vesseltype.csv".format(folderpath,foldername)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Vessel Type', 'Count'])
        for type_name, count in vessel_types.items():
            writer.writerow([type_name, count])







    files = os.listdir(path)
    new_path = os.path.join(folderpath,'船舶航向sin-cos-5')
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        print('文件夹已创建')

    for file in files:
        filepath = os.path.join(path, file)
        newfilepath = os.path.join(new_path, file)
        # 读取CSV文件
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # 读取列名
            data = list(reader)

        # 准备存储转换后的数据
        converted_data = []

        for row in data:
            # 提取原始数据
            time = row[0]
            mmsi = row[1]
            lat = float(row[2])
            lon = float(row[3])
            sog = float(row[4])
            cog = float(row[5])
            heading = float(row[6])
            vesseltype = row[7]
            length = row[8]
            width = row[9]

            # 将COG和HEADING转换为正弦和余弦值
            cog_sin = np.sin(np.radians(cog))
            cog_cos = np.cos(np.radians(cog))
            heading_sin = np.sin(np.radians(heading))
            heading_cos = np.cos(np.radians(heading))

            # 组合新的数据行
            new_row = [time, mmsi, lat, lon, sog,

                       cog_sin, cog_cos, heading_sin, heading_cos,

                       vesseltype, length, width]

            converted_data.append(new_row)

        # 保存转换后的数据到新的CSV文件
        with open(newfilepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入新的列名
            new_headers = [
                'TIME', 'MMSI', 'LAT', 'LON', 'SOG',
                'COG_SIN', 'COG_COS', 'HEADING_SIN', 'HEADING_COS',
                'TYPE', 'LENGTH', 'WIDTH'
            ]
            writer.writerow(new_headers)
            writer.writerows(converted_data)







    files = os.listdir(new_path)
    output_path = r"{}\{}".format(folderpath,'船舶数据按时间插值-6')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('文件夹已创建')

    for a, file in enumerate(files):
        csvfile = os.path.join(new_path, file)
        print(csvfile)

        # 读取数据，跳过第一行的列名
        with open(csvfile, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # 跳过列名
            data = list(reader)

        # 提取时间列并转换为时间戳秒数
        b0 = np.array([time_to_seconds(row[0]) for row in data], dtype='int')

        # 提取其他列数据
        aisdata = np.array(
            [[int(float(row[1])), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]),
              float(row[7]), float(row[8]), int(float(row[9])), int(float(row[10])), int(float(row[11]))] for row in
             data])

        a1 = aisdata[:, 0]  # MMSI
        a2 = aisdata[:, 1]  # LAT
        a3 = aisdata[:, 2]  # LON
        a4 = aisdata[:, 3]  # SOG
        a5 = aisdata[:, 4]  # COG-SIM
        a6 = aisdata[:, 5]  # COG-COS
        a7 = aisdata[:, 6]  # HEADING-SIN
        a8 = aisdata[:, 7]  # HEADING-COS
        a9 = aisdata[:, 8]  # TYPE
        a10 = aisdata[:, 9]  # LENGTH
        a11 = aisdata[:, 10]  # WIDTH

        numbers = b0[-1] - b0[0] + 1

        # 插值
        f1 = interp1d(b0, a1, kind='linear')
        f2 = interp1d(b0, a2, kind='cubic')
        f3 = interp1d(b0, a3, kind='cubic')
        f4 = interp1d(b0, a4, kind='cubic')
        f5 = interp1d(b0, a5, kind='cubic')
        f6 = interp1d(b0, a6, kind='cubic')
        f7 = interp1d(b0, a7, kind='cubic')
        f8 = interp1d(b0, a8, kind='cubic')
        f9 = interp1d(b0, a9, kind='linear')
        f10 = interp1d(b0, a10, kind='linear')
        f11 = interp1d(b0, a11, kind='linear')

        time = np.linspace(b0[0], b0[-1], num=numbers, endpoint=True)

        mmsi = f1(time)
        lat = f2(time)
        lon = f3(time)
        sog = f4(time)
        cogsin = f5(time)
        cogcos = f6(time)
        headsin = f7(time)
        headcos = f8(time)
        type = f9(time)
        length = f10(time)
        width = f11(time)

        # 重新调整形状
        time = time.reshape(-1, 1)
        mmsi = mmsi.reshape(-1, 1)
        lat = lat.reshape(-1, 1)
        lon = lon.reshape(-1, 1)
        sog = sog.reshape(-1, 1)
        cogsin = cogsin.reshape(-1, 1)
        cogcos = cogcos.reshape(-1, 1)
        headsin = headsin.reshape(-1, 1)
        headcos = headcos.reshape(-1, 1)
        type = type.reshape(-1, 1)
        length = length.reshape(-1, 1)
        width = width.reshape(-1, 1)

        # 合并数据
        alist = np.hstack((time, mmsi, lat, lon, sog, cogsin, cogcos, headsin, headcos, type, length, width))

        # 保存数据
        output_file = r"{}\{}".format(output_path, file)
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(alist)


# Ansystype(path4)




#第十二步，将插值填充的数据统一合并为同一个csv文件中
path5=r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\船舶数据按时间插值-6"

def merge_and_sort_csv(path):

    # 提取目录路径（即去掉文件名部分）
    folderpath = os.path.dirname(path)
    foldername = os.path.basename(folderpath)
    output_path = r'{}\{}timesort-7.csv'.format(folderpath,foldername)

    all_data = []
    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            print(file_path)
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # 读取并跳过列名
                for row in reader:
                    all_data.append(row)

    # 根据时间戳（第一列）进行排序
    all_data.sort(key=lambda x: int(float(x[0])))
    # 将排序后的数据写入新的CSV文件
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入列名
        writer.writerow(headers)
        writer.writerows(all_data)








    outfolderpath = r'{}\船舶数据同时刻-8'.format(folderpath)

    if not os.path.exists(outfolderpath):
        os.makedirs(outfolderpath)

        print('文件夹已经创建')

    df = pd.read_csv(output_path, header=None, skiprows=1)
    lens = len(df)
    b = 1
    # 初始化变量
    range_offset = 40
    prename = None
    pretime = None
    buffer = []

    for index, row in df.iterrows():
        # 将时间列转换为浮点数，然后转换为整数
        currenttime = int(float(row[0]))
        # 获取行切片，范围为当前行的前后range_offset行
        start_index = max(0, index - range_offset)
        end_index = min(lens, index + range_offset + 1)
        range_df = df.iloc[start_index:end_index]

        # 根据时间范围搜索整个 CSV 文件中具有相同时间值的行
        matching_rows = range_df[range_df[0].apply(lambda x: int(float(x))) == currenttime]
        # 计算用来判断的特征值
        currentname = set(matching_rows[1])

        if pretime is None and prename is None:
            pretime = currenttime
            prename = currentname
            buffer.append(matching_rows)

        elif pretime == currenttime:
            continue

        elif currenttime > pretime:
            if prename == currentname and (currenttime - pretime) == 1:  # 判断是否连续
                pretime = currenttime
                buffer.append(matching_rows)
            else:
                # 将所有匹配的 DataFrame 合并为一个 DataFrame
                combined_df = pd.concat(buffer, ignore_index=True)
                combined_df.to_csv(r"{}\{}.csv".format(outfolderpath,b),index=False, header=False)
                buffer = [matching_rows]
                pretime = currenttime
                prename = currentname
                print(b)
                b += 1

    if buffer:
        combined_df = pd.concat(buffer, ignore_index=True)
        combined_df.to_csv(r"{}\{}.csv".format(outfolderpath,b),index=False, header=False)





    # 读取 CSV 文件
    files = os.listdir(outfolderpath)
    outfolderpath2 = r'{}\重排列-9'.format(folderpath)
    if not os.path.exists(outfolderpath2):
        os.makedirs(outfolderpath2)
        print('文件夹已经创建')

    for file in files:
        filepath = os.path.join(outfolderpath, file)
        print(filepath)
        newpath = os.path.join(outfolderpath2, file)

        df = pd.read_csv(filepath, header=None)

        # 根据第一个特征值（时间）进行分组
        grouped = df.groupby(0)

        # 初始化一个列表来保存合并后的结果
        combined_rows = []

        # 遍历每个组（每个唯一的时间）
        for name, group in grouped:
            # 对每个组根据第二个特征值（船代号）进行排序
            sorted_group = group.sort_values(by=1).reset_index(drop=True)

            # 将每组的行合并为一行，使用flatten将DataFrame转为一维数组
            combined_row = sorted_group.values.flatten()

            # 添加到结果列表
            combined_rows.append(combined_row)

        # 将结果列表转换为DataFrame
        combined_df = pd.DataFrame(combined_rows)

        # 保存合并后的结果到新文件
        combined_df.to_csv(newpath, index=False, header=False)


# merge_and_sort_csv(path5)



path6 = r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\重排列-9"

#第xx步，将文件转移位置到不同的文件夹下
def removeplace(path):

    # 提取目录路径（即去掉文件名部分）
    folderpath = os.path.dirname(path)
    foldername = os.path.basename(folderpath)
    outfolderpath = r'{}\场景数量分类-10'.format(folderpath)
    if not os.path.exists(outfolderpath):
        os.makedirs(outfolderpath)
        print('文件夹已创建')

    # 读取指定路径下的所有文件
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        df = pd.read_csv(filepath, header=None)
        a, b = df.shape
        number = b // 12

        # 构建目标文件夹路径
        newfile = r"{}\{}".format(outfolderpath,number)

        # 检查目标文件夹是否存在，不存在则创建
        if not os.path.exists(newfile):
            os.makedirs(newfile)
            print('已完成文件夹创建')

        # 构建新文件路径
        newfilepath = os.path.join(newfile, file)

        # 复制文件到新目录
        shutil.copyfile(filepath, newfilepath)
        print('已完成文件转移')



    files = os.listdir(outfolderpath)
    with open(r"{}\csvtimecount.csv".format(folderpath), 'w',newline='') as f:
        writer = csv.writer(f)

        for file in files:
            filepath = os.path.join(outfolderpath, file)
            csvfiles = os.listdir(filepath)
            for csvfile in csvfiles:
                csvpath = os.path.join(filepath, csvfile)
                print(csvpath)
                df = pd.read_csv(csvpath, header=None)
                linenumber = df.shape[0]
                # 写入到 CSV 文件
                writer.writerow([linenumber])

# removeplace(path6)





path7 = r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\场景数量分类-10"


def deleteTAV(path):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        csvfiles = os.listdir(filepath)
        for csvfile in csvfiles:
            csvpath = os.path.join(filepath,csvfile)
            df = pd.read_csv(csvpath, header=None)
            linenumber = df.shape[0]
            if linenumber <= 200 or linenumber >= 600:
                os.remove(csvpath)  # 删除文件
                print(f"文件已删除: {csvpath}")
            else:
                continue


    files = os.listdir(path)
    for file in files:
        folder_path = os.path.join(path, file)  # 文件夹路径
        csvfiles = os.listdir(folder_path)
        for csvfile in csvfiles:
            csvpath = os.path.join(folder_path, csvfile)  # CSV 文件路径
            csvdata = np.loadtxt(csvpath, delimiter=',')
            a, b = csvdata.shape
            number = b // 12
            for i in range(number):
                speedata = csvdata[:, i * 12 + 4]
                max_speed = np.max(speedata)
                min_speed = np.min(speedata)
                if max_speed >= 25 or min_speed <= 1.5:
                    os.remove(csvpath)
                    print('已删除', csvpath)
                    break

                else:
                    continue



# deleteTAV(path7)


# 根据大数据分析一下速度的分布

def Ansysvelocityaga(path):
    files = os.listdir(path)
    with open(r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\Sanjuanisland-speedaga.csv"
            , 'w', newline='') as f:
        writer = csv.writer(f)
        for file in files:
            filepath = os.path.join(path, file)
            csvfiles = os.listdir(filepath)
            for csvfile in csvfiles:
                csvpath = os.path.join(filepath, csvfile)
                print(csvpath)
                with open(csvpath, 'r') as f:
                    csvlist = csv.reader(f)
                    csvlist = list(csvlist)
                    num_boats = len(csvlist[0]) // 12  # 每艘船舶有 12 列数据
                    # 遍历每艘船舶
                    for boat_index in range(num_boats):
                        # 获取船舶速度列的索引（每艘船舶的速度是第 4 列开始偏移的第一个值）
                        speed_column_index = boat_index * 12 + 4

                        # 获取速度值（只需统计第一行的速度）
                        speed_value = csvlist[0][speed_column_index]

                        # 记录速度结果，写入到输出文件
                        writer.writerow([speed_value])

# Ansysvelocityaga(path7)



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



def recreat(path):

    # 提取目录路径（即去掉文件名部分）
    folderpath = os.path.dirname(path)
    foldername = os.path.basename(folderpath)
    newfolderpath = r'{}\速度航向修正-11'.format(folderpath)
    if not os.path.exists(newfolderpath):
        os.makedirs(newfolderpath)
        print('{}文件夹已经创建'.format(newfolderpath))

    newfolderpath2 = r'{}\目的地特征-12'.format(folderpath)
    if not os.path.exists(newfolderpath2):
        os.makedirs(newfolderpath2)
        print('{}文件夹已经创建'.format(newfolderpath2))

    csvlist = os.listdir(path)
    for sv in csvlist:
        csvpath = os.path.join(path, sv)
        newpath = os.path.join(newfolderpath, sv)

        if not os.path.exists(newpath):
            os.makedirs(newpath)
            print('已完成文件夹创建')

        filelists = os.listdir(csvpath)
        for file in filelists:
            filepath = os.path.join(csvpath,file)
            newfilepath = os.path.join(newpath,file)
            print(newfilepath)

            readerdata = np.loadtxt(filepath, delimiter=',')
            a, b = readerdata.shape

            number = b//12
            datalist = []
            for i in range(number):
                time_list = readerdata[:, (i * 12) + 0].reshape(-1, 1)
                mmsi_list = readerdata[:, (i * 12) + 1].reshape(-1, 1)
                lat_list = readerdata[:, (i * 12) + 2].reshape(-1, 1)
                lon_list = readerdata[:, (i * 12) + 3].reshape(-1, 1)
                sog_list = readerdata[:, (i * 12) + 4].reshape(-1, 1)
                cogsin_list = readerdata[:, (i * 12) + 5].reshape(-1, 1)
                coscos_list = readerdata[:, (i * 12) + 6].reshape(-1, 1)
                headsin_list = readerdata[:, (i * 12) + 7].reshape(-1, 1)
                headcos_list = readerdata[:, (i * 12) + 8].reshape(-1, 1)
                type = readerdata[:, (i * 12) + 9].reshape(-1, 1)
                length = readerdata[:, (i * 12) + 10].reshape(-1, 1)
                width = readerdata[:, (i * 12) + 11].reshape(-1, 1)

                pre_lat = lat_list[0]
                pre_lon = lon_list[0]
                speed_list = []
                anglesin_list = []
                anglecos_list = []

                for j in range(a - 1):
                    new_lat = lat_list[j + 1]
                    new_lon = lon_list[j + 1]
                    distance = haversine_distance(pre_lon, pre_lat, new_lon, new_lat)
                    angle = calculate_angle(pre_lat, pre_lon, new_lat, new_lon)
                    speed = (distance / 1) * 3.6 / 1.852

                    # 将角度转换为弧度并计算sin和cos值
                    angle_rad = np.radians(angle)
                    anglesin = np.sin(angle_rad)
                    anglecos = np.cos(angle_rad)

                    speed_list.append(speed)
                    anglesin_list.append(anglesin)
                    anglecos_list.append(anglecos)

                    pre_lat = new_lat
                    pre_lon = new_lon

                # 处理最后一个元素
                speed_list.append(speed_list[-1])
                anglesin_list.append(anglesin_list[-1])
                anglecos_list.append(anglecos_list[-1])

                # 转换为numpy数组并重塑
                speed_list = np.array(speed_list).reshape(-1, 1)
                anglesin_list = np.array(anglesin_list).reshape(-1, 1)
                anglecos_list = np.array(anglecos_list).reshape(-1, 1)

                # 组合所有数据（现在是14列）
                shipdata = np.hstack((time_list, mmsi_list, lat_list, lon_list,
                                      speed_list, anglesin_list, anglecos_list,sog_list,
                                      cogsin_list, coscos_list, headsin_list, headcos_list,
                                      type, length, width))
                datalist.append(shipdata)

            datalist = np.concatenate(datalist, axis=1)
            with open(newfilepath, 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(datalist)




# 将原本的航向角转为余弦和正弦方向的角度，更为平滑  同时将船舶与目的地之间的距离加入特征
    files = os.listdir(newfolderpath)
    for file in files:
        filepath = os.path.join(newfolderpath, file)
        newfilepath = os.path.join(newfolderpath2, file)

        if not os.path.exists(newfilepath):
            os.makedirs(newfilepath)
            print('已完成文件夹创建')

        csvlists = os.listdir(filepath)
        for csvlist in csvlists:
            csvpath = os.path.join(filepath, csvlist)
            newcsvpath = os.path.join(newfilepath, csvlist)
            print(newcsvpath)
            csvdata = np.loadtxt(csvpath, delimiter=',')
            a, b = csvdata.shape
            number = b // 15
            scene_data = []
            for i in range(number):
                data = csvdata[:, 15 * i: 15 * (i + 1)]
                motiondata = data[:, :15]
                posdata = data[:, 2:4]
                posdata = normalize_distance(posdata)
                data = np.hstack([motiondata, posdata])
                scene_data.append(data)
            scene_data = np.hstack(scene_data)
            with open(newcsvpath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(scene_data)


# recreat(path7)



























