import csv
import datetime
import shutil
import numpy as np
import os
from scipy.interpolate import interp1d
import pandas as pd


def ct(a):# 将带有T的字符串转为时间格式,且直接用datetime.time模块内置的方法，得到年、月、日、时、分、秒
    dd = datetime.datetime.strptime(a, "%Y-%m-%dT%H:%M:%S")
    time_in_seconds = dd.timestamp()  # 将时间对象转换为时间戳（秒数）
    return int(time_in_seconds)

def tc(timestamp):
    # 将时间戳转换为datetime对象
    dt = datetime.datetime.fromtimestamp(timestamp)
    # 格式化datetime对象为指定格式的时间字符串
    time_string = dt.strftime("%Y-%m-%dT%H:%M:%S")
    return time_string


#第一步：将轨迹区分开来,按照船舶的识别码分类
path1=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\AIS数据原始数据1.csv"
#将从网站上下载的AIS原文件按照呼号分类
def choose(path):
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

            alist.append((a[1],a[0], float(a[2]), float(a[3]), float(a[4]), float(a[5]),
                          a[10],a[12],a[13]))
            for a2 in rows[i + 1:]:
                if value == a2[0] and float(a2[4])>1:

                    alist.append((a2[1], a2[0], float(a2[2]), float(a2[3]), float(a2[4]), float(a2[5]),
                                  a2[10],a2[12],a2[13]))
            with open(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\1-呼号分类\{}.csv"
                              .format(b), 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(alist)
            aname.append(value)
    print(len(aname))
# choose(path1)





#第二步，检查csv文件中的缺失情况,对齐进行填充
path2=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\1-呼号分类"
def read_and_modify_column(path):
    files  = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        output_file = os.path.join(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\2-缺失补全"
                                   ,file)
        # 读取CSV文件，没有表头
        df = pd.read_csv(filepath, header=None)

        for column_index in range(3):
            column_index = column_index + 6

            # 检查该列是否全是空值
            if df[column_index].isnull().all():
                df[column_index] = df[column_index].fillna(0)  # 使用指定值替换空值
            elif df[column_index].isnull().any():
                # 查找该列的最大值
                max_value = df[column_index].max()
                df[column_index] = df[column_index].fillna(max_value)
                print('不全')
                print(filepath)
            else:
                continue

        # 保存到新文件
        df.to_csv(output_file, index=False, header=False)

# read_and_modify_column(path2)






#第三步 直接分类处理的AIS分类数据中有的数据时间顺序混乱，不能直接使用，需要重新进行排序。
path3 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\2-缺失补全"
def sortime(path):
    files = os.listdir(path)
    for i in files:
        filepath = os.path.join(path,i)
        newpath = os.path.join(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\3-时间排序"
                               ,i)
        # 读取CSV文件
        with open(filepath,'r') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

            # 确保BaseDateTime列是日期时间格式，并添加一个排序键
            for row in data:
                row[0] = ct(row[0])

        # 根据BaseDateTime列进行排序
        data_sorted = sorted(data, key=lambda x: x[0])

        # 写入排序后的数据到新的CSV文件
        with open(newpath, 'w', encoding='utf-8',newline='') as csvfile:
            writer = csv.writer(csvfile)

            for row in data_sorted:
                writer.writerow(row)

# sortime(path3)





#第四步：由于这是两个月内的数据，难免有重复的船舶经过同一区域，
# 需要分析AIS数据一般的时间间隔是多少，确定划分的时间间隔。
#分析一下到底该采用什么样的AIS数据间隔，看一下大数据分布
path4=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\3-时间排序"

def Ansystime(path):
    files = os.listdir(path)
    with open(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\时间间隔.csv"
            , 'w', newline='') as f:
        writer = csv.writer(f)
        for file in files:
            csvfile = os.path.join(path, file)
            with open(csvfile, 'r') as f:
                csvlist = csv.reader(f)
                csvlist = list(csvlist)
            a = len(csvlist)
            for i in range(a - 1):
                time1 = int(csvlist[i][0])
                time2 = int(csvlist[i + 1][0])
                value = time2-time1
                writer.writerow([value])
# Ansystime(path4)



#第五步：删除其中大于1000的数据,1000秒的时间间隔对于AIS数据的时间间隔来说时间太长
#在间隔内的动作都未知
path5=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\时间间隔.csv"

def Removedata(path,threshold):

    df = pd.read_csv(path, header=None)  # header=None表示没有表头

    filtered_df = df[df[0] <= threshold]


    filtered_df.to_csv("D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\时间阈值.csv"
                       , index=False, header=False)

# Removedata(path5,300)



#第六步：从上一部的数据中随机选择10000个进行分析
path6=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\时间阈值.csv"
def Randomchoose(path,num):
    # 读取CSV文件
    df = pd.read_csv(path, header=None)  # header=None表示没有表头
    # 检查样本数是否大于可用数据行数
    if num > len(df):
        raise ValueError("样本数大于可用数据行数")

    # 随机选择指定数量的数据行
    sample_df = df.sample(n=num)

    # 保存到新文件
    sample_df.to_csv(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\10000nums.csv"
                     , index=False, header=False)

# Randomchoose(path6,10000)








#分析后、可以看出设置为最大阈值设置为200秒进行选择是比较最好的
#第七步：由于这是两个月内的数据，难免有重复的船舶经过同一区域，
#根据已经分析得到的时间，将AIS文件切割成不同文件,同时除去AIS中重复的数据
path7=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\3-时间排序"

def split(path):
    files = os.listdir(path)
    for a,file in enumerate(files):
        csvfile = os.path.join(path, file)
        with open(csvfile, 'r') as f:
            csvlist=csv.reader(f)
            csvlist=list(csvlist)
        data=[]
        b=1
        for i in csvlist:
            if len(data) == 0:
                data.append(i)
            else:
                if data[-1][0] != i[0]:#时间不同才读入
                    if int(i[0])-int(data[-1][0]) >0 and int(i[0])-int(data[-1][0])<=220:
                        data.append(i)
                    else:
                        with open(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\4-时间切片\{}.{}.csv"
                                          .format(a+1, b), 'w',  newline='') as f:
                            writer = csv.writer(f)
                            writer.writerows(data)
                        b = b + 1
                        data = []
                        data.append(i)
                else:#时间相同不读入
                    continue
        with open(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\4-时间切片\{}.{}.csv"
                          .format(a+1,b), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

# split(path7)











#第八步，根据大数据分析一下速度的分布
path8=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\3-时间切片"

def Ansysvelocity(path):
    files = os.listdir(path)
    with open(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\速度分布.csv"
            , 'w', newline='') as f:
        writer = csv.writer(f)
        for file in files:
            csvfile = os.path.join(path, file)
            with open(csvfile, 'r') as f:
                csvlist = csv.reader(f)
                csvlist = list(csvlist)
            for i in csvlist:
                writer.writerow([i[3]])
# Ansysvelocity(path7)

#第九步从上一部的数据中随机选择50000个进行分析
path9=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\速度分布.csv"
def Randomvelocity(path,num):
    # 读取CSV文件
    df = pd.read_csv(path, header=None)  # header=None表示没有表头
    # 检查样本数是否大于可用数据行数
    if num > len(df):
        raise ValueError("样本数大于可用数据行数")

    # 随机选择指定数量的数据行
    sample_df = df.sample(n=num)

    # 保存到新文件
    sample_df.to_csv(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\10000velocity.csv"
                     , index=False, header=False)

# Randomvelocity(path8,50000)




#第十步，删除速度过快和过慢的数据，在上一步分析中删除速度小于2节和大于30的数据，以及删除数据中少于5行的数据
path10=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\4-时间切片"
#这一步我们需要删除的工作在后面依然需要重复进行,这一步我们在原文件中进行
def sortdelete(path):
    files = os.listdir(path)
    print(len(files))
    for i in files:
        csvfile = os.path.join(path, i)
        with open(csvfile, 'r') as f:
            csvlist = csv.reader(f)
            csvlist = list(csvlist)
        if len(csvlist) < 4:
            os.remove(csvfile)
        else:
            for j in csvlist:
                if float(j[4]) > 30 or float(j[4]) <1:
                    os.remove(csvfile)
                    break

# sortdelete(path10)






#将检查后的数据进行插值填充
def normal(path):
    files = os.listdir(path)
    for a,file in enumerate(files):
        csvfile = os.path.join(path, file)
        b0 = np.loadtxt(csvfile, dtype='int', delimiter=',', usecols=0)  # 取出时间列

        aisdata = np.loadtxt(csvfile, dtype='float', delimiter=',',  usecols=(1, 2, 3, 4, 5, 6,7,8))

        a1 = aisdata[:, 0]
        a2 = aisdata[:, 1]
        a3 = aisdata[:, 2]
        a4 = aisdata[:, 3]
        a5 = aisdata[:, 4]
        a6 = aisdata[:, 5]
        a7 = aisdata[:, 6]
        a8 = aisdata[:, 7]

        numbers = b0[-1] - b0[0] + 1

        f1 = interp1d(b0, a1, kind='linear')  # 线性插值
        f2 = interp1d(b0, a2, kind='cubic')   # 二次插值
        f3 = interp1d(b0, a3, kind='cubic')   # 二次插值
        f4 = interp1d(b0, a4, kind='linear')   # 线性插值
        f5 = interp1d(b0, a5, kind='linear')   # 线性插值
        f6 = interp1d(b0, a6, kind='linear')  # 线性插值
        f7 = interp1d(b0, a7, kind='linear')  # 线性插值
        f8 = interp1d(b0, a8, kind='linear')  # 线性插值


        time = np.linspace(b0[0], b0[-1], num=numbers, endpoint=True)

        mmsi = f1(time)
        lat  = f2(time)
        lon  = f3(time)
        sog  = f4(time)
        cog  = f5(time)
        tye  = f6(time)
        len  = f7(time)
        wid = f8(time)


        time = time.reshape(-1, 1)
        mmsi = mmsi.reshape(-1, 1)
        lat = lat.reshape(-1, 1)
        lon = lon.reshape(-1, 1)
        sog = sog.reshape(-1, 1)
        cog = cog.reshape(-1, 1)
        tye = tye.reshape(-1, 1)
        len = len.reshape(-1, 1)
        wid = wid.reshape(-1, 1)


        alist = np.hstack((time,mmsi, lat, lon, sog, cog, tye, len, wid))

        with open(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\5-插值处理\{}.csv".format(a+1),
                  'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(alist)

# normal(path10)




#这个是用来计算数据中每个特征的最大最小值的
#检查是否存在插值处理后存在的一些错误的数据

path11 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\12-速度航向修正"
def sortmaxmin(path):
    files = os.listdir(path)
    mindata = np.array([1000, 1000, 1000, 1000,1000,1000,1000,]).reshape(1, 7)
    maxdata = np.array([-1000, -1000, -1000, -1000,-1000,-1000,-1000]).reshape(1, 7)
    for file in files:
        csvpath = os.path.join(path, file)
        csvdata = np.loadtxt(csvpath, delimiter=',')
        h, w = csvdata.shape
        number = w // 7
        for i in range(number):
            data = csvdata[:, i*7:(i+1)*7]
            # 计算每个特征的最大值和最小值
            min_vals = np.min(data, axis=0).reshape(1, 7)  # 沿着列方向计算最小值
            mindata = np.minimum(mindata, min_vals).reshape(1, 7)

            max_vals = np.max(data, axis=0).reshape(1, 7)  # 沿着列方向计算最大值
            maxdata = np.maximum(maxdata, max_vals).reshape(1, 7)

    print(mindata)
    print(maxdata)

# sortmaxmin(path)





#第十二步，将插值填充的数据统一合并为同一个csv文件中
path12=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\5-插值处理"
def concat(path):
   # 获取所有 CSV 文件路径
   all_files = [os.path.join(path, f) for f in os.listdir(path)]

   # 用于存储所有数据框的列表
   df_list = []

   # 读取所有 CSV 文件并追加到列表中
   for file in all_files:
       df = pd.read_csv(file, header=None)  # 假设文件没有表头
       df_list.append(df)

   # 合并所有数据框
   combined_df = pd.concat(df_list, ignore_index=True)

   # 保存合并后的数据框到新文件中
   combined_df.to_csv(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\6-数据合并\concat.csv",
                      index=False, header=False)

# concat(path12)






#第十三步，将插值填充的数据得到的数据重新进行排序
path13=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\6-数据合并\concat.csv"
def Resortime(path):
    # 读取 CSV 文件，没有表头
    df = pd.read_csv(path, header=None)

    # 根据第一列（索引为0）进行排序
    df_sorted = df.sort_values(by=[0])

    # 保存排序后的数据到新文件
    df_sorted.to_csv(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\6-数据合并\newconcat.csv",
                     index=False, header=False)


# Resortime(path13)







#第十五步，按照同一时间出现的次数保存同一时刻的轨迹，断点则开始下一轮的保存
#这种选择方式是用来设计多船轨迹预测任务的
path14=r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\6-数据合并\newconcat.csv"

def process(path):

    # 读取 CSV 文件
    df = pd.read_csv(path, header=None)
    lens = len(df)
    b = 1

    # 初始化变量
    range_offset=30
    prename = None
    pretime = None
    buffer = []

    for index, row in df.iterrows():

        currenttime = row[0]

        # 获取行切片，范围为当前行的前后range_offset行
        start_index = max(0, index-range_offset)
        end_index =  min(lens, index + range_offset + 1)
        range_df = df.iloc[start_index:end_index]

        # 根据时间范围搜索整个 CSV 文件中具有相同时间值的行
        matching_rows = range_df[range_df[0] == currenttime]

        # 计算用来判断的特征值
        currenttime = int(currenttime)
        currentname = set(matching_rows[1])


        if pretime ==None and prename==None:
            pretime = currenttime
            prename = currentname
            buffer.append(matching_rows)


        elif pretime == currenttime:
            continue

        elif currenttime > pretime:

            if prename == currentname and (currenttime-pretime)==1:#需要有一个判断是否连续

                pretime = currenttime
                buffer.append(matching_rows)

            else:
                # 将所有匹配的 DataFrame 合并为一个 DataFrame
                combined_df = pd.concat(buffer, ignore_index=True)
                combined_df.to_csv(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\7-同时刻选择\{}.csv".format(b),
                index=False, header=False)

                buffer = [matching_rows]
                pretime = currenttime
                prename = currentname
                b += 1
    if buffer:
        combined_df = pd.concat(buffer, ignore_index=True)
        combined_df.to_csv(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\7-同时刻选择\{}.csv".format(b),
        index=False, header=False)

# process(path14)




#第十七步，将船舶轨迹重新排列，每行包括相同时间的船舶。
path15 =r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\7-同时刻选择"
def combine_rows_by_time(path):
    # 读取 CSV 文件
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        newpath = os.path.join(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\8-重排列",file)

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

# combine_rows_by_time(path15)




#第十六步，按照时间对文件进行挑选
path16 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\8-重排列"
def deletetimeline(path):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        newfilepath = os.path.join("D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\9-筛选时间段",file)
        df = pd.read_csv(filepath, header=None)
        linenumber = df.shape[0]
        if linenumber >= 300 and linenumber<=700:
            shutil.copy(filepath, newfilepath)  # 复制文件到新路径
        else:
            continue

# deletetimeline(path16)





#第xx步，将文件转移位置到不同的文件夹下
def removeplace(path):
    # 读取 CSV 文件
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        df = pd.read_csv(filepath, header=None)
        a,b = df.shape
        number = b//9
        # 复制文件
        newfile = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\10-分类\{}".format(number)
        newfilepath = os.path.join(newfile,file)

        shutil.copyfile(filepath,newfilepath)

# removeplace(path18)




#第xx步按照时间间隔保存数据，这次选10s间隔保存依次,前期的程序写的有点问题，改了改。
path19 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\10-分类"
def intervel(path):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        newfilepath = os.path.join(r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\11-10秒差值",file)
        csvpaths = os.listdir(filepath)
        for j in csvpaths:
            csvpath = os.path.join(filepath,j)
            newcsvpath  =os.path.join(newfilepath,j)
            with open(csvpath, 'r') as f:
                csvlist = csv.reader(f)
                csvlist = list(csvlist)
                prenumber = None
                data = []
                for row in csvlist:
                    currenttime = int(float(row[0]))
                    if prenumber == None:
                        data.append(row)
                        prenumber = currenttime

                    elif currenttime - prenumber == 10:
                        data.append(row)
                        prenumber = currenttime

                    else:
                        continue

                with open(newcsvpath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(data)

# intervel(path19)





#分析场景持续时间、这个是有必要的，要不然删除就不合理
def timecount(path):
    files = os.listdir(path)
    with open(r"C:\Users\wuyuegao\Desktop\csvtimecount.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for file in files:
            csvpath = os.path.join(path, file)
            data = np.loadtxt(csvpath, delimiter=',')
            linedata = data[:, 0].reshape(-1, 1)
            value = [int(linedata[-1][0] - linedata[0][0])]
            writer.writerow(value)

# timecount(path18)




#第xx步，再一次的根据条件的不同而进行文件删除
path20 = r"D:\论文工作\期刊文章\端到端运动规划\地形图片及数据\轨迹数据\数据1\10秒差值"
def deletetimeline2(path):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        csvfiles = os.listdir(filepath)
        for csvfile in csvfiles:
            csvpath = os.path.join(filepath,csvfile)
            df = pd.read_csv(csvpath, header=None)
            linenumber = df.shape[0]


            if linenumber > 25 and linenumber < 65:
                continue
            else:
                os.remove(csvpath)
                print('删除')

# deletetimeline2(path20)












