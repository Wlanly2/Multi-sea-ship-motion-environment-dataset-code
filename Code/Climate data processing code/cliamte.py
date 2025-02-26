import pandas as pd
import csv
import numpy as np


def process1():

    input_csv_file = r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\climate.csv"

    df = pd.read_csv(input_csv_file)

    df['valid'] = pd.to_datetime(df['valid'], format="%Y-%m-%d %H:%M", errors='coerce')

    if df['valid'].isnull().any():
        print("存在无法解析的日期，已将这些行设为 NaT:")
        print(df[df['valid'].isnull()])



    df['timestamp'] = df['valid'].apply(lambda x: int(x.timestamp()) if pd.notnull(x) else None)


    output_df = df[['timestamp', 'drct','sknt',  'vsby']]

    # Output CSV file path
    output_csv_file = r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanjuanisland\extracted_data.csv"

    output_df.to_csv(output_csv_file, index=False)

    print("数据已成功保存至:", output_csv_file)


# process1()

def process2():

    input_file = r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanfrancisco\extracted_data.csv"

    df = pd.read_csv(input_file)


    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # 检查风向、风速和能见度的范围
    # 检查风向 (0-360)
    invalid_drct = df[(df['drct'] < 0) | (df['drct'] > 360)]
    if not invalid_drct.empty:
        print("风向值超出范围 (0-360)：")
        print(invalid_drct)

    # 将异常风向值替换为缺失值 NaN
    df.loc[(df['drct'] < 0) | (df['drct'] > 360), 'drct'] = np.nan

    # 检查风速 (假设风速 < 150 knots 为合理)
    invalid_sknt = df[df['sknt'] > 50]
    if not invalid_sknt.empty:
        print("风速值超出正常范围 (<150 knots)：")
        print(invalid_sknt)

    # 将异常风速值替换为缺失值 NaN
    df.loc[df['sknt'] > 150, 'sknt'] = np.nan

    # 检查能见度 (假设范围在 0-10)
    invalid_vsby = df[(df['vsby'] < 0) | (df['vsby'] > 20)]
    if not invalid_vsby.empty:
        print("能见度值超出范围 (0-20)：")
        print(invalid_vsby)

    # 将异常能见度值替换为缺失值 NaN
    df.loc[(df['vsby'] < 0) | (df['vsby'] > 10), 'vsby'] = np.nan

    # 插值处理
    # 在缺失列中进行线性插值
    df['drct'] = df['drct'].interpolate(method='linear', limit_direction='both')  # 插值风向
    df['sknt'] = df['sknt'].interpolate(method='linear', limit_direction='both')  # 插值风速
    df['vsby'] = df['vsby'].interpolate(method='linear', limit_direction='both')  # 插值能见度

    #检查是否仍存在 NaN（插值可能仍有无法处理的点）
    if df.isnull().any().any():  # 检查是否有剩余的 NaN
        print("仍然存在缺失值，请检查数据：")
        print(df[df.isnull().any(axis=1)])

    #插值为每秒的数据
    # 生成完整的秒级时间序列
    df = df.set_index('datetime')  # 将 datetime 设置为索引
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='s')  # 每秒时间序列
    df = df.reindex(full_index)  # 填充缺失的时间
    df['timestamp'] = df.index.astype(np.int64) // 10 ** 9  # 恢复时间戳列

    # 对新行插值
    df['drct'] = df['drct'].interpolate(method='linear', limit_direction='both')
    df['sknt'] = df['sknt'].interpolate(method='linear', limit_direction='both')
    df['vsby'] = df['vsby'].interpolate(method='linear', limit_direction='both')

    # 6. 保存数据到新的 CSV
    output_file = r"D:\论文工作\期刊文章\端到端运动规划\第二版\海域及周边环境信息\Sanfrancisco\interpolate_data.csv"
    df.reset_index(drop=True)[['timestamp', 'drct', 'sknt', 'vsby']].to_csv(output_file, index=False)
    print(f"数据已保存至 {output_file}")


# process2()