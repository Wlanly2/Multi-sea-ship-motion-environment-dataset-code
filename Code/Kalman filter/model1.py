import os
import numpy as np
import matplotlib.pyplot as plt




m_var1 = 0.015
v_var1 = 0.002

def Kalman1(a, data1,data2):


    t = np.linspace(1, a, a)  # 在1~100s内采样100次

    real_positions = [0] * a
    real_positions[0] = data1[0]

    measure_positions = [0] * a
    measure_positions[0] = data2[0]

    optim_positions = [0] * a
    optim_positions[0] = data1[0]

    predict_var = 0
    real_data = data1[0]
    optim_data = real_data

    for i in range(a-1):

        real_positions[i+1] = data1[i+1]

        measure_data = data2[i+1]
        measure_positions[i+1] =  measure_data

        # 以下是卡尔曼滤波的整个过程
        # 更新预测数据的方差
        predict_var += v_var1

        K = predict_var/(predict_var+m_var1)

        # 求得最优估计值
        optim_data = optim_data + K*(measure_data-optim_data)
        optim_positions[i+1] = optim_data

        # 更新
        predict_var = (1-K)*predict_var



    # plt.plot(t, real_positions, label='real positions')
    # plt.plot(t, measure_positions, label='measured positions')
    # plt.plot(t, optim_positions, label='kalman filtered positions')
    # # 预测噪声比测量噪声低，但是运动模型预测值比观测值差很多，原因是在于运动模型是基于前一刻预测结果进行下一次的预测，而测量噪声是基于当前位置给出的测量结果
    # # 意思就是，运动模型会积累噪声，而观测结果只是单次噪声
    # plt.legend()
    # plt.show()






















