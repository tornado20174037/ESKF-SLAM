# coding: utf-8
import numpy as np
import csv


# LiDAR: time(ms) X(m) Y(m) Z(m)
#                   roll(rad) pitch(rad) yaw(rad) computeTimeICP(s)
# IMU: time(ms) AccX(g) AccY(g) AccZ(g) GyroX(deg/s) GyroY(deg/s) GyroZ(deg/s)

class Sequence_reader:
    def __init__(self,
                 trajectoIMU_path="IMU3.txt",
                 trajectoLiDAR_path="trajecto.txt"):

        with open(trajectoIMU_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter="	")

            imu_data = []

            for row in reader:
                imu_data.append(row)

            self.imu_data = np.array(imu_data[1:], dtype='float64')

        with open(trajectoLiDAR_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter="	")

            icp_data = []

            for row in reader:
                icp_data.append(row)

            self.icp_data = np.array(icp_data, dtype='float64')
