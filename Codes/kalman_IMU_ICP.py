# -*- coding: utf-8 -*-

import copy as cp
import numpy as np
import matplotlib.pylab as plt

from sys_model import LOCA3D_system
from sequence_reader import Sequence_reader
from quaternion import Quaternion, euler_to_quaternion, skew_operator

import matplotlib.pyplot as plt
def find_closest_icp_timestamp(icp_data, imu_timestamp):
    """
        Find the index of the closest ICP timestamp to a given IMU timestamp.

        Parameters:
        icp_data (numpy.ndarray): The array containing ICP data, with timestamps in the first column.
        imu_timestamp (float): The IMU timestamp for which the closest ICP timestamp is to be found.

        Returns:
        int: The index of the closest ICP timestamp in the icp_data array.
        """
    # Extract the timestamps from the first column of ICP data
    icp_timestamps = icp_data[:, 0]
    # Find the index of the closest timestamp in ICP data to the given IMU timestamp.
    # This is achieved by computing the absolute difference between the IMU timestamp
    # and all ICP timestamps, and then finding the index of the minimum value.
    index = np.abs(icp_timestamps - imu_timestamp).argmin()
    return index

def main():
    seq = Sequence_reader()
    print("nbr timestamps IMU: ", len(seq.imu_data))
    print("nbr timestamps ICP: ", len(seq.icp_data))
    print("1st timestamps IMU: ", seq.imu_data[0, 0], " ms")
    icp_1st_tstp = 0
    print("1st timestamps ICP: ", seq.icp_data[icp_1st_tstp, 0], " ms")

    initial_state = {'x': 0, 'y': 0, 'z': 0,
                     'vx': 0, 'vy': 0, 'vz': 0,
                     'qw': 1, 'qx': 0, 'qy': 0, 'qz': 0,
                     'accbx': 0, 'accby': 0, 'accbz': 0,
                     'omegabx': 0, 'omegaby': 0, 'omegabz': 0,
                     'gravx': 0, 'gravy': 0, 'gravz': -9.81,
                     'sigma_acc': 0.1, 'sigma_omega': 0.1,
                     'sigma_accb': 0.1, 'sigma_omegab': 0.1}

    xy_imu = []

    system_model = LOCA3D_system(initial_state)
    previous_t = float( seq.imu_data[0,0])
    seq.imu_data = seq.imu_data[0:]
    seq.imu_data[:, 1:4] *= 9.81
    seq.imu_data[:, 4:] *= np.pi/180
    print(seq.imu_data)


    # MOD - FOR VISUALISATION

    actual_traj = np.zeros((len(seq.imu_data)-1, 3))
    #estimated_traj = np.zeros((len(seq.imu_data)-1, 3))

    for i in range(len(seq.imu_data)-2):
    #for i in range(len(seq.icp_data) - 2):
        # Get the current timestamp from the IMU data
        current_t = float(seq.imu_data[i, 0]) # Calculate the time difference since the last IMU reading in seconds
        delta = (current_t - previous_t)/1000
        print("timestamp: ", seq.imu_data[i, 1:4])
        print("rotation: ", seq.imu_data[i, 4:7])
        # PREDICTION Use the IMU data to predict the next state
        system_model.prediction(seq.imu_data[i, 1:], delta)
        previous_t = current_t # Update the previous timestamp to the current one
        print()
        print("iter number: ", i)
        print("===============")
        print("x, y, z: ", system_model.state[:3])
        print("v: ", system_model.state[3:6])
        print("heading: ", system_model.quaternion.to_euler_angles())
        print("acceleration: ", np.linalg.norm(seq.imu_data[i+1, 1:4]))
        #  input()
        xy_imu.append(cp.deepcopy(system_model.state[:3]))
        # Find the index of the closest ICP timestamp to the current IMU timestamp
        icp_index = find_closest_icp_timestamp(seq.icp_data, current_t)

        # Prepare the ICP data for correction
        temp = np.zeros((1, 7))
        temp[0, 0:3] = seq.icp_data[icp_index, 1:4]
        q_temp = euler_to_quaternion(seq.icp_data[icp_index, 4:7])
        temp[0, 3:7] = q_temp.quaternion_vector

        # CORRECTION  Correct the predicted state with the ICP data
        system_model.correction(temp)
        # Apply error injection and reset the ESKF
        system_model.error_injection()
        system_model.ESKF_reset()


        # MOD - STORE RESULT Save the current state for later analysis or visualization
        actual_traj[i, :3] = system_model.state[:3]

    # MOD - VISUALIZE
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    x,y,z = actual_traj[:,0],actual_traj[:,1],actual_traj[:,2]
    ax.plot3D(x, y, z, 'green')
    ax.set_title('Trajecotry with ESKF')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    x,y,z = seq.icp_data[:,1],seq.icp_data[:,2],seq.icp_data[:,3]
    ax.plot3D(x, y, z, 'blue')
    ax.set_title('Sesnor readout trajecotory')
    plt.show()

    #plot the comparsion
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x, y, z = actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2]
    ax.plot3D(x, y, z, 'green')
    x, y, z = seq.icp_data[:, 1], seq.icp_data[:, 2], seq.icp_data[:, 3]
    ax.plot3D(x, y, z, 'blue')
    ax.set_title('Compare trajecotory')
    plt.show()


if __name__ == "__main__":
    main()

