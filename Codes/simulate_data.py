# -*- coding: utf-8 -*-



import numpy as np
from quaternion import Quaternion


def generate_path(Ax, Ay, Az, Bx, By, Bz,
                  ARx, ARy, ARz, BRx, BRy, BRz):
    time = np.arange(100000)
    x = time * Ax + Bx
    y = time * Ay + By
    z = time * Az + Bz
    Rx = time * ARx + BRx
    Ry = time * ARy + BRy
    Rz = time * ARz + BRz

    path = np.vstack([time, x, y, z, Rx, Ry, Rz])
    path = path.T

    return path


def compute_sensors_data(state):
    previous_log = state[0]
    for log in range(1, len(state)):
        delta_t = log[0]-previous_log[0]
        acc_x = log[1] / delta_t
        acc_y = log[2] / delta_t

def main():
    data = generate_path(0, 5, 0, 0, 0, 0,
                         2, 0, 0, 0, 0, 0)
    print(data)


if __name__ == "__main__":
    main()
