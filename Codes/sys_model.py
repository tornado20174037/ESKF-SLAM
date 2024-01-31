# -*- coding: utf-8 -*-
import numpy as np
from quaternion import Quaternion, skew_operator
from quaternion import rot_vec_to_rot_mat, rot_mat_to_quaternion
from quaternion import euler_to_quaternion


class LOCA3D_system:
    def __init__(self, kParams):

        # Position
        x = kParams['x']
        y = kParams['y']
        z = kParams['z']

        # Vitesse
        vx = kParams['vx']
        vy = kParams['vy']
        vz = kParams['vz']

        # Attitude
        qw = kParams['qw']
        qx = kParams['qx']
        qy = kParams['qy']
        qz = kParams['qz']

        # Acceleration bias
        accbx = kParams['accbx']
        accby = kParams['accby']
        accbz = kParams['accbz']

        # Angular velocity bias
        omegabx = kParams['omegabx']
        omegaby = kParams['omegaby']
        omegabz = kParams['omegabz']

        # Gravity
        gravx = kParams['gravx']
        gravy = kParams['gravy']
        gravz = kParams['gravz']

        # Noise covariance
        self.sigma_acc = kParams['sigma_acc']
        self.sigma_omega = kParams['sigma_omega']
        self.sigma_accb = kParams['sigma_accb']
        self.sigma_omegab = kParams['sigma_omegab']

        self.state = np.array([x, y, z,
                               vx, vy, vz,
                               qw, qx, qy, qz,
                               accbx, accby, accbz,
                               omegabx, omegaby, omegabz,
                               gravx, gravy, gravz], dtype="float64")

        self.quaternion = Quaternion(qw, qx, qy, qz)

        self.error_state = np.zeros(18, dtype="float64")
        self.P = np.eye(18, dtype="float64")*0.01


    def get_Fx(self, um, delta):
        am = um[:3]
        ab = self.state[10:13]
        omegam = um[3:]
        omegab = self.state[13:16]
        I_delta = delta * np.eye(3)
        R = self.quaternion.to_rotation_matrix()
        bloc12 = -R.dot(skew_operator(am-ab)) * delta
        bloc13 = -R * delta
        bloc22 = rot_vec_to_rot_mat((omegam-omegab) * delta).T

        L1 = np.hstack([np.eye(3), I_delta, np.zeros((3, 12))])
        L2 = np.hstack([np.zeros((3, 3)),
                        np.eye(3),
                        bloc12,
                        bloc13,
                        np.zeros((3, 3)),
                        I_delta])
        L3 = np.hstack([np.zeros((3, 6)),
                        bloc22,
                        np.zeros((3, 3)),
                        -I_delta,
                        np.zeros((3, 3))])
        L4 = np.hstack([np.zeros((3, 9)), np.eye(3), np.zeros((3, 6))])
        L5 = np.hstack([np.zeros((3, 12)), np.eye(3), np.zeros((3, 3))])
        L6 = np.hstack([np.zeros((3, 15)), np.eye(3)])

        Fx = np.vstack([L1, L2, L3, L4, L5, L6])

        return Fx


    def get_Fi(self):
        line_1 = np.zeros((3, 12))
        line_2 = np.hstack([np.eye(3), np.zeros((3, 9))])
        line_3 = np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 6))])
        line_4 = np.hstack([np.zeros((3, 6)), np.eye(3), np.zeros((3, 3))])
        line_5 = np.hstack([np.zeros((3, 9)), np.eye(3)])
        line_6 = np.zeros((3, 12))
        return np.vstack([line_1, line_2, line_3, line_4, line_5, line_6])


    def get_Qi(self, delta):
        Vi = self.sigma_acc * self.sigma_acc * delta * delta * np.eye(3)
        Thi = self.sigma_omega * self.sigma_omega * delta * delta * np.eye(3)
        Ai = self.sigma_accb * self.sigma_accb * delta * np.eye(3)
        Omi = self.sigma_omegab * self.sigma_omegab * delta * np.eye(3)

        Vi = np.hstack([Vi, np.zeros((3, 9))])
        Thi = np.hstack([np.zeros((3, 3)), Thi, np.zeros((3, 6))])
        Ai = np.hstack([np.zeros((3, 6)), Ai, np.zeros((3, 3))])
        Omi = np.hstack([np.zeros((3, 9)), Omi])

        Qi = np.vstack([Vi, Thi, Ai, Omi])

        return Qi


    def get_Hx(self):
        line_1 = np.hstack([np.eye(3), np.zeros((3, 16))])
        line_2 = np.hstack([np.zeros((4, 6)), np.eye(4), np.zeros((4, 9))])
        return np.vstack([line_1, line_2])


    def get_X_deltax(self):
        mat = np.array([[-self.state[7], -self.state[8], -self.state[9]],
                        [self.state[6], -self.state[9], self.state[8]],
                        [self.state[9], self.state[6], -self.state[7]],
                        [-self.state[8], self.state[7], self.state[6]]])
        Q_delta_theta = 0.5 * mat
        line_1 = np.hstack([np.eye(6), np.zeros((6, 12))])
        line_2 = np.hstack([np.zeros((4, 6)), Q_delta_theta, np.zeros((4, 9))])
        line_3 = np.hstack([np.zeros((9, 9)), np.eye(9)])
        return np.vstack([line_1, line_2, line_3])


    def get_H(self):
        Hx = self.get_Hx()
        X_deltax = self.get_X_deltax()
        return Hx.dot(X_deltax)


    def get_h(self):
        position = self.state[:3]
        orientation = self.quaternion.quaternion_vector
        return np.hstack([position, orientation]).T


    def nom_state_prediction(self, um, delta):
        p = self.state[:3]
        v = self.state[3:6]
        am = um[:3]
        ab = self.state[10:13]
        omegam = um[3:]
        omegab = self.state[13:16]
        g = self.state[16:]
        R = self.quaternion.to_rotation_matrix()

        global_acc = R.dot(am-ab)+g
        print("global acc : ", global_acc)
        p += v * delta + 0.5 * global_acc * delta * delta
        v += global_acc*delta
        rotation = euler_to_quaternion((omegam-omegab) * delta)
        self.quaternion *= rotation


    def error_state_prediction(self, um, delta):
        Fx = self.get_Fx(um, delta)
        Fi = self.get_Fi()
        Qi = self.get_Qi(delta)
        # delta_x = Fx.dot(delta_x) # Useless, delta_x always 0
        self.P = Fx.dot(self.P).dot(Fx.T) + Fi.dot(Qi).dot(Fi.T)


    def prediction(self, um, delta):
        self.nom_state_prediction(um, delta)
        self.error_state_prediction(um, delta)


    def correction(self, measure):
        H = self.get_H()
        V = np.eye(7) # self.slam_covar_mat
        K = self.P.dot(H.T).dot(np.linalg.inv(H.dot(self.P).dot(H.T) + V))
        self.error_state = K.dot(np.transpose(measure - self.get_h())).flatten()
        self.P -= K.dot(H).dot(self.P)


    def error_injection(self):
        self.state[:3] += self.error_state[:3] # p
        self.state[3:6] += self.error_state[3:6] # v
        self.quaternion *= euler_to_quaternion(self.error_state[6:9]) # q
        self.state[10:13] += self.error_state[9:12] # accb
        self.state[13:16] += self.error_state[12:15] # omegab
        self.state[16:19] += self.error_state[15:18] # gravity


    def ESKF_reset(self):
        self.error_state = np.zeros(18, dtype="float64")
        # P = G.dot(P).dot(G.T)


if __name__ == "__main__":

    initial_state = {'x': 0, 'y': 0, 'z': 0,
                     'vx': 0, 'vy': 0, 'vz': 30,
                     'qw': 1, 'qx': 0, 'qy': 0, 'qz': 0,
                     'accbx': 0, 'accby': 0, 'accbz': 0,
                     'omegabx': 0, 'omegaby': 0, 'omegabz': 0,
                     'gravx': 0, 'gravy': 0, 'gravz': -9.81,
                     'sigma_acc': 0.1, 'sigma_omega': 0.1,
                     'sigma_accb': 0.1, 'sigma_omegab': 0.1}

    test = LOCA3D_system(initial_state)

    Fi = test.get_Fi()
    print(Fi, Fi.shape)
    Qi = test.get_Qi(32.0)
    print(Qi, Qi.shape)
    Fx = test.get_Fx((0, 0, 0, 0, 0, 0), 32.0)
    print(Fx, Fx.shape)

    prod = Fi.dot(Qi).dot(Fi.T)
    print(prod, prod.shape)

    print("Error state covar mat", test.P)
    for i in range(20):
        test.prediction((0, 0, 0, 0, 0, 0), 32.0)
    print("Error state covar mat", test.P)

    print(test.get_X_deltax().shape)

    test.correction(np.ones(7))
    test.error_injection()
    test.ESKF_reset()
