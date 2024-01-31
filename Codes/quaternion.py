# -*- coding: utf-8 -*-
import numpy as np


class Quaternion:
    def __init__(self, *args, precision='float64'):
        if len(args) == 4:
            self.quaternion_vector = np.array([args[0],
                                               args[1],
                                               args[2],
                                               args[3]], dtype=precision)
        elif isinstance(args[0], np.ndarray):
            self.quaternion_vector = np.array(args[0], dtype=precision)

    def __add__(self, quat_b):
        return Quaternion(self.quaternion_vector + quat_b.quaternion_vector)

    def __sub__(self, quat_b):
        return Quaternion(self.quaternion_vector - quat_b.quaternion_vector)

    def __mul__(self, quat_b):
        p = self.quaternion_vector
        q = quat_b.quaternion_vector
        prod = np.array([p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
                         p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
                         p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
                         p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]])
        return Quaternion(prod)

    def __truediv__(self, scalar):
        q = self.quaternion_vector
        return Quaternion(q/scalar)

    def left_quat_prod(self):
        q = self.quaternion_vector
        mat = np.array([[q[0], -q[1], -q[2], -q[3]],
                        [q[1],  q[0], -q[3],  q[2]],
                        [q[2],  q[3],  q[0], -q[1]],
                        [q[3], -q[2],  q[1],  q[0]]])
        return mat

    def right_quat_prod(self):
        q = self.quaternion_vector
        mat = np.array([[q[0], -q[1], -q[2], -q[3]],
                        [q[1],  q[0],  q[3], -q[2]],
                        [q[2], -q[3],  q[0],  q[1]],
                        [q[3],  q[2], -q[1],  q[0]]])
        return mat

    def squared_norm(self):
        q = self.quaternion_vector
        return q[0]**2+q[1]**2+q[2]**2+q[3]**2

    def norm(self):
        return np.sqrt(self.squared_norm())

    def conjugate(self):
        q = self.quaternion_vector
        conj = np.array([q[0], -q[1], -q[2], -q[3]])
        return Quaternion(conj)

    def invert(self):
        return self.conjugate()/self.squared_norm()

    def to_rotation_matrix(self):
        a = self.quaternion_vector[0]
        b = self.quaternion_vector[1]
        c = self.quaternion_vector[2]
        d = self.quaternion_vector[3]
        return np.array([[a*a + b*b - c*c - d*d,
                          2*(b*c + a*d),
                          2*(b*d + a*c)],
                         [2*(b*c - a*d),
                          a*a - b*b + c*c - d*d,
                          2*(c*d + a*b)],
                         [2*(b*d - a*c),
                          2*(c*d - a*b),
                          a*a - b*b - c*c + d*d]])

    def to_euler_angles(self):
        a = self.quaternion_vector[0]
        b = self.quaternion_vector[1]
        c = self.quaternion_vector[2]
        d = self.quaternion_vector[3]
        phi = np.arctan2(2*c*d + 2*a*b, a*a - b*b - c*c + d*d)
        theta = np.arcsin((-2*b*d + 2*a*c))
        psi = np.arctan2(2*b*c + 2*a*d, a*a + b*b - c*c - d*d)
        return np.array([phi, theta, psi])

    def to_vector_rotation(self):
        raise NotImplementedError

    def __str__(self):
        return self.quaternion_vector.__str__()

    def __repr__(self):
        return self.quaternion_vector.__repr__()


def skew_operator(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def rot_mat_to_euler(mat):
    phi = np.arctan2(mat[2, 1], mat[2, 2])
    theta = np.arcsin(-mat[2, 0])
    psi = np.arctan2(mat[1, 0], mat[0, 0])
    return np.array([phi, theta, psi])


def euler_to_rot_mat(euler):
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]
    cos = np.cos
    sin = np.sin
    return np.array([[cos(theta)*cos(psi),
                      sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),
                      cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],
                     [cos(theta)*sin(psi),
                      sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),
                      cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)],
                     [-sin(theta),
                      sin(phi)*cos(theta),
                      cos(phi)*cos(theta)]], dtype='float64')


def rot_mat_to_quaternion(mat):
    r11 = mat[0, 0]
    r12 = mat[0, 1]
    r13 = mat[0, 2]
    r21 = mat[1, 0]
    r22 = mat[1, 1]
    r23 = mat[1, 2]
    r31 = mat[2, 0]
    r32 = mat[2, 1]
    r33 = mat[2, 2]

    T = 1 + np.trace(mat)

    if T > 1e-8:
        S = 2*np.sqrt(T)
        a = S/4
        b = (r32 - r23)/S
        c = (r13 - r31)/S
        d = (r21 - r12)/S
    else:
        if r11 > r22 and r11 > r33:
            S = 2*np.sqrt(1 + r11 - r22 - r33)
            a = (r23 - r32)/S
            b = -S/4
            c = (r21 - r12)/S
            d = (r13 - r31)/S
        elif r22 > r33:
            S = 2*np.sqrt(1 - r11 + r22 - r33)
            a = (r31 - r13)/S
            b = (r21 - r12)/S
            c = -S/4
            d = (r32 - r23)/S
        else:
            S = 2*np.sqrt(1 - r11 - r22 + r33)
            a = (r12 - r21)/S
            b = (r13 - r31)/S
            c = (r32 - r23)/S
            d = -S/4
    return Quaternion(a, b, c, d)


def rot_vec_to_rot_mat(vector):
    theta = np.linalg.norm(vector)
    if theta == 0:
        R = np.eye(3)
    else:
        u = vector/theta
        omega = skew_operator(u)
        R = np.eye(3) + np.sin(theta)*omega +\
            (1-np.cos(theta))*omega.dot(omega)
    return R


def euler_to_quaternion(euler):
    return rot_mat_to_quaternion(euler_to_rot_mat(euler))
