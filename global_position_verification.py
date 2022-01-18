
#Verification Code

import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Davenport Q-method
def B(w, vb, vi):
    return (w*(vb*np.transpose(vi)))

def K(B):
    S = B + np.transpose(B)
    Z = np.array([[B[1, 2] - B[2, 1]], 
                  [B[2, 0] - B[0, 2]],
                  [B[0, 1] - B[1, 0]]])
    sigma = np.array([[B.trace()]])
    top_right = S - sigma*np.identity(3)
    K1 = np.concatenate((top_right, np.transpose(Z)), axis = 0)
    K2 = np.concatenate((Z, sigma), axis = 0)
    K = np.concatenate((K1, K2), axis = 1)
    return K

def q(K):                               
    eigval, eigvec = np.linalg.eig(K)
    if np.max(eigval) == eigval[0]:
        return eigvec[:, 0]
    elif np.max(eigval) == eigval[1]:
        return eigvec[:, 1]
    elif np.max(eigval) == eigval[2]:
        return eigvec[:, 2] 
    elif np.max(eigval) == eigval[3]:
        return eigvec[:, 3]



#Sigel and Wettergreen
def n_sensed(q):                                                                #normal vector extraction from optimal quaternion 
    n1 = q[0]/(np.sin(np.arccos(q[3]*(np.pi/180))))
    n2 = q[1]/(np.sin(np.arccos(q[3]*(np.pi/180))))
    n3 = q[2]/(np.sin(np.arccos(q[3]*(np.pi/180))))
    return np.array([[n1], [n2], [n3]])

def n_true(phi, theta, psi, n_sensed):                                          #corrected normal vector 
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(-phi*(np.pi/180)), -np.sin(-phi*(np.pi/180))],
                      [0, np.sin(-phi*(np.pi/180)), np.cos(-phi*(np.pi/180))]])
    y_rot = np.array([[np.cos(-theta*(np.pi/180)), 0, np.sin(-theta*(np.pi/180))],
                      [0, 1, 0],
                      [-np.sin(-theta*(np.pi/180)), 0, np.cos(-theta*(np.pi/180))]])
    z_rot = np.array([[np.cos(psi*(np.pi/180)), -np.sin(psi*(np.pi/180)), 0],
                      [np.sin(psi*(np.pi/180)), np.cos(psi*(np.pi/180)), 0],
                      [0, 0, 1]])
    
    n_true = np.dot(np.dot(x_rot, np.dot(y_rot, z_rot)), n_sensed)
   
    return n_true

def pos(phi, theta, psi, n_sensed):                                             #position estimate
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(-phi*(np.pi/180)), -np.sin(-phi*(np.pi/180))],
                      [0, np.sin(-phi*(np.pi/180)), np.cos(-phi*(np.pi/180))]])
    y_rot = np.array([[np.cos(-theta*(np.pi/180)), 0, np.sin(-theta*(np.pi/180))],
                      [0, 1, 0],
                      [-np.sin(-theta*(np.pi/180)), 0, np.cos(-theta*(np.pi/180))]])
    z_rot = np.array([[np.cos(psi*(np.pi/180)), -np.sin(psi*(np.pi/180)), 0],
                      [np.sin(psi*(np.pi/180)), np.cos(psi*(np.pi/180)), 0],
                      [0, 0, 1]])

    r = 6371 #approximate radius of celestial body "Earth" [m]
    n_true = np.dot(np.dot(x_rot, np.dot(y_rot, z_rot)), n_sensed)
    #lat = np.pi/2 - np.arccos(n_true[2]*(np.pi/180))
    #lon = np.arctan(n_true[1]/n_true[0]*((np.pi/180)))
    lat = m.degrees(m.sin(n_true[2]))  #(np.pi/180)))
    lon = m.degrees(np.pi/2) - m.degrees(m.atan2(n_true[1],n_true[0]))
    return print(f'latitude: {lat}'), print(f'longitude: {lon}')

#Verification definitions
def transform(phi, theta, psi, init_pos):                                       #homogeneous transformation matrix 4x4
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])

    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    top = np.concatenate((R, init_pos), axis = 1)
    bottom = np.array([[0, 0, 0, 1]])
    T = np.concatenate((top, bottom), axis = 0)
    return T

def check_vec(r, v1b, v1i, v2b, v2i):                                           #code verifies vector addition between frames
    a = v1b - v1i                                                               
    b = v2b - v2i
    if a.all() == r.all() and b.all() == r.all():                   
        print('true')
    if a.all() == r.all() and b.all() != r.all():
        print('a == r1 but b != r2')
    if a.all() == r.all() and b.all() != r.all():
        print('a != r1 but b == r2')
    if a.all() != r.all() or b.all() != r.all():
        print('false')



#r = np.random.rand(3, 1)/np.linalg.norm(np.random.rand(3, 1))                   #radius vector
r = np.random.uniform(low = -5, high = 5, size = (3,1))

T = transform(0, 0, 0, r)                                                       #4x4 homogeneous transformation vector

#v1i = np.random.rand(3, 1)                                                      #vector 1 expressed in inertial frame
v1i = np.random.uniform(low = -5, high = 5, size = (3,1))
v_1i = np.concatenate((v1i, np.array([[1]])), axis = 0)                         #turn 3x1 matrix into 4x1 matrix
v_1b = np.dot(T, v_1i)                                                          #intermidiate step to transform frames of ref
v1b = np.array([v_1b[0], v_1b[1], v_1b[2]])                                     #vector 1 expressed in body frame

#v2i = np.random.rand(3, 1)                                                      #vector 2 expressed in inertial frame
v2i = np.random.uniform(low = -5, high = 5, size = (3,1))
v_2i = np.concatenate((v2i, np.array([[1]])), axis = 0)                         #turn 3x1 matrix into 4x1 matrix
v_2b = np.dot(T, v_2i)                                                          #intermidiate step to transform frames of ref
v2b = np.array([v_2b[0], v_2b[1], v_2b[2]])                                     #vector 2 expressed in body frame

w = 1                                                                           #weighted scalar

#Q-Method
B1 = B(w, v1b, v1i)
B2 = B(w, v2b, v2i)
B = B1 + B2

K = K(B)

q = q(K)

n = n_sensed(q)

P = pos(0, 0, 0, n)

ntrue = n_true(0, 0, 0, n)
                
check = check_vec(r, v1b, v1i, v2b, v2i)
        

def EulerRot(phi, theta, psi):
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(-phi*(np.pi/180)), -np.sin(-phi*(np.pi/180))],
                      [0, np.sin(-phi*(np.pi/180)), np.cos(-phi*(np.pi/180))]])
    y_rot = np.array([[np.cos(-theta*(np.pi/180)), 0, np.sin(-theta*(np.pi/180))],
                      [0, 1, 0],
                      [-np.sin(-theta*(np.pi/180)), 0, np.cos(-theta*(np.pi/180))]])
    z_rot = np.array([[np.cos(psi*(np.pi/180)), -np.sin(psi*(np.pi/180)), 0],
                      [np.sin(psi*(np.pi/180)), np.cos(psi*(np.pi/180)), 0],
                      [0, 0, 1]])
    
    R = np.dot(z_rot, np.dot(y_rot, x_rot))
    
    return R
