#Verification Code

import numpy as np

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


def n_sensed(q):
    n1 = q[0]/(np.sin(np.arccos(q[3])))
    n2 = q[1]/(np.sin(np.arccos(q[3])))
    n3 = q[2]/(np.sin(np.arccos(q[3])))
    return np.array([[n1], [n2], [n3]])


def pos(phi, theta, psi, n_sensed):
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(-phi), -np.sin(-phi)],
                      [0, np.sin(-phi), np.cos(-phi)]])

    y_rot = np.array([[np.cos(-theta), 0, np.sin(-theta)],
                      [0, 1, 0],
                      [-np.sin(-theta), 0, np.cos(-theta)]])

    z_rot = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
    
    n_true = np.dot(np.dot(x_rot, np.dot(y_rot, z_rot)), n_sensed)
    
    lat = np.pi/2 - np.arccos(n_true[2])
    
    lon = np.arctan(n_true[1]/n_true[0])
    
    return print(f'true position: {n_true}'), print(f'latitude: {lat}'), print(f'longitude: {lon}')


def n_true(phi, theta, psi, n_sensed):
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(-phi), -np.sin(-phi)],
                      [0, np.sin(-phi), np.cos(-phi)]])

    y_rot = np.array([[np.cos(-theta), 0, np.sin(-theta)],
                      [0, 1, 0],
                      [-np.sin(-theta), 0, np.cos(-theta)]])

    z_rot = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
    
    n_true = np.dot(np.dot(x_rot, np.dot(y_rot, z_rot)), n_sensed)
   
    return n_true


#Verification definitions
def transform(phi, theta, psi, init_pos):
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


def init_coordinates(init_pos):                                                 #initial coordinates in terms of latitude and longitude
    latitude = np.arcsin(init_pos[2])
    longitude = np.arctan(init_pos[1]/init_pos[0])
    return print(f'initial lattitude: {latitude}'), print(f'initial longitude: {longitude}')
  

translate = np.random.rand(3, 1)
#init_pos = translate/np.linalg.norm(translate)                                  #initial 3x1 position vector
init_pos = np.array([[1], [1], [1]])

rand_1 = np.random.rand(3, 1)                                                   #random 3x1 vector
rand_2 = np.random.rand(3, 1)                                                   #random 3x1 vector

T = transform(0, 0, 0, init_pos)                                               #4x4 homogeneous transformation vector

v1i = rand_1/np.linalg.norm(rand_1)                                             #vector 1 expressed in inertial frame
v_1i = np.concatenate((v1i, np.array([[1]])), axis = 0)
v_1b = np.dot(T, v_1i)
v1b = np.array([v_1b[0], v_1b[1], v_1b[2]])                                     #vector 1 expressed in body frame

v2i = rand_2/np.linalg.norm(rand_2)                                             #vector 2 expressed in inertial frame
v_2i = np.concatenate((v2i, np.array([[1]])), axis = 0)
v_2b = np.dot(T, v_2i)
v2b = np.array([v_2b[0], v_2b[1], v_2b[2]])                                     #vector 2 expressed in body frame

w = 1                                                                           #weighted scalar

#Q-Method
B1 = B(w, v1b, v1i)
B2 = B(w, v2b, v2i)
B = B1 + B2

K = K(B)

q = q(K)

n = n_sensed(q)


N = n_true(0, 0, 0, n)





